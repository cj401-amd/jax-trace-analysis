#!/usr/bin/env python3
"""Why did a JAX program run out of GPU memory? Answers from its jax.profiler capture (.xplane.pb).

usage: oom_report.py CAPTURE_DIR_or_XPLANE.pb [--module NAME] [--top N] [--json OUT.json]
       oom_report.py --self-test

Standard library only. Two views, both read from the .xplane.pb:
  runtime  the BFC allocator's MemoryAllocation / MemoryDeallocation events: capacity, bytes in use
           at the start of each execution of each XLA module, the peak, and what was free at the end.
  plan     each module's buffer assignment (the HloProto XProf keeps in /host:metadata): parameters
           (donated or not), outputs, temp block, the heap peak (as XProf's Memory Viewer replays it),
           and the buffers live at that peak with role (forward / backward / other), op and source line.
Then a diagnosis: the step alone doesn't fit / bytes in use grow every step / other arrays take the room.
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics
import struct
import sys

MIB = 1 << 20
GIB = 1 << 30


# ---- protobuf wire format (no protoc; field numbers from xplane.proto and hlo.proto) ----------
def varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if b < 0x80:
            return result, i
        shift += 7


def fields(buf):
    i = 0
    while i < len(buf):
        key, i = varint(buf, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            val, i = varint(buf, i)
        elif wire == 2:
            n, i = varint(buf, i)
            val, i = buf[i:i + n], i + n
        elif wire == 1:
            val, i = buf[i:i + 8], i + 8
        elif wire == 5:
            val, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"wire type {wire}")
        yield num, val


def message(buf, names):
    """names: {field_number: (name, repeated)}."""
    out = {}
    for num, val in fields(buf):
        if num in names:
            name, repeated = names[num]
            if repeated:
                out.setdefault(name, []).append(val)
            else:
                out[name] = val
    return out


def packed(vals):
    out = []
    for v in vals:
        if isinstance(v, bytes):
            i = 0
            while i < len(v):
                x, i = varint(v, i)
                out.append(x)
        else:
            out.append(v)
    return out


def signed(v):
    return v - (1 << 64) if v >= 1 << 63 else v


# ---- XSpace: HloProtos, allocator events, executions -----------------------------------------
def load_xspace(path):
    hlo = {}                              # "jit_f(123)" -> HloProto bytes
    allocs = collections.defaultdict(list)  # allocator -> [[t_ps, kind, in_use_after, requested, capacity,
    #                                                       addr, chunk bytes, allocating host event, its start]]
    execs = []                            # (t_ps, end_ps, module)
    thunks = []                           # (t_ps, module): GpuExecutable::ExecuteThunks, i.e. kernels launched
    retries = []                          # (t_ps, dur_ps): AllocatorRetry::AllocateRaw (BFC waiting for frees)
    between = {}                          # module -> top-level event names between its last two executions
    for plane in message(open(path, "rb").read(), {1: ("planes", True)})["planes"]:
        p = message(plane, {2: ("name", False), 3: ("lines", True), 4: ("em", True), 5: ("sm", True)})
        name = p.get("name", b"").decode()
        if name.startswith("/device:"):
            continue
        em, sm, em_stats = {}, {}, {}
        for e in p.get("em", []):
            m = message(message(e, {2: ("v", False)}).get("v", b""), {1: ("id", False), 2: ("name", False),
                                                                    5: ("stats", True)})
            em[m.get("id", 0)] = m.get("name", b"").decode()
            em_stats[m.get("id", 0)] = m.get("stats", [])
        if name == "/host:metadata":
            for i, n in em.items():
                for st in em_stats[i]:
                    b = message(st, {6: ("bytes", False)}).get("bytes")
                    if b:
                        hlo[n] = b
            continue
        for e in p.get("sm", []):
            m = message(message(e, {2: ("v", False)}).get("v", b""), {1: ("id", False), 2: ("name", False)})
            sm[m.get("id", 0)] = m.get("name", b"").decode()
        mem_ids = {i for i, n in em.items() if n in ("MemoryAllocation", "MemoryDeallocation")}
        # One top-level Execute per call; SPMD adds a "[k] ...Execute (m)" per device, which is skipped.
        exe_ids = {i: m.group(1) for i, n in em.items() for m in [re.match(r"[^\[]\S*::Execute \((.+)\)$", n)] if m}
        thunk_ids = {i for i, n in em.items() if n.endswith("GpuExecutable::ExecuteThunks")}
        retry_ids = {i for i, n in em.items() if n.startswith("AllocatorRetry::AllocateRaw")}
        if not (mem_ids or exe_ids):
            continue
        for line in p.get("lines", []):
            ln = message(line, {3: ("ts", False), 4: ("ev", True)})
            base = ln.get("ts", 0) * 1000
            spans, line_execs, mems = [], [], []
            for ev in ln.get("ev", []):
                e = message(ev, {1: ("mid", False), 2: ("off", False), 3: ("dur", False), 4: ("st", True)})
                mid = e.get("mid", 0)
                spans.append((base + e.get("off", 0), e.get("dur", 0), mid))
                if mid in exe_ids:
                    line_execs.append((base + e.get("off", 0), e.get("dur", 0), exe_ids[mid]))
                if mid in retry_ids:
                    retries.append((base + e.get("off", 0), e.get("dur", 0)))
                if mid not in mem_ids and mid not in exe_ids and mid not in thunk_ids:
                    continue
                t = base + e.get("off", 0)
                st = {}
                for s in e.get("st", []):
                    x = message(s, {1: ("m", False), 3: ("u", False), 4: ("i", False), 5: ("s", False), 7: ("r", False)})
                    k = sm.get(x.get("m"), "")
                    if "r" in x:
                        st[k] = sm.get(x["r"], "")
                    elif "s" in x:
                        st[k] = x["s"].decode()
                    else:
                        st[k] = signed(x.get("i", x.get("u", 0)))
                if mid in exe_ids:
                    execs.append((t, t + e.get("dur", 0), exe_ids[mid]))
                elif mid in thunk_ids:
                    thunks.append((t, st.get("module_name", "?")))
                else:
                    used = st.get("bytes_allocated", 0)
                    rec = [t, "alloc" if em[mid] == "MemoryAllocation" else "free", used, st.get("requested_bytes", 0),
                           used + st.get("bytes_available", 0), st.get("addr", 0), st.get("allocation_bytes", 0), "", 0]
                    allocs[st.get("allocator_name", "?")].append(rec)
                    mems.append(rec)
            if mems:
                label_allocations(spans, mems, em, mem_ids)
            by_mod = collections.defaultdict(list)
            for t, d, m in line_execs:
                by_mod[m].append((t, d))
            for m, xs in by_mod.items():
                if len(xs) >= 2:
                    between[m] = top_level(spans, xs[-2][0] + xs[-2][1], xs[-1][0], em)
    for v in allocs.values():
        v.sort(key=lambda e: e[0])
    execs.sort()
    return hlo, dict(allocs), execs, sorted(thunks), sorted(retries), between


def label_allocations(spans, mems, em, mem_ids):
    """Labels each memory event with its innermost enclosing host event named like "X(...)": e.g.
    "CommonPjRtLoadedExecutable::Execute (jit_step)" or "BufferFromHostBuffer(s32[...])"."""
    todo = collections.defaultdict(list)
    for rec in mems:
        todo[rec[0]].append(rec)
    stack = []   # (end, name, start)
    for t, d, mid in sorted((t, -d, mid) for t, d, mid in spans):
        while stack and stack[-1][0] < t:
            stack.pop()
        if mid in mem_ids:
            for rec in todo.pop(t, []):
                rec[7], rec[8] = next(((n, s) for _, n, s in reversed(stack) if "(" in n), ("", 0))
            continue
        stack.append((t - d, em.get(mid, "?"), t))


def live_at(evs, t_end, execs):
    """Allocations live just before t_end, grouped by what allocated them (executions get their ordinal)."""
    live = {}
    for e in evs:
        if e[0] >= t_end:
            break
        if e[1] == "alloc":
            live[e[5]] = e
        else:
            live.pop(e[5], None)
    starts = collections.defaultdict(list)   # module -> top-level execution start times
    for t0, _, m in execs:
        starts[m].append(t0)
    groups = collections.defaultdict(lambda: [0, 0])
    for e in live.values():
        m = re.search(r"::Execute \((.+)\)$", e[7])
        k = _bisect(starts[m.group(1)], e[8] + 1) if m else 0   # SPMD per-device Executes start inside the call's
        label = f"outputs of {m.group(1)} execution {k}" if k else e[7] or "(no enclosing host event)"
        groups[label][0] += e[6] or e[3]
        groups[label][1] += 1
    return sorted(((k, b, n) for k, (b, n) in groups.items()), key=lambda g: -g[1])


def top_level(spans, a, b, em):
    """Names of the outermost events inside (a, b), in order: what the host did between two executions."""
    inside = sorted((t, -d, mid) for t, d, mid in spans if a <= t and t + d <= b)
    out, end = [], -1
    for t, nd, mid in inside:
        if t >= end:
            out.append(em.get(mid, "?"))
            end = t - nd
    return out


# ---- one module's buffer assignment ------------------------------------------------------------
_PRIM = {1: "pred", 2: "s8", 3: "s16", 4: "s32", 5: "s64", 6: "u8", 7: "u16", 8: "u32", 9: "u64", 10: "f16",
         11: "f32", 12: "f64", 13: "tuple", 15: "c64", 16: "bf16", 17: "token", 18: "c128", 19: "f8e5m2",
         20: "f8e4m3fn"}


def shape_tree(buf):
    m = message(buf, {2: ("et", False), 3: ("dims", True), 4: ("tuple", True)})
    if m.get("et") == 13:
        return [shape_tree(t) for t in m.get("tuple", [])]
    return f"{_PRIM.get(m.get('et', 0), m.get('et', 0))}[{','.join(map(str, packed(m.get('dims', []))))}]"


def subshape(tree, index):
    for i in index:
        if isinstance(tree, list) and i < len(tree):
            tree = tree[i]
    return tree if isinstance(tree, str) else "(tuple)"


def role(op_name, opcode, shape):
    """Coarse role of a buffer, from the JAX name stack in its op_name."""
    if opcode == "custom-call" and shape.startswith("s8["):
        return "workspace (library scratch)"
    if "transpose(" in op_name:
        return "backward (gradients and their temporaries)"
    if "jvp(" in op_name:
        return "forward, kept for the backward pass"
    parts = [x for x in op_name.split("/")[1:] if x]
    return f"other: {parts[0]}" if len(parts) > 1 else "other"


def stack_frames(buf):
    """StackFrameIndexProto -> {frame id: "file:line (function)"} for the innermost frame (ids are 1-based)."""
    s = message(buf, {1: ("files", True), 2: ("funcs", True), 3: ("locs", True), 4: ("frames", True)})
    files = [x.decode() for x in s.get("files", [])]
    funcs = [x.decode() for x in s.get("funcs", [])]
    out = {}
    for fid, fr in enumerate(s.get("frames", []), 1):
        loc = message(s["locs"][message(fr, {1: ("loc", False)}).get("loc", 1) - 1],
                      {1: ("file", False), 2: ("func", False), 3: ("line", False)})
        out[fid] = (f"{os.path.basename(files[loc.get('file', 1) - 1])}:{loc.get('line', 0)} "
                    f"({funcs[loc.get('func', 1) - 1]})")
    return out


def load_plan(blob):
    hp = message(blob, {1: ("module", False), 3: ("ba", False)})
    if "ba" not in hp:
        return None
    mod = message(hp["module"], {3: ("comps", True), 17: ("sfi", False)})
    frames = stack_frames(mod.get("sfi", b""))
    instrs = {}   # id -> (name, opcode, shape tree, op_name, source)
    remat = 0
    for comp in mod.get("comps", []):
        for ins in message(comp, {2: ("ins", True)}).get("ins", []):
            m = message(ins, {1: ("name", False), 2: ("opcode", False), 3: ("shape", False), 7: ("md", False),
                              35: ("id", False)})
            md = message(m.get("md", b""), {2: ("op_name", False), 3: ("file", False), 4: ("line", False),
                                             15: ("frame", False)})
            name = m.get("name", b"").decode()
            remat += ".remat" in name
            src = frames.get(md.get("frame", 0)) or (f"{md['file'].decode()}:{md.get('line', 0)}" if md.get("file") else "")
            instrs[m.get("id", 0)] = (name, m.get("opcode", b"").decode(), shape_tree(m.get("shape", b"")),
                                      md.get("op_name", b"").decode(), src)
    ba = message(hp["ba"], {1: ("lbs", True), 3: ("allocs", True), 4: ("traces", True)})
    lbs = {}
    for lb in ba.get("lbs", []):
        m = message(lb, {1: ("id", False), 2: ("size", False), 3: ("at", False), 4: ("color", False)})
        at = message(m.get("at", b""), {3: ("idx", True), 4: ("iid", False)})
        name, opcode, tree, op_name, src = instrs.get(at.get("iid", -1), ("?", "", "?", "", ""))
        idx = packed(at.get("idx", []))
        shape = subshape(tree, idx)
        if idx:
            name += "{" + ",".join(map(str, idx)) + "}"
        lbs[m.get("id", 0)] = dict(size=m.get("size", 0), color=m.get("color", 0), instr=name, opcode=opcode,
                                   shape=shape, op_name=op_name, source=src, role=role(op_name, opcode, shape))
    t = collections.Counter()
    outputs = []
    for al in ba.get("allocs", []):
        m = message(al, {2: ("size", False), 3: ("tl", False), 5: ("param", False), 7: ("out", False),
                         8: ("color", False), 12: ("const", False), 9: ("asg", True)})
        size = m.get("size", 0)
        if m.get("color", 0) != 0:
            t[f"memory space {m['color']}"] += size
        elif m.get("tl") or m.get("const"):
            t["constants and thread-local"] += size
        elif m.get("param"):
            t["parameters"] += size
            t["parameters donated (aliased with an output)"] += size if m.get("out") else 0
        elif m.get("out"):
            t["outputs (not aliased)"] += size
            ids = [message(x, {1: ("id", False)}).get("id", 0) for x in m.get("asg", [])]
            first = max((lbs.get(i, {}) for i in ids), key=lambda b: b.get("size", 0), default={})
            outputs.append(dict(size=size, shape=first.get("shape", "?"), instr=first.get("instr", "?"),
                                op_name=first.get("op_name", ""), source=first.get("source", "")))
        else:
            t["temp"] += size
    peak, live = replay_heap(ba.get("traces", []), lbs)
    return dict(totals=dict(t), heap_peak=peak, live=live, outputs=sorted(outputs, key=lambda o: -o["size"]),
                remat=remat, need=t["parameters"] + t["outputs (not aliased)"] + t["temp"] +
                t["constants and thread-local"])


def replay_heap(traces, lbs):
    """Replays the heap-simulator trace with the most events (as XProf's Memory Viewer does).

    Returns (peak bytes, buffers live at the peak); buffers that SHARE_WITH another count once. A shared
    buffer takes its role from the member with a JAX name stack: a scan's stacked activations are created
    by a broadcast (no name stack) and shared with the while loop that fills them (jvp()/while)."""
    best = max((message(tr, {1: ("ev", True)}).get("ev", []) for tr in traces), key=len, default=[])
    events = [message(e, {1: ("kind", False), 2: ("buf", False), 5: ("canon", False)}) for e in best]

    def run(stop):
        heap = peak = peak_i = 0
        canon, refs, members = {}, collections.Counter(), collections.defaultdict(set)
        for i, e in enumerate(events[:stop]):
            b, kind = e.get("buf", 0), e.get("kind", 0)
            if kind == 0:                                   # ALLOC
                canon[b], refs[b] = b, 1
                members[b].add(b)
                heap += lbs.get(b, {}).get("size", 0)
            elif kind == 2:                                 # SHARE_WITH
                c = canon.get(e.get("canon", 0), e.get("canon", 0))
                canon[b] = c
                refs[c] += 1
                members[c].add(b)
                if refs[c] == 1:
                    heap += lbs.get(c, {}).get("size", 0)
            elif kind == 1:                                 # FREE
                c = canon.get(b, b)
                refs[c] -= 1
                members[c].discard(b)
                if refs[c] == 0:
                    heap -= lbs.get(c, {}).get("size", 0)
            if heap > peak:
                peak, peak_i = heap, i + 1
        return peak, peak_i, refs, members

    peak, peak_i, _, _ = run(len(events))
    _, _, refs, members = run(peak_i)
    rank = lambda b: (not b["role"].startswith("other"), not b["role"].startswith("workspace"))
    live = []
    for c, n in refs.items():
        if n > 0 and c in lbs:
            # Keep the buffer's own role; borrow an alias's only when its own has no JAX name stack.
            rep = lbs[c] if rank(lbs[c]) == (True, True) else max(
                (lbs[m] for m in sorted(members[c]) if m in lbs), key=rank, default=lbs[c])
            live.append(dict(rep, size=lbs[c]["size"], id=c, aliases=len(members[c]) - 1))
    return peak, sorted(live, key=lambda b: -b["size"])


# ---- runtime: bytes in use per execution -------------------------------------------------------
def runtime(allocs, execs, thunks):
    out = {}
    for name, evs in allocs.items():
        times = [e[0] for e in evs]
        cap = max(e[4] for e in evs)
        peak_e = max(evs, key=lambda e: e[2])
        per_exec = []
        for k, (t0, _, mod) in enumerate(execs):
            t1 = execs[k + 1][0] if k + 1 < len(execs) else float("inf")
            i0 = _bisect(times, t0)
            before = evs[i0 - 1][2] if i0 else 0
            window = [e for e in evs[i0:] if e[0] < t1]
            if not window and not i0:
                continue
            per_exec.append(dict(module=mod, in_use_at_start=before,
                                 peak_until_next=max([before] + [e[2] for e in window])))
        out[name] = dict(capacity=cap, peak=peak_e[2], peak_time_ps=peak_e[0], events=len(evs),
                         in_use_at_end=evs[-1][2], executions=per_exec)
    counts = collections.Counter(m for _, _, m in execs)
    ok = collections.Counter(m for t0, t1, m in execs if any(t0 <= t <= t1 and tm == m for t, tm in thunks))
    last_ok = {m: any(t0 <= t <= t1 and tm == m for t, tm in thunks) for t0, t1, m in execs}   # last one wins
    return out, {m: dict(executions=n, launched=ok[m] if thunks else None, last_launched=last_ok[m] if thunks else None)
                 for m, n in counts.items()}


def _bisect(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def growth(starts):
    """Median step-to-step change of bytes in use at execution start, if it is steady and positive."""
    d = [b - a for a, b in zip(starts[1:], starts[2:])]   # skip the first: it includes setup
    if len(d) < 2:
        return None
    med = statistics.median(d)
    steady = sum(abs(x - med) <= max(0.1 * abs(med), MIB) for x in d) >= 0.7 * len(d)
    return med if med > MIB and steady else None


# ---- report ------------------------------------------------------------------------------------
def analyze(path, module=None, top=15):
    hlo, allocs, execs, thunks, retries, between = load_xspace(path)
    plans = {}
    for name, blob in hlo.items():
        p = load_plan(blob)
        if p:
            plans[name] = p
    rt, exec_counts = runtime(allocs, execs, thunks)
    base = lambda n: re.sub(r"\(\d+\)$", "", n)
    if module is None:   # the module whose last execution never launched; else the biggest that ran; else the biggest
        failed = [n for n in plans if exec_counts.get(base(n), {}).get("last_launched") is False]
        ran = [n for n in plans if base(n) in exec_counts]
        cands = failed or ran or list(plans)
        module = max(cands, key=lambda n: plans[n]["need"]) if cands else None
    gpu = {k: v for k, v in rt.items() if "gpu" in k.lower() and "host" not in k.lower()} or rt
    r = dict(file=path, module=module, plans=plans, runtime=rt, executions=exec_counts, diagnosis=[],
             gpu_allocators=sorted(gpu), between=between, allocator_retries=[dict(time_ps=t, seconds=d / 1e12) for t, d in retries])
    if retries:
        t, d = max(retries, key=lambda x: x[1])
        r["diagnosis"].append(f"AllocatorRetry::AllocateRaw: {len(retries)} events, longest {d / 1e12:.2f} s: the BFC "
                              "allocator found no room and waited for frees; a long wait ends in an OOM [measured]")
    if module is None:
        r["diagnosis"].append("no buffer assignment in /host:metadata: the capture has no plan to read")
        return r
    plan = plans[module]
    mod_execs = {k: [e for e in v["executions"] if e["module"] == base(module)] for k, v in gpu.items()}
    dx = r["diagnosis"]
    # SPMD GPUs look alike; report the tightest one (highest peak / capacity).
    tight = max(gpu, key=lambda k: gpu[k]["peak"] / max(gpu[k]["capacity"], 1)) if gpu else None
    if len(gpu) > 1:
        dx.append(f"{len(gpu)} GPU allocators; the lines below are for the tightest, {tight} (peak "
                  f"{gpu[tight]['peak'] / GIB:.2f} of {gpu[tight]['capacity'] / GIB:.2f} GiB) [measured]")
    for alloc, v in [(tight, gpu[tight])] if gpu else []:
        cap, ex = v["capacity"], mod_execs[alloc]
        if plan["need"] > cap:
            dx.append(f"{alloc}: {module} alone needs {plan['need'] / GIB:.2f} GiB (params + outputs + temp) "
                      f"> capacity {cap / GIB:.2f} GiB: the step itself doesn't fit [measured]")
        g = growth([e["in_use_at_start"] for e in ex])
        if g:
            outs = plan["totals"].get("outputs (not aliased)", 0)
            note = (f"; that is {g / max(outs, 1):.2f}x the module's non-aliased outputs ({outs / MIB:.1f} MiB): "
                    "the program keeps a step's outputs alive [inferred]") if outs and abs(g - outs) <= 0.05 * outs else ""
            dx.append(f"{alloc}: bytes in use at the start of {base(module)} grow by {g / MIB:.1f} MiB per execution "
                      f"({ex[0]['in_use_at_start'] / MIB:.0f} -> {ex[-1]['in_use_at_start'] / MIB:.0f} MiB over "
                      f"{len(ex)} executions) [measured]{note}")
        if ex:
            t_last = max(t0 for t0, _, m in execs if m == base(module))
            r["live_at_last_start"] = dict(allocator=alloc, in_use=ex[-1]["in_use_at_start"],
                                           groups=live_at(allocs[alloc], t_last, execs))
            free = cap - ex[-1]["in_use_at_start"]
            want = plan["totals"].get("temp", 0) + plan["totals"].get("outputs (not aliased)", 0)
            verdict = "does not fit" if want > free else "fits"
            dx.append(f"{alloc}: last {base(module)} execution started with {ex[-1]['in_use_at_start'] / MIB:.0f} MiB "
                      f"in use, {free / MIB:.0f} MiB free; it needs temp + new outputs = {want / MIB:.0f} MiB: {verdict} "
                      "[measured; BFC also needs a contiguous chunk]")
            held = ex[-1]["in_use_at_start"] - plan["totals"].get("parameters", 0)
            if held > 0.25 * cap and not g and plan["need"] <= cap:
                top_g = max((x for x in r["live_at_last_start"]["groups"] if not x[0].startswith(f"outputs of {base(module)}")),
                            key=lambda x: x[1], default=None)
                dx.append(f"{alloc}: {held / MIB:.0f} MiB in use beyond {base(module)}'s parameters: other live arrays "
                          "take the room [inferred]" + (f"; the largest, {top_g[1] / MIB:.1f} MiB, came from {top_g[0]} "
                                                         "[measured]" if top_g else ""))
    t = plan["totals"]
    outs, params = t.get("outputs (not aliased)", 0), t.get("parameters", 0)
    if not t.get("parameters donated (aliased with an output)") and outs >= MIB and outs >= 0.5 * params:
        dx.append(f"{module}: no parameter is donated, and {outs / MIB:.0f} MiB of new outputs sit next to "
                  f"{params / MIB:.0f} MiB of parameters: old and new state are both live; donating the state "
                  f"(donate_argnums) can save up to {min(outs, params) / MIB:.0f} MiB [inferred]")
    c = exec_counts.get(base(module), {})
    if c.get("last_launched") is False:
        dx.append(f"{base(module)}: {c['executions']} executions started, {c['launched']} launched kernels; the last one "
                  "launched none: it stopped while getting its buffers (an OOM there), or the capture ended first [inferred]")
    if plan["remat"]:
        dx.append(f"{module}: XLA already rematerialized {plan['remat']} instructions (*.remat*) to fit its "
                  "memory limit [measured]")
    return r


def report(r, top=15):
    w = print
    w(f"== {r['file']}")
    w("\n-- runtime: allocators (MemoryAllocation / MemoryDeallocation events)")
    for name, v in sorted(r["runtime"].items()):
        w(f"{name:28s} capacity {v['capacity'] / GIB:8.2f} GiB   peak {v['peak'] / GIB:8.2f} GiB "
          f"({100 * v['peak'] / max(v['capacity'], 1):5.1f}%)   in use at end {v['in_use_at_end'] / MIB:9.1f} MiB   "
          f"events {v['events']}")
    w("\n-- runtime: executions per module (Execute events; 'launched' = reached ExecuteThunks)")
    for m, c in sorted(r["executions"].items(), key=lambda kv: -kv[1]["executions"]):
        w(f"  {m:40s} executions {c['executions']:4d}   launched {c['launched'] if c['launched'] is not None else '?'}")
    if r["module"]:
        b = re.sub(r"\(\d+\)$", "", r["module"])
        for name in r["gpu_allocators"]:
            ex = [e for e in r["runtime"][name]["executions"] if e["module"] == b]
            if ex:
                w(f"\n-- runtime: {name}, each {b} execution: MiB in use at start / peak before the next execution")
                w("  " + "  ".join(f"{e['in_use_at_start'] / MIB:.0f}/{e['peak_until_next'] / MIB:.0f}" for e in ex))
    if r["module"] and r["between"].get(re.sub(r"\(\d+\)$", "", r["module"])):
        names = r["between"][re.sub(r"\(\d+\)$", "", r["module"])]
        runs = [[names[0], 1]] if names else []
        for n in names[1:]:
            if n == runs[-1][0]:
                runs[-1][1] += 1
            else:
                runs.append([n, 1])
        w(f"\n-- runtime: top-level host events between the last two executions ({len(names)}; Python calls "
          "if the Python tracer was on)")
        w("  " + "  ->  ".join(f"{n[:60]}" + (f" x{k}" if k > 1 else "") for n, k in runs[:20]))
    la = r.get("live_at_last_start")
    if la:
        known = sum(b for _, b, _ in la["groups"])
        mod = re.sub(r"\(\d+\)$", "", r["module"])
        w(f"\n-- runtime: {la['allocator']}, live when the last {mod} execution "
          f"started ({la['in_use'] / MIB:.1f} MiB), by the host event that allocated it")
        for label, b, n in la["groups"][:12]:
            w(f"  {b / MIB:9.1f} MiB  x{n:<5d} {label[:110]}")
        if la["in_use"] - known > MIB:
            w(f"  {(la['in_use'] - known) / MIB:9.1f} MiB         allocated before the capture started")
    w("\n-- plans: modules by device need (params + non-aliased outputs + temp), MiB")
    w(f"  {'module':40s} {'need':>9s} {'params':>9s} {'donated':>9s} {'outputs':>9s} {'temp':>9s} {'heap peak':>9s}")
    for n, p in sorted(r["plans"].items(), key=lambda kv: -kv[1]["need"])[:8]:
        t = p["totals"]
        w(f"  {n:40s} {p['need'] / MIB:9.1f} {t.get('parameters', 0) / MIB:9.1f} "
          f"{t.get('parameters donated (aliased with an output)', 0) / MIB:9.1f} "
          f"{t.get('outputs (not aliased)', 0) / MIB:9.1f} {t.get('temp', 0) / MIB:9.1f} {p['heap_peak'] / MIB:9.1f}")
    if r["module"]:
        p = r["plans"][r["module"]]
        live = p["live"]
        w(f"\n-- plan of {r['module']}: temp block {p['totals'].get('temp', 0) / MIB:.1f} MiB, heap peak "
          f"{p['heap_peak'] / MIB:.1f} MiB, {len(live)} buffers live at the peak")
        by_role = collections.Counter()
        for x in live:
            by_role[x["role"]] += x["size"]
        for k, v in by_role.most_common():
            w(f"  {v / MIB:9.1f} MiB  {100 * v / max(p['heap_peak'], 1):5.1f}%  {k}")
        w("  by source line:")
        by_src = collections.Counter()
        for x in live:
            by_src[(x["source"] or "(no source)", x["role"].split(",")[0].split(" ")[0])] += x["size"]
        for (s, rl), v in by_src.most_common(10):
            w(f"  {v / MIB:9.1f} MiB  {rl:10s} {s}")
        w(f"  largest live buffers:")
        for x in live[:top]:
            name = x["instr"] + (f" +{x['aliases']}" if x.get("aliases") else "")
            w(f"  {x['size'] / MIB:9.1f} MiB  {x['shape'][:34]:34s} {name[:30]:30s} {x['source'] or '(no source)'}  | "
              f"{x['op_name'][-60:]}")
        if p["outputs"]:
            w(f"  largest non-aliased outputs ({len(p['outputs'])}):")
            for o in p["outputs"][:5]:
                w(f"  {o['size'] / MIB:9.1f} MiB  {o['shape'][:34]:34s} {o['instr'][:28]:28s} {o['op_name'][-70:]}")
    w("\n-- diagnosis")
    for d in r["diagnosis"]:
        w("  * " + d)


# ---- self-test: a synthetic capture through the real decoder -----------------------------------
def _v(n):
    out = bytearray()
    n &= (1 << 64) - 1
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _f(num, val):
    if isinstance(val, int):
        return _v(num << 3) + _v(val)
    return _v(num << 3 | 2) + _v(len(val)) + val


def _msg(*pairs):
    return b"".join(_f(n, v) for n, v in pairs if v is not None)


def _synthetic(steps, keep_per_step, temp, params, capacity, held=MIB):
    """XSpace with one module 'jit_step(1)' and `steps` executions; the last one fails if it can't fit."""
    def instr(i, name, opcode, op_name, dims):
        return _msg((1, name.encode()), (2, opcode.encode()), (3, _msg((2, 11), (3, _v(dims[0])))),
                    (7, _msg((2, op_name.encode()), (3, b"model.py"), (4, 10 + i))), (35, i))
    ins = [instr(1, "p0", "parameter", "", [params // 4]),
           instr(2, "act", "fusion", "jit(step)/jvp()/softmax", [temp // 8]),
           instr(3, "grad", "fusion", "jit(step)/transpose(jvp())/dot", [temp // 8]),
           instr(4, "out", "fusion", "jit(step)/transpose(jvp())/dot", [max(keep_per_step, 4) // 4]),
           instr(5, "stack", "fusion", "broadcast.1", [temp // 16]),
           instr(6, "while", "while", "jit(step)/jvp()/while", [temp // 16]),
           instr(7, "late", "fusion", "jit(step)/adam/mul", [temp // 16]),
           instr(9, "bwd_inplace", "fusion", "jit(step)/transpose(jvp())/add", [MIB // 4]),
           instr(10, "act2", "fusion", "jit(step)/jvp()/mul", [MIB // 4])]
    comp = _msg(*[(2, x) for x in ins])
    module = _msg((1, b"jit_step"), (3, comp))
    lb = lambda i, size: _msg((1, i), (2, size), (3, _msg((4, i))))
    heap = [_msg((1, 0), (2, 10)), _msg((1, 2), (2, 9), (5, 10)),   # forward buffer reused in place by backward
            _msg((1, 0), (2, 5)), _msg((1, 2), (2, 6), (5, 5)), _msg((1, 0), (2, 2)), _msg((1, 0), (2, 3)),
            _msg((1, 1), (2, 2)), _msg((1, 1), (2, 3)), _msg((1, 1), (2, 6)), _msg((1, 1), (2, 5)),
            _msg((1, 0), (2, 7)), _msg((1, 1), (2, 7)), _msg((1, 1), (2, 9)), _msg((1, 1), (2, 10))]   # allocated after the frees: the peak holds only if FREE works
    ba = _msg((1, lb(1, params)), (1, lb(2, temp // 4)), (1, lb(3, temp // 2)), (1, lb(4, keep_per_step)),
              (1, lb(5, temp // 4)), (1, lb(6, temp // 4)), (1, lb(7, temp // 4)), (1, lb(9, MIB)), (1, lb(10, MIB)),
              (3, _msg((1, 0), (2, params), (5, 1), (9, _msg((1, 1), (3, params))))),
              (3, _msg((1, 1), (2, temp), (9, _msg((1, 2))), (9, _msg((1, 3))), (9, _msg((1, 5))), (9, _msg((1, 6))), (9, _msg((1, 7))), (9, _msg((1, 9))), (9, _msg((1, 10))))),
              (3, _msg((1, 2), (2, keep_per_step), (7, 1), (9, _msg((1, 4))))),
              (4, _msg(*[(1, h) for h in heap])))
    hlo_proto = _msg((1, module), (3, ba))
    meta = _msg((1, 2), (2, b"/host:metadata"),
                (4, _msg((1, 1), (2, _msg((1, 1), (2, b"jit_step(1)"), (5, _msg((1, 1), (6, hlo_proto))))))))
    names = {1: "MemoryAllocation", 2: "MemoryDeallocation", 3: "CommonPjRtLoadedExecutable::Execute (jit_step)",
             4: "[0] GpuExecutable::ExecuteThunks", 5: "[0] CommonPjRtLoadedExecutable::Execute (jit_step)",
             6: "AllocatorRetry::AllocateRaw", 7: "PjRtCApiCompiler::BufferFromHostBuffer(s32[9])",
             8: "PjitFunction(step)"}
    stats = {1: "allocator_name", 2: "bytes_allocated", 3: "bytes_available", 4: "requested_bytes", 5: "GPU_0_bfc",
             6: "module_name", 7: "jit_step", 8: "addr", 9: "allocation_bytes"}
    evs, t, used = [], 0, params

    def mem(kind, req, addr):
        nonlocal used
        used += req if kind == 1 else -req
        return _msg((1, kind), (2, t * 1000), (4, _msg((1, 1), (7, 5))), (4, _msg((1, 2), (4, used))),
                    (4, _msg((1, 3), (4, capacity - used))), (4, _msg((1, 4), (4, req))), (4, _msg((1, 8), (4, addr))),
                    (4, _msg((1, 9), (4, req))))
    evs += [_msg((1, 7), (2, 0), (3, 500)), mem(1, held, 1)]   # an array put on the GPU before the loop
    for k in range(steps):
        t += 10
        evs.append(_msg((1, 8), (2, t * 1000 - 5), (3, 6000)))   # encloses the Execute: the label is the innermost
        evs.append(_msg((1, 3), (2, t * 1000), (3, 5000)))
        evs.append(_msg((1, 5), (2, t * 1000 + 10), (3, 4000)))   # SPMD per-device copy: not a second execution
        if used + temp + keep_per_step > capacity:
            evs.append(_msg((1, 6), (2, t * 1000 + 20), (3, 2 * 10**12)))
            break
        evs += [mem(1, temp, 100 + 2 * k), mem(1, keep_per_step, 101 + 2 * k)]
        evs.append(_msg((1, 4), (2, t * 1000 + 100), (4, _msg((1, 6), (7, 7)))))
        evs.append(mem(2, temp, 100 + 2 * k))
    em = b"".join(_f(4, _msg((1, i), (2, _msg((1, i), (2, n.encode()))))) for i, n in names.items())
    sm = b"".join(_f(5, _msg((1, i), (2, _msg((1, i), (2, n.encode()))))) for i, n in stats.items())
    cpu = _msg((1, 3), (2, b"/host:CPU"), (3, _msg((1, 1), (3, 0), *[(4, e) for e in evs]))) + em + sm
    return _msg((1, meta), (1, cpu))


def self_test():
    import tempfile
    d = tempfile.mkdtemp()
    # growth: 2 MiB kept per step, plan fits; runs until the pool is full.
    g = os.path.join(d, "g.xplane.pb")
    open(g, "wb").write(_synthetic(steps=40, keep_per_step=2 * MIB, temp=8 * MIB, params=16 * MIB, capacity=64 * MIB))
    r = analyze(g)
    p = r["plans"]["jit_step(1)"]
    assert p["totals"]["parameters"] == 16 * MIB and p["totals"]["temp"] == 8 * MIB, p["totals"]
    assert p["totals"]["outputs (not aliased)"] == 2 * MIB, p["totals"]
    assert p["heap_peak"] == 9 * MIB and len(p["live"]) == 4, (p["heap_peak"], p["live"])
    roles = {x["instr"]: x["role"] for x in p["live"]}
    assert roles == {"act": "forward, kept for the backward pass", "while": "forward, kept for the backward pass",
                     "act2": "forward, kept for the backward pass",
                     "grad": "backward (gradients and their temporaries)"}, roles
    assert r["executions"]["jit_step"]["executions"] == r["executions"]["jit_step"]["launched"] + 1, r["executions"]
    dx = " | ".join(r["diagnosis"])
    assert "grow by 2.0 MiB per execution" in dx and "keeps a step's outputs alive" in dx, dx
    assert "does not fit" in dx and "the last one launched none" in dx, dx
    assert "longest 2.00 s" in dx and r["executions"]["jit_step"]["executions"] == 20, (dx, r["executions"])
    groups = {g[0]: g[1] for g in r["live_at_last_start"]["groups"]}
    assert groups.pop("PjRtCApiCompiler::BufferFromHostBuffer(s32[9])") == MIB, groups
    assert groups == {f"outputs of jit_step execution {k}": 2 * MIB for k in range(1, 20)}, groups
    # an array held outside the step takes the room.
    h = os.path.join(d, "h.xplane.pb")
    open(h, "wb").write(_synthetic(steps=2, keep_per_step=MIB, temp=24 * MIB, params=16 * MIB, capacity=64 * MIB,
                                   held=30 * MIB))
    dx = " | ".join(analyze(h)["diagnosis"])
    assert "the largest, 30.0 MiB, came from PjRtCApiCompiler::BufferFromHostBuffer(s32[9])" in dx, dx
    assert "the step itself doesn't fit" not in dx, dx
    # plan too big: one execution, never launched.
    b = os.path.join(d, "b.xplane.pb")
    open(b, "wb").write(_synthetic(steps=3, keep_per_step=MIB, temp=80 * MIB, params=16 * MIB, capacity=64 * MIB))
    r = analyze(b)
    dx = " | ".join(r["diagnosis"])
    assert "the step itself doesn't fit" in dx and "grow by" not in dx, dx
    assert r["runtime"]["GPU_0_bfc"]["capacity"] == 64 * MIB, r["runtime"]
    # no donation: outputs as big as the parameters.
    n = os.path.join(d, "n.xplane.pb")
    open(n, "wb").write(_synthetic(steps=2, keep_per_step=16 * MIB, temp=8 * MIB, params=16 * MIB, capacity=256 * MIB))
    r = analyze(n)
    dx = " | ".join(r["diagnosis"])
    assert "no parameter is donated" in dx and "save up to 16 MiB" in dx, dx
    assert {x["source"] for x in r["plans"]["jit_step(1)"]["live"]} == {"model.py:12", "model.py:13", "model.py:16", "model.py:20"}
    # role rules.
    assert role("jit(f)/transpose(jvp(attn))/dot", "fusion", "f32[2]").startswith("backward")
    assert role("jit(f)/adam/mul", "fusion", "f32[2]") == "other: adam"
    assert role("", "custom-call", "s8[64]").startswith("workspace")
    print("self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", nargs="?", help="a .xplane.pb, or a directory holding one")
    ap.add_argument("--module", help="plan to break down, e.g. 'jit_train_step(25673)' (default: the module of "
                    "the last execution, else the biggest)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", help="also write the results as JSON")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.capture:
        ap.error("capture required")
    path = a.capture
    if os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, "**", "*.xplane.pb"), recursive=True))
        if len(hits) != 1:
            sys.exit(f"{path}: expected one .xplane.pb, found {len(hits)}: {hits[:5]}")
        path = hits[0]
    r = analyze(path, a.module, a.top)
    report(r, a.top)
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
