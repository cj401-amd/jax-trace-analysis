#!/usr/bin/env python3
"""SKIP-style kernel launch stats (arXiv 2504.11750) for a jax.profiler trace.json.gz.

Joins every GPU event to its host launch call by correlation_id and reports the
paper's metrics: launch lag t_l = start(kernel) - start(launch call), TKLQT = sum
of t_l, AKD, IL, GPU idle = IL - sum of t_k, top-k kernels, and kernel chains with
proximity score PS = f(chain) / f(first kernel). Two additions for graph-launched,
multi-GPU XLA, where t_l alone misleads:
  - each t_l is split into queue (the stream is still busy with earlier work),
    launch (up to a floor: the smallest t_l seen on an idle stream for that API,
    first kernel of a launch call only) and wait (stream idle, kernel not started:
    a cross-stream or peer dependency, or a graph's internal dispatch);
  - each stream's time is split into busy, host-idle (the launch call hasn't been
    made yet), launch and wait, which gives the CPU- or GPU-bound verdict.
TKLQT and the splits cover memcpys too; AKD and chains cover kernels only.
Kernels inside command buffers carry hlo_op=command_buffer_N on ROCm; --dump names
them through an XLA dump of the same program (--xla_dump_to), joining thunk ids
in thunk_sequence.txt (kernel name) and thunk_metadata.txt (HLO instruction).
How to read the output, and its limits: ../SKILL.md.

usage: skip_stats.py TRACE.json[.gz] [--dump DIR/module_N.jit_NAME] [--out DIR]
       skip_stats.py --self-test
"""
import argparse
import bisect
import collections
import csv
import glob
import gzip
import itertools
import json
import os
import re
import statistics


def load(path):
    with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
        return json.load(f)


def category(e):
    n = e["name"]
    if "memcpy_details" in e["args"] or n.startswith(("Memcpy", "__amd_rocclr_copy")):
        return "memcpy"
    return "collective" if "nccl" in n.lower() else "compute"


def kernel_map(seq, meta, hlo):
    """Kernel name -> sorted [(HLO instruction, op_name)] from an XLA dump's thunk_sequence.txt,
    thunk_metadata.txt and *_after_optimizations.txt. XLA reuses one kernel for identical
    fusions, so a name can stand for several instructions."""
    ann = {int(i): a for a, i in re.findall(r'profile_annotation: "([^"]*)"\s+thunk_id: (\d+)', meta)}
    ops = dict(re.findall(r'^\s*(?:ROOT )?%([\w.\-]+) = .*?metadata=\{[^}]*?op_name="([^"]*)"', hlo, re.M))
    m = collections.defaultdict(set)
    for i, k in re.findall(r"^\s*(\d+): kCustomKernel\s+\[[^\]]*\]\s+(\S+)", seq, re.M):
        m[k].add((ann[int(i)], ops.get(ann[int(i)], "")))
    return {k: sorted(v) for k, v in m.items()}


NO_EVENT = "(no event on the launch thread)"


def self_time(evs):
    """[(start, end, name)] of nested events on one thread -> disjoint [(start, end, innermost name)]."""
    # shortcut: assumes proper nesting (true for TraceMe); a child outliving its parent gets mislabelled.
    out, stack, t = [], [], None
    for s, e, n in sorted(evs, key=lambda x: (x[0], -x[1])):
        while stack and stack[-1][1] <= s:
            _, pe, pn = stack.pop()
            if pe > t:
                out.append((t, pe, pn))
                t = pe
        if stack and s > t:
            out.append((t, s, stack[-1][2]))
        t = s
        stack.append((s, e, n))
    while stack:
        _, pe, pn = stack.pop()
        if pe > t:
            out.append((t, pe, pn))
            t = pe
    return out


def innermost(starts, evs, a, b, k=64):
    """Duration of the tightest event enclosing [a, b] among the k events in evs (sorted by start)
    that start last before a; None if none does."""
    i = bisect.bisect_right(starts, a) - 1
    # shortcut: looks back only k events; an encloser that started before k finished siblings is missed
    return min((evs[j][1] - evs[j][0] for j in range(i, max(-1, i - k), -1) if evs[j][1] >= b), default=None)


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap_by_name(iv, segs, ends):
    """Overlap of intervals iv with sorted disjoint (start, end, name) segs (ends = their ends), per name."""
    out = collections.Counter()
    for a, b in iv:
        j = bisect.bisect_right(ends, a)
        while j < len(segs) and segs[j][0] < b:
            out[segs[j][2]] += min(b, segs[j][1]) - max(a, segs[j][0])
            j += 1
    return out


def analyze(trace, kmap=None, max_chain=8, ps_min=1.0):
    ev = trace["traceEvents"]
    pname = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "process_name"}
    tname = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "thread_name"}
    launch, steps, streams, host = {}, [], collections.defaultdict(list), collections.defaultdict(list)
    for e in ev:
        if e.get("ph") != "X":
            continue
        args = e.get("args", {})
        if pname.get(e["pid"], "").startswith("/device:GPU:"):
            streams[(pname[e["pid"]], tname.get((e["pid"], e["tid"]), str(e["tid"])))].append(e)
            continue
        host[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + e.get("dur", 0), e["name"]))
        if "correlation_id" in args:
            launch[args["correlation_id"]] = e
        elif "step_num" in args:
            steps.append(e)

    steps.sort(key=lambda s: s["ts"])
    starts = [s["ts"] for s in steps]
    step_start = {int(s["args"]["step_num"]): s["ts"] for s in steps}

    def step_of(t):  # steps tile the trace: a step runs until the next one starts
        i = bisect.bisect_right(starts, t) - 1
        if i < 0 or (i == len(steps) - 1 and t >= steps[i]["ts"] + steps[i]["dur"]):
            return None
        return int(steps[i]["args"]["step_num"])

    first = {}  # (gpu, correlation_id) -> start of the launch call's first GPU event
    for (gpu, _), es in streams.items():
        for e in es:
            k = (gpu, e["args"].get("correlation_id"))
            first[k] = min(first.get(k, e["ts"]), e["ts"])

    rows, negative, overlapped = [], 0, 0
    idle = collections.defaultdict(list)  # (gpu, stream) -> [(start, end, launch thread)] of host-idle time
    for (gpu, stream), es in sorted(streams.items()):
        es.sort(key=lambda e: e["ts"])
        prev_end = None  # running max end of earlier events on this stream
        for e in es:
            args, k0 = e["args"], e["ts"]
            r = dict(gpu=gpu, stream=stream, step=None, name=e["name"], cat=category(e),
                     corr=args.get("correlation_id", ""), api="", start=k0, dur=e["dur"],
                     lag=0.0, queue=0.0, launch=0.0, wait=0.0, idle_after=0.0, host_idle=0.0,
                     gap=0.0 if prev_end is None else max(0.0, k0 - prev_end),
                     first_of_launch=False, negative=False, cls="none",
                     hlo_op=args.get("hlo_op", ""), op_name=args.get("name", ""))
            overlapped += prev_end is not None and k0 < prev_end  # graph nodes run concurrently
            l = launch.get(args.get("correlation_id"))
            if l is not None:
                l0 = l["ts"]
                r.update(api=l["name"], step=step_of(l0), negative=k0 < l0)
                negative += k0 < l0
                if k0 >= l0:
                    r.update(lag=k0 - l0, first_of_launch=k0 == first[(gpu, r["corr"])])
                    if prev_end is not None:
                        r["queue"] = min(max(prev_end - l0, 0.0), k0 - l0)
                        r["host_idle"] = max(0.0, l0 - prev_end)
                        if r["host_idle"] > 0:
                            idle[(gpu, stream)].append((prev_end, l0, (l["pid"], l["tid"])))
                    r["idle_after"] = r["lag"] - r["queue"]
            rows.append(r)
            prev_end = k0 + e["dur"] if prev_end is None else max(prev_end, k0 + e["dur"])

    # shortcut: the floor is the smallest idle-stream lag in this trace; replace it with
    # the null-kernel benchmark (prompt step 3) once that exists.
    floor = {}
    for r in rows:
        if r["first_of_launch"] and r["queue"] == 0:
            floor[r["api"]] = min(floor.get(r["api"], r["idle_after"]), r["idle_after"])
    for r in rows:
        r["launch"] = min(r["idle_after"], floor.get(r["api"], 0.0)) if r["first_of_launch"] else 0.0
        r["wait"] = r["idle_after"] - r["launch"]
        h = r["hlo_op"]
        hit = kmap.get(r["name"].removesuffix(".kd")) if kmap else None
        if h and not h.startswith("command_buffer"):
            r["cls"] = "exact"
        elif hit:  # kernel inside a command buffer: name it through the XLA dump
            ops = sorted({o for _, o in hit})
            r.update(cls="name" if len(ops) == 1 else "ambiguous",
                     hlo_op="|".join(i for i, _ in hit), op_name="|".join(ops))

    by_step = collections.defaultdict(list)
    for r in rows:
        if r["step"] is not None:
            by_step[(r["gpu"], r["step"])].append(r)
    # busy = time any kernel runs on the GPU, any stream, any step: unlike sum t_k it neither double-counts
    # overlapping streams nor calls another step's kernels idle time when the host runs ahead
    kern = collections.defaultdict(list)
    for r in rows:
        if r["cat"] != "memcpy":
            kern[r["gpu"]].append((r["start"], r["start"] + r["dur"]))
    busy_iv = {}
    for gpu, iv in kern.items():
        m = [(a, b, "busy") for a, b in merge(iv)]
        busy_iv[gpu] = (m, [x[1] for x in m])
    step_rows = []
    for (gpu, s), rs in sorted(by_step.items()):
        ks = [r for r in rs if r["cat"] != "memcpy"]
        il = max(r["start"] + r["dur"] for r in rs) - step_start[s]
        kt = sum(r["dur"] for r in ks)
        busy = overlap_by_name([(step_start[s], step_start[s] + il)], *busy_iv.get(gpu, ([], [])))["busy"]
        step_rows.append(dict(gpu=gpu, step=s, kernels=len(ks), il=il, kernel_time=kt, idle=il - kt,
                              busy=busy, idle_merged=il - busy,
                              tklqt=sum(r["lag"] for r in rs), akd=kt / len(ks) if ks else 0.0))

    # API calls sit on their own lines ("Host Threads/<tid>"), apart from the TraceMe line of the same
    # thread: give each API line the line whose innermost event around its calls is tightest.
    api_lines = {(l["pid"], l["tid"]) for l in launch.values()}
    for evs in host.values():
        evs.sort()
    starts = {th: [x[0] for x in evs] for th, evs in host.items() if th not in api_lines}
    calls = collections.defaultdict(list)
    for l in launch.values():
        calls[(l["pid"], l["tid"])].append((l["ts"], l["ts"] + l.get("dur", 0)))
    owner = {}
    for th, cs in calls.items():
        votes = collections.Counter()
        for a, b in sorted(cs)[::max(1, len(cs) // 50)]:
            best = min(((d, c) for c in starts if (d := innermost(starts[c], host[c], a, b)) is not None),
                       default=None)
            if best:
                votes[best[1]] += 1
        owner[th] = votes.most_common(1)[0][0] if votes else th

    stream_rows, seqs, segs = [], [], {}  # segs: thread line -> self_time() of its events
    for (gpu, stream), rs in itertools.groupby(rows, key=lambda r: (r["gpu"], r["stream"])):
        rs = list(rs)
        span = max(r["start"] + r["dur"] for r in rs) - rs[0]["start"]
        inner = rs[1:]  # the first event's lag lies before the stream's span
        d = dict(gpu=gpu, stream=stream, events=len(rs), span=span,
                 busy=span - sum(r["gap"] for r in inner),
                 **{c: sum(r["dur"] for r in rs if r["cat"] == c) for c in ("compute", "collective", "memcpy")},
                 **{k: sum(r[k] for r in inner) for k in ("host_idle", "launch", "wait")})
        d["other"] = span - d["busy"] - d["host_idle"] - d["launch"] - d["wait"]  # unlinked or negative lag
        # what the launch thread was doing (innermost event) while the stream waited for its launch call
        causes, lines = collections.Counter(), set()
        for th in {t for _, _, t in idle[(gpu, stream)]}:
            mine = merge([(a, b) for a, b, t in idle[(gpu, stream)] if t == th])
            line = owner.get(th, th)
            lines.add(tname.get(line, str(line[1])))
            if line not in segs:
                sl = self_time(host[line])
                segs[line] = (sl, [x[1] for x in sl])
            causes += overlap_by_name(mine, *segs[line])
            causes[NO_EVENT] += sum(b - a for a, b in mine)
        causes[NO_EVENT] -= sum(v for n, v in causes.items() if n != NO_EVENT)
        d["causes"] = {n: v for n, v in causes.items() if v > 0}
        d["cause_line"] = ", ".join(sorted(lines))
        stream_rows.append(d)
        seqs.append([r["name"] for r in rs if r["cat"] != "memcpy"])

    f1 = collections.Counter(n for q in seqs for n in q)
    k_eager = sum(f1.values())
    chain_rows, chain_list = [], []
    for L in range(2, max_chain + 1):
        fc = collections.Counter(tuple(q[i:i + L]) for q in seqs for i in range(len(q) - L + 1))
        if not fc:
            break
        det = {c for c, n in fc.items() if n / f1[c[0]] >= ps_min}
        # shortcut: the paper doesn't say how overlapping chains count toward C_fused; this
        # fuses greedily, left to right, without overlap.
        fused = 0
        for q in seqs:
            i = 0
            while i + L <= len(q):
                hit = tuple(q[i:i + L]) in det
                fused += hit
                i += L if hit else 1
        chain_rows.append(dict(L=L, distinct=len(fc), instances=sum(fc.values()),
                               det_instances=sum(fc[c] for c in det), fused=fused,
                               k_eager=k_eager, k_fused=k_eager - fused * (L - 1)))
        chain_list += [dict(L=L, count=n, ps=n / f1[c[0]], chain=" > ".join(c)) for c, n in fc.most_common()]

    ops = collections.defaultdict(lambda: dict(kernels=0, dur=0.0, lag=0.0))
    for r in rows:
        o = ops[(r["cls"], r["op_name"], r["hlo_op"])]
        o["kernels"] += 1
        o["dur"] += r["dur"]
        o["lag"] += r["lag"]
    op_rows = sorted((dict(cls=c, op_name=n, hlo_op=h, **v) for (c, n, h), v in ops.items()),
                     key=lambda o: -o["dur"])
    return dict(rows=rows, negative=negative, overlap=overlapped, floor=floor, n_steps=len(steps), steps=step_rows,
                streams=stream_rows, chains=chain_rows, chain_list=chain_list, ops=op_rows)


def table(header, lines):
    cols = [header] + lines
    w = [max(len(str(c[i])) for c in cols) for i in range(len(header))]
    return "\n".join("  ".join(str(x).ljust(w[i]) if i == 0 else str(x).rjust(w[i]) for i, x in enumerate(c))
                     for c in cols)


def rng(xs, scale=1.0):
    xs = [x * scale for x in xs]
    return f"{statistics.median(xs):.1f} [{min(xs):.1f}-{max(xs):.1f}]"


def pct(x, total):
    return f"{100 * x / total:.1f}" if total else "-"


VERDICT = {"compute": "GPU-bound (compute kernels)",
           "collective": "collective-bound (RCCL kernels running or spinning on peers)",
           "host_idle": "CPU-bound (stream idle until the host calls launch)",
           "launch": "launch-bound",
           "wait": "dependency-bound (launched, stream idle, kernel not started)"}


def verdict(shares):
    (k1, v1), (k2, v2) = sorted(shares.items(), key=lambda x: -x[1])[:2]
    return VERDICT[k1] if v1 - v2 >= 0.05 else f"mixed, no single bottleneck ({k1} ~ {k2})"


def summary(res, top=10):
    rows, out = res["rows"], []
    p = out.append
    linked = sum(1 for r in rows if r["api"])
    p(f"GPU events {len(rows)}: linked to a host launch call {linked}, unlinked {len(rows) - linked}, "
      f"negative lag {res['negative']}; starting before the previous event on the stream ends "
      f"{res['overlap']} (their lag counts as queue)")
    p("launch floor per API (smallest lag on an idle stream, us): "
      + ", ".join(f"{a} {v:.1f}" for a, v in sorted(res["floor"].items())))

    per_gpu = collections.defaultdict(list)
    for s in res["steps"]:
        per_gpu[s["gpu"]].append(s)
    if per_gpu:
        p(f"\n== Paper metrics per GPU, median [min-max] over {res['n_steps']} steps (a GPU event belongs to "
          "the step its launch call falls in). The paper's idle, IL - sum t_k, double-counts overlapping "
          "streams (it can go negative) and counts other steps' kernels as idle when the host runs ahead. "
          "Use idle = IL - busy, where busy is the time in the step's window with any kernel running ==")
        p(table(["GPU", "kernels", "IL ms", "sum t_k ms", "paper idle ms", "busy ms", "idle ms (IL - busy)",
                 "TKLQT ms", "AKD us"],
                [[g, rng([s["kernels"] for s in ss]), rng([s["il"] for s in ss], 1e-3),
                  rng([s["kernel_time"] for s in ss], 1e-3), rng([s["idle"] for s in ss], 1e-3),
                  rng([s["busy"] for s in ss], 1e-3), rng([s["idle_merged"] for s in ss], 1e-3),
                  rng([s["tklqt"] for s in ss], 1e-3), rng([s["akd"] for s in ss])]
                 for g, ss in sorted(per_gpu.items())]))

    tot = collections.defaultdict(collections.Counter)
    for r in rows:
        for k in ("lag", "queue", "launch", "wait"):
            tot[r["gpu"]][k] += r[k]
    p("\n== TKLQT split, whole trace: queue = stream busy with earlier work; launch = up to the floor, "
      "first kernel of a launch call only; wait = stream idle, kernel not started ==")
    p(table(["GPU", "TKLQT s", "queue %", "launch %", "wait %"],
            [[g, f"{t['lag'] / 1e6:.2f}", pct(t["queue"], t["lag"]), pct(t["launch"], t["lag"]),
              pct(t["wait"], t["lag"])] for g, t in sorted(tot.items())]))

    p("\n== Stream time, first event start to last event end (compute/collective/memcpy are summed "
      "durations; other = gaps before unlinked or negative-lag events) ==")
    p(table(["GPU", "stream", "events", "span ms", "busy %", "compute %", "collective %", "memcpy %",
             "host-idle %", "launch %", "wait %", "other %"],
            [[s["gpu"], s["stream"], s["events"], f"{s['span'] / 1e3:.1f}"]
             + [pct(s[k], s["span"]) for k in ("busy", "compute", "collective", "memcpy", "host_idle",
                                                "launch", "wait", "other")] for s in res["streams"]]))

    p("\n== Verdict, on each GPU's main compute stream (most compute-kernel time); "
      "mixed when the top two shares are within 5 points ==")
    main = {}
    for s in res["streams"]:
        if s["gpu"] not in main or s["compute"] > main[s["gpu"]]["compute"]:
            main[s["gpu"]] = s
    for g, s in sorted(main.items()):
        sh = {k: s[k] / s["span"] if s["span"] else 0.0 for k in VERDICT}
        p(f"{g} {s['stream']}: {verdict(sh)}; {s['span'] / 1e3:.0f} ms = "
          + ", ".join(f"{k} {100 * v:.0f}%" for k, v in sh.items()))
        if s["host_idle"]:
            p(f"    host-idle time, innermost event on {s['cause_line']}: " + ", ".join(
                f"{n} {pct(v, s['host_idle'])}%" for n, v in sorted(s["causes"].items(), key=lambda x: -x[1])[:5]))

    agg = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["cat"] != "memcpy":
            a = agg[r["name"]]
            a["n"] += 1
            for k in ("dur", "lag", "queue", "launch", "wait"):
                a[k] += r[k]
    for key, title in (("n", "count"), ("dur", "total GPU time"), ("lag", "total launch lag")):
        p(f"\n== Top {top} kernels by {title} (all GPUs) ==")
        p(table(["kernel", "count", "GPU ms", "lag ms", "queue %", "launch %", "wait %"],
                [[n[:70], a["n"], f"{a['dur'] / 1e3:.1f}", f"{a['lag'] / 1e3:.1f}", pct(a["queue"], a["lag"]),
                  pct(a["launch"], a["lag"]), pct(a["wait"], a["lag"])]
                 for n, a in sorted(agg.items(), key=lambda x: -x[1][key])[:top]]))

    if res["chains"]:
        p("\n== Kernel chains in stream order: deterministic = PS >= T; fused = greedy non-overlapping "
          "instances; speedup = K_eager / K_fused (paper's idealized launch-count speedup) ==")
        p(table(["L", "distinct", "instances", "deterministic %", "fused", "K_eager", "K_fused", "speedup"],
                [[c["L"], c["distinct"], c["instances"], pct(c["det_instances"], c["instances"]), c["fused"],
                  c["k_eager"], c["k_fused"], f"{c['k_eager'] / c['k_fused']:.2f}"] for c in res["chains"]]))

    cls = collections.Counter()
    for r in rows:
        cls[r["cls"]] += r["dur"]
    total = sum(cls.values())
    p("\n== Operator attribution: exact = hlo_op from the tracer; name = command-buffer kernel named "
      "through the XLA dump; ambiguous = that kernel is shared by instructions with different op_names; "
      "none = unattributed ==")
    p("share of GPU time: " + ", ".join(f"{c} {pct(cls[c], total)}%" for c in ("exact", "name", "ambiguous", "none")))
    p(table(["op_name", "hlo_op", "how", "kernels", "GPU ms", "lag ms"],
            [[o["op_name"][:60], o["hlo_op"][:30], o["cls"], o["kernels"], f"{o['dur'] / 1e3:.1f}",
              f"{o['lag'] / 1e3:.1f}"] for o in res["ops"][:top]]))
    return "\n".join(out)


def write_csvs(res, out):
    os.makedirs(out, exist_ok=True)
    for name in ("rows", "steps", "streams", "chains", "chain_list", "ops"):
        if res[name]:
            with open(os.path.join(out, name + ".csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(res[name][0]))
                w.writeheader()
                w.writerows(res[name])


def self_test():
    """A hand-built trace; every expected value below was worked out by hand."""
    def M(pid, tid, kind, name):
        return {"ph": "M", "pid": pid, "tid": tid, "name": kind, "args": {"name": name}}

    def X(pid, tid, ev, ts, dur, **args):
        return {"ph": "X", "pid": pid, "tid": tid, "name": ev, "ts": ts, "dur": dur, "args": args}

    k1, k2 = "loop_add_fusion_1.kd", "input_reduce_fusion_2.kd"
    trace = {"traceEvents": [
        M(1, 0, "process_name", "/device:GPU:0"), M(100, 0, "process_name", "/host:CPU"),
        M(1, 0, "thread_name", "Stream #0"), M(1, 1, "thread_name", "Stream #1"),
        M(1, 2, "thread_name", "Stream #2"), M(1, 3, "thread_name", "Stream #3"),
        M(100, 1, "thread_name", "python3"), M(100, 2, "thread_name", "Host Threads/2"),
        M(100, 3, "thread_name", "py_xla_execute/1"),
        X(100, 1, "train", 0, 100, step_num="0"), X(100, 1, "train", 100, 100, step_num="1"),
        X(100, 2, "hipModuleLaunchKernel", 10, 5, correlation_id="1"),
        X(100, 2, "hipGraphLaunch", 35, 5, correlation_id="2"),
        X(100, 2, "hipModuleLaunchKernel", 150, 5, correlation_id="3"),
        X(100, 2, "hipModuleLaunchKernel", 170, 5, correlation_id="4"),
        X(100, 2, "hipModuleLaunchKernel", 180, 5, correlation_id="5"),
        X(100, 2, "hipModuleLaunchKernel", 60, 5, correlation_id="7"),
        # XLA's events for the same OS thread sit on another line; each launch call sits in a thunk
        X(100, 3, "ExecuteHelper", 66, 114), X(100, 3, "prepare", 66, 2),  # same start: prepare is the child
        X(100, 3, "command_buffer::update", 70, 90),
        *(X(100, 3, "thunk", a, b - a) for a, b in ((9, 16), (34, 41), (149, 156), (169, 176), (180, 186),
                                                   (399, 406))),
        # decoy like a Python-tracer line: tiny events that end just before each launch call
        *(X(100, 4, "py_frame", a - 2, 1) for a in (10, 35, 150, 170, 180, 400)),
        X(1, 0, k1, 20, 10, correlation_id="1", hlo_op="loop_add_fusion.1", name="jit(f)/add"),
        X(1, 0, k2, 45, 10, correlation_id="2", hlo_op="command_buffer_3"),  # graph: stream idle at launch
        X(1, 0, k1, 60, 5, correlation_id="2", hlo_op="command_buffer_3"),  # graph: queued behind k2
        X(1, 1, "ncclDevKernel_Generic_1", 160, 30, correlation_id="3", hlo_op="all-gather.1",
          name="jit(f)/all_gather"),
        X(1, 1, "fusion_9.kd", 192, 3, correlation_id="5", hlo_op="fusion.9",
          name="jit(f)/mul"),  # launched while the stream is busy: must not lower the floor
        X(1, 0, "Cijk_gemm.kd", 200, 10, correlation_id="4", hlo_op="custom-call.5",
          name="jit(f)/dot_general"),  # host launched late: 105 us host-idle before it
        X(1, 2, "Cijk_x.kd", 300, 5, correlation_id="99", hlo_op="command_buffer_4"),  # no launch call
        # launched in step 0, runs in step 1's window on another stream, overlapping Cijk_gemm
        X(1, 2, "fusion_7.kd", 200, 4, correlation_id="7", hlo_op="fusion.7", name="jit(f)/sub"),
        # after the steps: one graph whose first two nodes run concurrently, then a third
        X(100, 2, "hipGraphLaunch", 400, 5, correlation_id="6"),
        X(1, 3, "p.kd", 410, 20, correlation_id="6", hlo_op="command_buffer_9"),
        X(1, 3, "q.kd", 415, 5, correlation_id="6", hlo_op="command_buffer_9"),  # starts inside p
        X(1, 3, "r.kd", 432, 2, correlation_id="6", hlo_op="command_buffer_9"),
    ]}
    # XLA dump: kernel loop_add_fusion_1 serves two instructions with one op_name,
    # input_reduce_fusion_2 serves two with different op_names
    seq = ("001: kCustomKernel   [source | next=002 (1)] loop_add_fusion_1 grid: [1, 1, 1]\n"
           "002: kCustomKernel   [prev=001 (1) | sink] input_reduce_fusion_2 grid: [1, 1, 1]\n"
           "003: kCopy           [source | sink] kCopy\n"
           "    004: kCustomKernel   [source | sink] input_reduce_fusion_2 grid: [1, 1, 1]\n"
           "005: kCustomKernel   [source | sink] loop_add_fusion_1 grid: [1, 1, 1]\n")
    meta = "".join(f'thunk_metadata {{\n  thunk_info {{\n    profile_annotation: "{a}"\n    thunk_id: {i}\n  }}\n}}\n'
                   for i, a in enumerate(["loop_add_fusion.1", "input_reduce_fusion.2", "copy.1",
                                          "input_reduce_fusion.3", "loop_add_fusion.4"], 1))
    hlo = ('  %loop_add_fusion.1 = f32[8]{0} fusion(%p), metadata={op_name="jit(f)/add" scheduling_name="x"}\n'
           '  %input_reduce_fusion.2 = f32[] fusion(%p), metadata={op_name="jit(f)/reduce_sum"}\n'
           '  %input_reduce_fusion.3 = f32[] fusion(%p), metadata={scheduling_name="y" op_name="jit(f)/reduce_max"}\n'
           '  ROOT %loop_add_fusion.4 = f32[8]{0} fusion(%p), metadata={op_name="jit(f)/add"}\n')
    res = analyze(trace, kernel_map(seq, meta, hlo))
    rows = {(r["stream"], r["start"]): r for r in res["rows"]}

    def split(stream, ts):
        return tuple(rows[(stream, ts)][k] for k in ("lag", "queue", "launch", "wait", "host_idle"))

    assert split("Stream #0", 20) == (10, 0, 10, 0, 0)
    assert split("Stream #0", 45) == (10, 0, 10, 0, 5)
    assert split("Stream #0", 60) == (25, 20, 0, 5, 0)
    assert split("Stream #0", 200) == (30, 0, 10, 20, 105)
    assert split("Stream #1", 160) == (10, 0, 10, 0, 0)
    assert split("Stream #1", 192) == (12, 10, 2, 0, 0)
    assert split("Stream #3", 410) == (10, 0, 10, 0, 0)
    assert split("Stream #3", 415) == (15, 15, 0, 0, 0)  # overlapping node: all queue
    assert split("Stream #3", 432) == (32, 30, 0, 2, 0)
    assert res["overlap"] == 1
    assert res["floor"] == {"hipModuleLaunchKernel": 10, "hipGraphLaunch": 10}, res["floor"]
    assert sum(1 for r in res["rows"] if not r["api"]) == 1 and res["negative"] == 0
    steps = {s["step"]: (s["kernels"], s["il"], s["kernel_time"], s["idle"], s["busy"], s["idle_merged"], s["tklqt"])
             for s in res["steps"]}
    # step 0's window [0, 204] holds step 1's kernels too: paper idle 175 counts them as idle, merged idle 142 doesn't
    assert steps == {0: (4, 204, 29, 175, 62, 142, 185), 1: (3, 110, 43, 67, 43, 67, 52)}, steps
    s0 = next(s for s in res["streams"] if s["stream"] == "Stream #0")
    got = tuple(s0[k] for k in ("span", "busy", "compute", "host_idle", "launch", "wait", "other"))
    assert got == (190, 35, 35, 110, 20, 25, 0), got
    # host-idle [30,35] + [65,170]: innermost event on the launch thread during it
    assert s0["causes"] == {"command_buffer::update": 83, "ExecuteHelper": 11, "thunk": 9, "prepare": 2,
                            NO_EVENT: 5}, s0["causes"]
    assert s0["cause_line"] == "py_xla_execute/1", s0["cause_line"]
    s3 = next(s for s in res["streams"] if s["stream"] == "Stream #3")
    got = tuple(s3[k] for k in ("span", "busy", "host_idle", "launch", "wait", "other"))
    assert got == (24, 22, 0, 0, 2, 0), got
    got = [(c["L"], c["det_instances"], c["fused"], c["k_fused"]) for c in res["chains"]]
    assert got == [(2, 5, 4, 7), (3, 2, 2, 7), (4, 0, 0, 11)], got
    cls = collections.Counter()
    for r in res["rows"]:
        cls[r["cls"]] += r["dur"]
    assert cls == {"exact": 57, "name": 5, "ambiguous": 10, "none": 32}, cls
    assert rows[("Stream #0", 60)]["hlo_op"] == "loop_add_fusion.1|loop_add_fusion.4"
    assert rows[("Stream #0", 45)]["op_name"] == "jit(f)/reduce_max|jit(f)/reduce_sum"
    assert "CPU-bound" in summary(res).split("Verdict")[1]
    assert verdict({"compute": .31, "host_idle": .27, "wait": .22}).startswith("mixed")
    assert verdict({"compute": .31, "host_idle": .25, "wait": .22}).startswith("GPU-bound")
    print("self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", help="trace.json or trace.json.gz written by jax.profiler / XProf")
    ap.add_argument("--dump", help="XLA dump prefix of the traced program, e.g. dump/module_2701230.jit_train_step, "
                                   "to name kernels inside command buffers")
    ap.add_argument("--out", help="directory for the CSV tables and summary.txt "
                                  "(default: skip_out/<trace dir name> in the current directory)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--max-chain", type=int, default=8)
    ap.add_argument("--ps", type=float, default=1.0, help="proximity-score threshold T for fusion candidates")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.trace:
        ap.error("trace is required")
    kmap = None
    if a.dump:
        (hlo,) = glob.glob(a.dump + ".*_after_optimizations.txt")
        kmap = kernel_map(*(open(f).read() for f in (a.dump + ".thunk_sequence.txt",
                                                      a.dump + ".thunk_metadata.txt", hlo)))
    res = analyze(load(a.trace), kmap, a.max_chain, a.ps)
    text = f"trace: {a.trace}\n" + summary(res, a.top)
    out = a.out or os.path.join("skip_out", os.path.basename(os.path.dirname(os.path.abspath(a.trace))))
    write_csvs(res, out)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\ntables: {out}")


if __name__ == "__main__":
    main()
