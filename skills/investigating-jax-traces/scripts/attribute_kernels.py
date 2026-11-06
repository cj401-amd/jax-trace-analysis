#!/usr/bin/env python3
"""Attribute every GPU kernel in a jax.profiler capture to its HLO instruction and Python line.

usage: attribute_kernels.py CAPTURE_DIR_OR_XPLANE [--program ID] [--out DIR] [--user-re REGEX]
       attribute_kernels.py --self-test

Uses only the capture: the .xplane.pb holds the events and the HLO of every program. The xplane is
copied to OUT first, because XProf's converter writes the HLO protos next to it.

Kernels launched one per call carry the tracer's hlo_op ("exact") and serve as anchors. Kernels in
a HIP/CUDA graph (command buffer) carry hlo_op=command_buffer_N on ROCm. Each is matched to an
instruction executed between its graph launch's neighbouring anchors, in the program's schedule
with while loops expanded by their known trip count:
  name   the only instruction in that window with this kernel name
  order  several candidates (a shared kernel, a library GEMM, a copy, a collective): the n-th such
         kernel in the window takes the n-th candidate
  none   no candidate left (counted per window as an overflow)
"""
import argparse
import bisect
import collections
import csv
import glob
import os
import re
import shutil

COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute")
LIB_FRAME = r"/(site-packages|dist-packages)/|/jax/_src/|/lib/python3\.\d+/"


def kernel_kind(name, groups):
    if name.startswith(("Memcpy", "__amd_rocclr_copy")):
        return "copy"
    if "nccl" in name.lower():
        return "collective"
    return "kernel" if name in groups else "library"


def inst_kind(name, opcode):
    if opcode == "custom-call":
        return "library"
    if opcode == "copy" or name.startswith("wrapped_slice"):  # contiguous slices run as a D2D copy
        return "copy"
    if opcode.startswith(COLLECTIVES) and not opcode.endswith("-done"):  # -done launches nothing
        return "collective"
    return "kernel" if opcode == "fusion" else None


def assign(seq, info, launches):
    """Match kernels to instructions.

    seq:      instruction names in execution order for one run of the program
    info:     name -> (kind, kernel group) for instructions that launch work
    launches: [(the tracer's hlo_op, or None for a graph launch, [(kernel kind, kernel group), ...])]
              in host launch order; an hlo_op found in seq also anchors the windows
    Returns ([[(instruction or None, how)] per kernel] per launch, overflowing windows). For an hlo_op not
    in seq, the instruction is the candidate matched by order: it gives the line, the hlo_op the label.
    """
    occ = collections.defaultdict(list)
    for i, n in enumerate(seq):
        occ[n].append(i)
    pos, cur = [], -1
    for anchor, _ in launches:
        p = None
        if anchor in occ:
            o = occ[anchor]
            j = bisect.bisect_right(o, cur)
            p = o[j] if j < len(o) else o[0]  # no later occurrence: a new run of the program
            cur = p
        pos.append(p)
    nxt, after = [None] * len(launches), None
    for i in range(len(launches) - 1, -1, -1):
        nxt[i] = after
        if pos[i] is not None:
            after = i
    out, used, buckets, overflow, prev = [], collections.Counter(), {}, set(), None
    for i, (anchor, kernels) in enumerate(launches):
        if pos[i] is not None:
            prev = i
            out.append([(anchor, "exact")] * len(kernels))
            continue
        lo = pos[prev] + 1 if prev is not None else 0
        hi = pos[nxt[i]] if nxt[i] is not None else len(seq)
        key = (prev, nxt[i])
        if key not in buckets:
            window = seq[lo:hi] if hi >= lo else seq[lo:] + seq[:hi]  # hi < lo: the window wraps a run
            b = collections.defaultdict(list)
            for n in window:
                if n in info:
                    k, g = info[n]
                    b[(k, g if k == "kernel" else "")].append(n)
            buckets[key] = b
        row, launched = [], {g for k, g in kernels if k == "kernel"}
        for kind, grp in kernels:
            if anchor is not None and kind != "collective":  # only collectives are labelled with unscheduled ops
                row.append((None, "exact"))
                continue
            c = buckets[key][(kind, grp if kind == "kernel" else "")]
            if kind == "copy":  # a wrapped_slice launched as a kernel in this graph is not a copy
                c = [n for n in c if info[n][1] not in launched]
            j = used[(key, kind, grp)]
            used[(key, kind, grp)] += 1
            how = "exact" if anchor is not None else "name" if kind == "kernel" and len(c) == 1 else "order"
            if j < len(c):
                row.append((c[j], how))
            else:
                row.append((None, "exact" if anchor is not None else "none"))
                overflow.add(key)
        out.append(row)
    return out, overflow


def hlo_text(xplane, program, out):
    """Copy the xplane to out, let XProf write the HLO protos next to it, and print the program's HLO."""
    from google.protobuf.internal import decoder
    from jax._src.lib import hlo as H
    from xprof.convert import raw_to_tool_data
    dst = os.path.join(out, "xplane", os.path.basename(xplane))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy(xplane, dst)
        raw_to_tool_data.xspace_to_tool_names([os.path.abspath(dst)])
    (pb,) = glob.glob(os.path.join(os.path.dirname(dst), f"*({program}).hlo_proto.pb"))
    b, p = open(pb, "rb").read(), 0
    while p < len(b):  # HloProto: field 1 is the HloModuleProto
        key, p = decoder._DecodeVarint(b, p)
        if key & 7 == 2:
            n, p = decoder._DecodeVarint(b, p)
            if key >> 3 == 1:
                # Loading renumbers stack_frame_ids, so decode them against the printed table, not the proto's.
                return os.path.basename(pb), H.HloModule.from_serialized_hlo_module_proto(b[p:p + n]).to_string()
            p += n
        elif key & 7 == 0:
            _, p = decoder._DecodeVarint(b, p)
        else:
            p += 4 if key & 7 == 5 else 8
    raise ValueError(f"no HloModuleProto in {pb}")


def parse_hlo(text):
    """-> (inst {name: (opcode, op_name, kernel group, stack)}, computations {name: [inst]}, entry, whiles)."""
    files, funcs, locs, frames, section = [], [], [], [], None
    inst, comps, whiles, comp, entry = {}, {}, {}, None, None
    for line in text.splitlines():
        if line in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
            section = line
            continue
        if section and (line[:1].isdigit() or not line.strip()):
            if line[:1].isdigit():
                if section == "FileNames":
                    files.append(line.split(" ", 1)[1].strip('"'))
                elif section == "FunctionNames":
                    funcs.append(line.split(" ", 1)[1].strip('"'))
                else:
                    d = dict(re.findall(r"(\w+)=(-?\d+)", line))
                    if section == "FileLocations":
                        locs.append((int(d["file_name_id"]), int(d["function_name_id"]), int(d["line"])))
                    else:
                        frames.append((int(d["file_location_id"]), int(d["parent_frame_id"]) - 1))  # printed +1
            continue
        section = None
        m = re.match(r"^(ENTRY )?%?([\w.\-]+) .*\{\s*$", line)
        if m and not line.startswith(" "):
            comp = m.group(2)
            comps[comp] = []
            if m.group(1):
                entry = comp
            continue
        m = re.match(r"^\s*(?:ROOT )?%([\w.\-]+) = .*? ([a-z][\w\-]*)\(", line)
        if not m or comp is None:
            continue
        name, opcode = m.groups()
        comps[comp].append(name)
        md = re.search(r"metadata=\{([^}]*)\}", line)
        md = md.group(1) if md else ""
        op = re.search(r'op_name="([^"]*)"', md)
        dd = re.search(r'deduplicated_name="([^"]*)"', md)
        fid = re.search(r"stack_frame_id=(\d+)", md)
        inst[name] = [opcode, op.group(1) if op else "", (dd.group(1) if dd else name).replace(".", "_").replace("-", "_"),
                      int(fid.group(1)) if fid else 0]
        if opcode == "while":
            body = re.search(r"body=%?([\w.\-]+)", line)
            trip = re.search(r'known_trip_count[^0-9]*(\d+)', line)
            whiles[name] = (body.group(1), int(trip.group(1)) if trip else 1)

    def stack(fid):
        out = []
        while fid:
            loc, fid = frames[fid - 1]
            f, fn, ln = locs[loc - 1]
            out.append(f"{files[f - 1]}:{ln} {funcs[fn - 1]}")
        return out

    for v in inst.values():
        v[3] = stack(v[3])
    return inst, comps, entry, whiles


def flatten(comp, comps, whiles):
    out = []
    for n in comps[comp]:
        if n in whiles:
            body, trip = whiles[n]
            out += flatten(body, comps, whiles) * trip
        else:
            out.append(n)
    return out


def load_events(xplane):
    from jax._src.lib import _profile_data
    gpu, host = [], {}
    for pl in _profile_data.ProfileData.from_file(xplane).planes:
        is_gpu = pl.name.startswith("/device:GPU:")
        for line in pl.lines:
            for e in line.events:
                st = dict(e.stats)
                if is_gpu:
                    gpu.append(dict(gpu=int(pl.name.rsplit(":", 1)[1]), stream=line.name, kernel=e.name,
                                    start_ns=e.start_ns, dur_ns=e.duration_ns, corr=st.get("correlation_id"),
                                    hlo_op=st.get("hlo_op") or "", op=st.get("name") or "", program=st.get("program_id")))
                elif pl.name == "/host:CPU" and "correlation_id" in st:
                    host[st["correlation_id"]] = (e.start_ns, e.name)
    return gpu, host


def analyze(xplane, program=None, out="attribution", user_re=None):
    gpu, host = load_events(xplane)
    t = collections.Counter()
    for e in gpu:
        t[e["program"]] += e["dur_ns"]
    program = program or max((p for p in t if p is not None), key=t.get)
    pbname, text = hlo_text(xplane, program, out)
    open(os.path.join(out, pbname.replace(".hlo_proto.pb", ".hlo.txt")), "w").write(text)  # shapes, for a roofline
    inst, comps, entry, whiles = parse_hlo(text)
    groups = {n.replace(".", "_").replace("-", "_"): v[2] for n, v in inst.items()}
    info = {n: (k, v[2]) for n, v in inst.items() if (k := inst_kind(n, v[0]))}
    seq = flatten(entry, comps, whiles)

    mine = [e for e in gpu if e["program"] == program]
    by_corr = collections.defaultdict(list)
    for e in mine:
        e["name"] = e["kernel"].removesuffix(".kd")
        by_corr[(e["gpu"], e["corr"])].append(e)
    rows, overflow = [], 0
    for g in sorted({e["gpu"] for e in mine}):
        calls = sorted(((host.get(c, (s[0]["start_ns"], ""))[0], c) for (gg, c), s in by_corr.items() if gg == g))
        launches, evs = [], []
        for _, c in calls:
            es = sorted(by_corr[(g, c)], key=lambda e: e["start_ns"])
            anchor = next((e["hlo_op"] for e in es if e["hlo_op"] and not e["hlo_op"].startswith("command_buffer")), None)
            launches.append((anchor,
                             [(kind_of := kernel_kind(e["name"], groups), groups.get(e["name"], e["name"]) if kind_of == "kernel" else "")
                              for e in es]))
            evs.append(es)
        res, ov = assign(seq, info, launches)
        overflow += len(ov)
        for (anchor, _), es, r in zip(launches, evs, res):
            for e, (src, how) in zip(es, r):
                own = e["hlo_op"] if e["hlo_op"] and not e["hlo_op"].startswith("command_buffer") else ""
                src = own if own in inst else src  # the instruction whose metadata gives op_name and line
                _, op, _, st = inst.get(src, [None, e["op"], "", []])
                user = next((s for s in st if (re.search(user_re, s) if user_re else not re.search(LIB_FRAME, s))), "")
                rows.append(dict(gpu=g, stream=e["stream"], start_ns=e["start_ns"], dur_ns=e["dur_ns"], kernel=e["kernel"],
                                 corr=e["corr"], api=host.get(e["corr"], (0, ""))[1], hlo=own or src or "", how=how,
                                 inst=src or "", op_name=op, frame=st[0] if st else "", user_frame=user))
    return dict(program=program, hlo=pbname, rows=rows, overflow=overflow, n_seq=len(seq), whiles=whiles)


def rollup(rows, key):
    t, n = collections.Counter(), collections.Counter()
    for r in rows:
        t[key(r)] += r["dur_ns"]
        n[key(r)] += 1
    total = sum(t.values()) or 1
    return [dict(key=k, kernels=n[k], gpu_ms=round(v / 1e6, 3), share=round(100 * v / total, 2)) for k, v in t.most_common()]


def write(res, out, top):
    os.makedirs(out, exist_ok=True)
    tables = {"kernels": res["rows"],
              "how": rollup(res["rows"], lambda r: r["how"]),
              "ops": rollup(res["rows"], lambda r: f'{r["op_name"]} | {r["hlo"]}'),
              "lines": rollup(res["rows"], lambda r: r["user_frame"] or "(no user frame)")}
    for name, rows in tables.items():
        if rows:
            with open(os.path.join(out, name + ".csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    gpus = sorted({r["gpu"] for r in res["rows"]})
    lines = [f"program {res['program']} ({res['hlo']}); {len(res['rows'])} kernels on {len(gpus)} GPUs; "
             f"schedule of one run: {res['n_seq']} instructions, while loops: "
             + (", ".join(f"{w} x{t}" for w, (_, t) in res["whiles"].items()) or "none"),
             f"kernels left without a candidate: {sum(1 for r in res['rows'] if r['how'] == 'none')} in "
             f"{res['overflow']} windows (the 'none' share below; XLA reuses some kernels beyond deduplicated_name)",
             "share of GPU time by match class: " + ", ".join(f"{r['key']} {r['share']}%" for r in tables["how"])]
    for name in ("lines", "ops"):
        lines.append(f"\ntop {top} by GPU time, {name}:")
        lines += [f"  {r['share']:6.2f}%  {r['gpu_ms']:10.1f} ms  {r['key'][:150]}" for r in tables[name][:top]]
    text = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(text + "\n")
    return text


def self_test():
    # One run: a0 (anchor), graph [f1, gemm], while body {b_fus, copy_x} x2, a1 (anchor), graph [f2 shared, f2 shared]
    info = {"a0": ("kernel", "a0"), "f1": ("kernel", "f1"), "gemm.1": ("library", ""), "b_fus": ("kernel", "b_fus"),
            "copy.x": ("copy", "copy_x"), "a1": ("kernel", "a1"), "f2.1": ("kernel", "f2"), "f2.2": ("kernel", "f2")}
    seq = ["a0", "f1", "gemm.1", "b_fus", "copy.x", "b_fus", "copy.x", "a1", "f2.1", "f2.2"]
    one = [("a0", [("kernel", "a0")]),
           (None, [("kernel", "f1"), ("library", "")]),
           (None, [("kernel", "b_fus"), ("copy", "")]),  # loop iteration 1
           (None, [("kernel", "b_fus"), ("copy", "")]),  # loop iteration 2
           ("a1", [("kernel", "a1")]),
           (None, [("kernel", "f2"), ("kernel", "f2")])]
    res, ov = assign(seq, info, one + one)  # two runs: the second anchor a0 wraps
    want = [[("a0", "exact")], [("f1", "name"), ("gemm.1", "order")], [("b_fus", "order"), ("copy.x", "order")],
            [("b_fus", "order"), ("copy.x", "order")], [("a1", "exact")], [("f2.1", "order"), ("f2.2", "order")]]
    assert res == want + want and not ov, res
    # a third f2 has no candidate left: none, and its window overflows
    res, ov = assign(seq, info, one[:5] + [(None, [("kernel", "f2")] * 3)])
    assert res[5] == [("f2.1", "order"), ("f2.2", "order"), (None, "none")] and len(ov) == 1, res
    # a kernel whose own hlo_op isn't in the schedule (an async collective's inner op, which the captured
    # HLO doesn't name) keeps its exact label; its line comes from the next collective start, by order
    info2 = dict(info, **{"ag.start": ("collective", "")})
    seq2 = ["a0", "ag.start", "ag.done", "f1", "a1"]
    res, ov = assign(seq2, info2, [("a0", [("kernel", "a0")]), ("ag.inner", [("collective", "")]),
                                  (None, [("kernel", "f1")]), ("a1", [("kernel", "a1")])])
    assert res == [[("a0", "exact")], [("ag.start", "exact")], [("f1", "name")], [("a1", "exact")]] and not ov, res
    assert inst_kind("ag.done", "all-gather-done") is None and inst_kind("ag.start", "all-gather-start") == "collective"
    # ...but only a collective is matched that way: an unknown exact library kernel must not take the GEMM's slot
    res, ov = assign(seq, info, [("a0", [("kernel", "a0")]), ("x.inner", [("library", "")]),
                                 (None, [("kernel", "f1"), ("library", "")])])
    assert res[1:] == [[(None, "exact")], [("f1", "name"), ("gemm.1", "order")]] and not ov, res
    # a graph launched before the first anchor of the trace looks at the schedule's head
    res, _ = assign(seq, info, [(None, [("kernel", "f1")]), ("a1", [("kernel", "a1")])])
    assert res[0] == [("f1", "name")], res
    text = """HloModule m, is_scheduled=true

FileNames
1 "/home/u/train.py"
2 "/usr/lib/python3.12/site-packages/flax/linear.py"

FunctionNames
1 "step"
2 "__call__"

FileLocations
1 {file_name_id=1 function_name_id=1 line=10}
2 {file_name_id=2 function_name_id=2 line=422}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=2}

%body (p: f32[]) -> f32[] {
  %b_fus = f32[] fusion(%p), kind=kLoop, calls=%fb, metadata={op_name="jit(f)/mul" stack_frame_id=2}
  ROOT %copy.x = f32[] copy(%b_fus)
}

ENTRY %main (x: f32[]) -> f32[] {
  %a0 = f32[] fusion(%x), kind=kLoop, calls=%fa, metadata={op_name="jit(f)/add" deduplicated_name="a.9" stack_frame_id=1}
  %while.1 = f32[] while(%a0), condition=%cond, body=%body, backend_config={"known_trip_count":{"n":"2"}}
  ROOT %c = f32[] custom-call(%while.1), custom_call_target="__cublas$lt$matmul"
}
"""
    inst, comps, entry, whiles = parse_hlo(text)
    assert entry == "main" and whiles == {"while.1": ("body", 2)}, (entry, whiles)
    assert flatten(entry, comps, whiles) == ["a0", "b_fus", "copy.x", "b_fus", "copy.x", "c"]
    assert inst["a0"][1:] == ["jit(f)/add", "a_9", ["/home/u/train.py:10 step"]], inst["a0"]
    assert inst["b_fus"][3] == ["/usr/lib/python3.12/site-packages/flax/linear.py:422 __call__", "/home/u/train.py:10 step"]
    assert inst_kind("c", "custom-call") == "library" and inst_kind("copy.x", "copy") == "copy"
    assert kernel_kind("Cijk_x", {"a_9"}) == "library" and kernel_kind("a_9", {"a_9"}) == "kernel"
    print("self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", nargs="?", help="capture directory (holding <host>.xplane.pb) or the .xplane.pb")
    ap.add_argument("--program", type=int, help="program id (default: the one with the most GPU time)")
    ap.add_argument("--out", default="attribution", help="output directory (default: ./attribution)")
    ap.add_argument("--user-re", help="regex for the user's own frames (default: first frame outside libraries)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.capture:
        ap.error("capture is required")
    xplane = a.capture if a.capture.endswith(".xplane.pb") else glob.glob(os.path.join(a.capture, "*.xplane.pb"))[0]
    print(write(analyze(xplane, a.program, a.out, a.user_re), a.out, a.top))
    print(f"\ntables: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
