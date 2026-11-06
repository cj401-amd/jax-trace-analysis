#!/usr/bin/env python3
"""Check a jax.profiler trace.json.gz for known ROCm GPU-tracer defects.

usage: check_tracer.py TRACE.json[.gz] [--json OUT.json]
       check_tracer.py --self-test

Each check gives a count (and a share of GPU time where that applies) plus one example:
  join             GPU events with no host launch call (correlation_id); launches with negative lag
  device           launch calls whose host device_id differs from the GPU plane their events ran on
  graph_labels     GPU time whose hlo_op is only command_buffer_N (kernels inside HIP graphs)
  kd_collapse      graph launches with >1 kernel name but 1 distinct kernel_details
  kd_code          kernel names (per hlo_module) shown with >1 (regs, static_shared): fixed by the
                   code object, so more than one means some rows carry another kernel's record
  kd_zero          kernel_details with grid or block 0,0,0 (copy kernels before the fix)
  occupancy        kernel_details with occ_pct:0 (the tracer doesn't fill it in)
  memcpy           memcpy rows with kind:Unknown, size:0, or async:0 on an *Async call
  dropped          when every GPU makes the same launch calls (SPMD): kernel rows missing on some GPUs,
                   out of the rows expected; "skipped" when the launch sequences differ
"defect" means the count is above 0; graph_labels is "info" (the tracer can't label graph nodes yet).
"""
import argparse
import collections
import gzip
import json
import re


def load(path):
    with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
        return json.load(f)


def details(s):
    return dict(re.findall(r"(\w+):(\S+)", s or ""))


def check(trace):
    ev = trace["traceEvents"]
    pname = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "process_name"}
    host, gpu = {}, collections.defaultdict(list)
    for e in ev:
        if e.get("ph") != "X" or "correlation_id" not in e.get("args", {}):
            continue
        c = e["args"]["correlation_id"]
        p = pname.get(e["pid"], "")
        if p.startswith("/device:GPU:"):
            gpu[c].append((int(p.rsplit(":", 1)[1]), e))
        else:
            host[c] = e
    total = sum(e["dur"] for es in gpu.values() for _, e in es) or 1
    out = collections.OrderedDict()

    def put(name, count, example="", share=None, kind="defect", of=None):
        out[name] = dict(status=("ok" if count == 0 else kind), count=count, of=of,
                         share_gpu_time=None if share is None else round(100 * share / total, 2), example=example)

    unjoined = [c for c in gpu if c not in host]
    neg = [c for c in gpu if c in host and min(e["ts"] for _, e in gpu[c]) < host[c]["ts"]]
    put("join", len(unjoined) + len(neg), (f"corr {unjoined[0]}: no host call" if unjoined else
                                            f"corr {neg[0]}: kernel starts before its launch call" if neg else ""),
        share=sum(e["dur"] for c in unjoined + neg for _, e in gpu[c]), of=len(gpu))

    wrong = []
    for c, es in gpu.items():
        d = host.get(c, {}).get("args", {}).get("device_id")
        if d is not None and any(g != int(d) for g, _ in es):
            wrong.append(c)
    put("device", len(wrong), f"corr {wrong[0]}: host device_id {host[wrong[0]]['args']['device_id']}, "
                              f"ran on GPU:{gpu[wrong[0]][0][0]}" if wrong else "",
        share=sum(e["dur"] for c in wrong for _, e in gpu[c]), of=len(gpu))

    cb = [e for es in gpu.values() for _, e in es if e["args"].get("hlo_op", "").startswith("command_buffer")]
    put("graph_labels", len(cb), cb[0]["name"][:60] + " -> " + cb[0]["args"]["hlo_op"] if cb else "",
        share=sum(e["dur"] for e in cb), kind="info", of=sum(len(es) for es in gpu.values()))

    collapsed, multi = [], 0
    for c, es in gpu.items():
        ks = [e for _, e in es if "kernel_details" in e["args"]]
        if host.get(c, {}).get("name") == "hipGraphLaunch" and len({e["name"] for e in ks}) > 1:
            multi += 1
            if len({e["args"]["kernel_details"] for e in ks}) == 1:
                collapsed.append(c)
    put("kd_collapse", len(collapsed), (f"corr {collapsed[0]}: {len({e['name'] for _, e in gpu[collapsed[0]]})} kernel names, "
                                        f"1 kernel_details" if collapsed else ""),
        share=sum(e["dur"] for c in collapsed for _, e in gpu[c]), of=multi)

    code = collections.defaultdict(set)
    for es in gpu.values():
        for _, e in es:
            d = details(e["args"].get("kernel_details"))
            if d:
                code[(e["args"].get("hlo_module", ""), e["name"])].add((d.get("regs"), d.get("static_shared")))
    bad = sorted(k for k, v in code.items() if len(v) > 1)
    put("kd_code", len(bad), f"{bad[0][1][:50]} ({bad[0][0]}): {sorted(code[bad[0]])[:3]}" if bad else "", of=len(code))

    rows = [e for es in gpu.values() for _, e in es if "kernel_details" in e["args"]]
    zero = [e for e in rows if "0,0,0" in (details(e["args"]["kernel_details"]).get("grid"),
                                           details(e["args"]["kernel_details"]).get("block"))]
    put("kd_zero", len(zero), f"{zero[0]['name'][:50]}: {zero[0]['args']['kernel_details']}" if zero else "",
        share=sum(e["dur"] for e in zero), of=len(rows))
    occ0 = [e for e in rows if details(e["args"]["kernel_details"]).get("occ_pct") == "0"]  # a kernel that ran can't be at 0%
    put("occupancy", len(occ0), f"{occ0[0]['name'][:50]}: occ_pct:0" if occ0 else "",
        share=sum(e["dur"] for e in occ0), of=len(rows))

    mc, why = [], collections.Counter()
    for c, h in host.items():
        if "memcpy_details" not in h.get("args", {}):
            continue
        d = details(h["args"]["memcpy_details"])
        r = [k for k, bad_ in (("kind:Unknown", d.get("kind") == "Unknown"), ("size:0", d.get("size") == "0"),
                                ("async:0 on an Async call", h["name"].endswith("Async") and d.get("async") == "0")) if bad_]
        if r:
            mc.append((c, h["name"], r))
            why.update(r)
    n_mc = sum(1 for h in host.values() if "memcpy_details" in h.get("args", {}))
    put("memcpy", len(mc), (f"{mc[0][1]} corr {mc[0][0]}: " + ", ".join(mc[0][2]) + "; all: "
                            + ", ".join(f"{k} {v}" for k, v in why.most_common()) if mc else ""), of=n_mc)

    # SPMD: when every GPU makes the same launch calls, a launch with fewer kernels on one GPU lost rows.
    # A row lost on every GPU at once stays invisible.
    seqs = collections.defaultdict(list)
    for c, h in host.items():
        if h.get("args", {}).get("device_id") is not None and not h["name"].startswith("hipMemcpy"):
            seqs[int(h["args"]["device_id"])].append((h["ts"], c, h["name"]))
    groups = collections.defaultdict(list)  # kernel-launch sequence -> GPUs making it; copies differ by GPU
    for d in sorted(seqs):
        groups[tuple(a for *_, a in sorted(seqs[d]))].append(d)
    devs = max(groups.values(), key=len, default=[])
    lists = [sorted(seqs[d]) for d in devs]
    left = [d for d in sorted(seqs) if d not in devs]
    if len(devs) < 2:
        out["dropped"] = dict(status="skipped", count=0, of=None, share_gpu_time=None,
                              example="no two GPUs make the same kernel-launch sequence (not SPMD, or cut differently)")
    else:
        lost, expected, ex = 0, 0, ""
        for k in range(len(lists[0])):
            per = [(d, [e["name"] for _, e in gpu.get(l[k][1], [])]) for d, l in zip(devs, lists)]
            full = max((n for _, n in per), key=len)
            expected += len(full) * len(per)
            for d, names in per:
                if len(names) < len(full):
                    lost += len(full) - len(names)
                    miss = collections.Counter(full) - collections.Counter(names)
                    ex = ex or f"GPU:{d} launch #{k} ({lists[0][k][2]}): missing " + ", ".join(sorted(miss))[:80]
        put("dropped", lost, ex + (f"; left out: " + ", ".join(f"GPU:{d}" for d in left) + " (other launch sequence)" if left else ""),
            of=expected)
    return out


def report(res):
    lines = [f"{'check':<14}{'status':<8}{'count':>8}  {'of':>8}  {'GPU time %':>10}  example"]
    for k, r in res.items():
        share = "" if r["share_gpu_time"] is None else f"{r['share_gpu_time']:.2f}"
        lines.append(f"{k:<14}{r['status']:<8}{r['count']:>8}  {r['of'] if r['of'] is not None else '':>8}  {share:>10}  {r['example']}")
    return "\n".join(lines)


def self_test():
    def M(pid, name):
        return {"ph": "M", "pid": pid, "tid": 0, "name": "process_name", "args": {"name": name}}

    def X(pid, name, ts, dur, **a):
        return {"ph": "X", "pid": pid, "tid": 0, "name": name, "ts": ts, "dur": dur, "args": a}

    kd = "regs:{} static_shared:{} grid:1,1,1 block:64,1,1"
    t = {"traceEvents": [
        M(1, "/device:GPU:0"), M(2, "/device:GPU:1"), M(9, "/host:CPU"),
        # graph 1: two kernel names, one kernel_details -> collapsed; also hlo_op is command_buffer_1
        X(9, "hipGraphLaunch", 0, 1, correlation_id=1, device_id=0),
        X(1, "a.kd", 5, 10, correlation_id=1, hlo_op="command_buffer_1", hlo_module="m", kernel_details=kd.format(8, 0)),
        X(1, "b.kd", 15, 10, correlation_id=1, hlo_op="command_buffer_1", hlo_module="m", kernel_details=kd.format(8, 0)),
        # b launched alone shows its own regs: b has two (regs, static_shared) -> kd_code
        X(9, "hipModuleLaunchKernel", 30, 1, correlation_id=2, device_id=0),
        X(1, "b.kd", 40, 10, correlation_id=2, hlo_op="b", hlo_module="m", kernel_details=kd.format(32, 512)),
        # launched for device 0 but ran on GPU:1 -> device; starts before its launch call -> join
        X(9, "hipModuleLaunchKernel", 60, 1, correlation_id=3, device_id=0),
        X(2, "c.kd", 55, 20, correlation_id=3, hlo_op="c", hlo_module="m", kernel_details=kd.format(4, 0)),
        # GPU event with no host call -> join
        X(2, "d.kd", 100, 5, correlation_id=4, hlo_op="d", hlo_module="m", kernel_details=kd.format(4, 0)),
        # memcpy: an Async call recorded as kind Unknown, size 0, async 0
        X(9, "hipMemcpyDtoDAsync", 200, 1, correlation_id=5, device_id=1, memcpy_details="kind:Unknown size:0 dest:1 async:0"),
        X(2, "MemcpyD2D", 205, 5, correlation_id=5, memcpy_details="kind:Unknown size:0 dest:1 async:0",
          kernel_details="regs:0 static_shared:0 grid:0,0,0 block:0,0,0 occ_pct:0"),  # a copy kernel with empty dims
        X(9, "hipMemcpyHtoD", 300, 1, correlation_id=6, device_id=1, memcpy_details="kind:HtoD size:64 dest:1 async:0"),
        X(2, "MemcpyH2D", 305, 5, correlation_id=6, memcpy_details="kind:HtoD size:64 dest:1 async:0"),
        # graph 2: two kernel names with their own kernel_details -> not collapsed
        X(9, "hipGraphLaunch", 400, 1, correlation_id=7, device_id=0),
        X(1, "e.kd", 405, 5, correlation_id=7, hlo_op="command_buffer_2", hlo_module="m", kernel_details=kd.format(16, 0)),
        X(1, "f.kd", 410, 5, correlation_id=7, hlo_op="command_buffer_2", hlo_module="m", kernel_details=kd.format(24, 0)),
    ]}
    r = check(t)
    got = {k: (v["status"], v["count"], v["of"], v["share_gpu_time"]) for k, v in r.items()}
    # 9 GPU events, 75 us: join = d (5) + c (20); device = c; graph labels = a, b, e, f (30); collapse = a + b (20)
    want = {"join": ("defect", 2, 7, 33.33), "device": ("defect", 1, 7, 26.67), "graph_labels": ("info", 4, 9, 40.0),
            "kd_collapse": ("defect", 1, 2, 26.67), "kd_code": ("defect", 1, 7, None), "memcpy": ("defect", 1, 2, None),
            "kd_zero": ("defect", 1, 8, 6.67), "occupancy": ("defect", 1, 8, 6.67), "dropped": ("skipped", 0, None, None)}
    assert got == want, got
    assert "kind:Unknown 1, size:0 1, async:0 on an Async call 1" in r["memcpy"]["example"], r["memcpy"]["example"]
    assert "b.kd (m)" in r["kd_code"]["example"], r["kd_code"]["example"]
    assert r["dropped"]["status"] == "skipped", r["dropped"]  # the GPUs above don't run the same launches

    # SPMD: two GPUs make the same launches; GPU:1's graph is one kernel short -> 1 dropped row
    def spmd(drop):
        ev = [M(1, "/device:GPU:0"), M(2, "/device:GPU:1"), M(3, "/device:GPU:2"), M(9, "/host:CPU")]
        ev += [X(9, "hipMemcpyHtoDAsync", 0, 1, correlation_id=99, device_id=0)]  # copies don't count
        ev += [X(9, "hipModuleLaunchKernel", 30, 1, correlation_id=299, device_id=2), X(3, "w.kd", 31, 1, correlation_id=299)]
        for g, pid in ((0, 1), (1, 2), (2, 3)):
            for k, (api, names) in enumerate((("hipModuleLaunchKernel", ["x.kd"]), ("hipGraphLaunch", ["y.kd", "z.kd"]))):
                c = 100 * g + k
                ev.append(X(9, api, 10 * k, 1, correlation_id=c, device_id=g))
                ev += [X(pid, n, 10 * k + 2 + j, 1, correlation_id=c, hlo_op="h") for j, n in enumerate(names)
                       if not (drop and g == 1 and n == "z.kd")]
        return {"traceEvents": ev}
    d = check(spmd(True))["dropped"]
    assert (d["status"], d["count"], d["of"]) == ("defect", 1, 6) and "GPU:1" in d["example"] and "z.kd" in d["example"], d
    assert "left out: GPU:2" in d["example"], d
    assert check(spmd(False))["dropped"]["status"] == "ok"
    print("self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--json", help="also write the results as JSON")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.trace:
        ap.error("trace is required")
    res = check(load(a.trace))
    print(f"trace: {a.trace}\n" + report(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
