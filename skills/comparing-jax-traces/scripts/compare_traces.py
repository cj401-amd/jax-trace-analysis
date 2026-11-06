#!/usr/bin/env python3
"""Compare two jax.profiler captures of the same workload (before / after a change).

usage: compare_traces.py A.trace.json.gz B.trace.json.gz [--top N]
       compare_traces.py --self-test

Reuses skip_stats.py (analyzing-jax-launch-overhead) and check_tracer.py (checking-rocm-tracer-defects)
from this plugin. Per GPU-step it compares the work (kernel counts and names), step latency (IL), idle
time (IL - busy), each GPU's main-stream verdict, the kernels whose time changed most, and the tracer
checks. A change counts as real only when the two captures' ranges over steps and GPUs don't overlap.
"""
import argparse
import collections
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("analyzing-jax-launch-overhead", "checking-rocm-tracer-defects"):
    sys.path.insert(0, os.path.join(HERE, "..", "..", d, "scripts"))
import check_tracer  # noqa: E402
import skip_stats  # noqa: E402


def summarize(trace):
    res = skip_stats.analyze(trace)
    steps = res["steps"]
    per = collections.Counter()  # kernel name -> total us over all GPU-steps
    cnt = collections.Counter()
    for r in res["rows"]:
        if r["step"] is not None and r["cat"] != "memcpy":
            per[r["name"]] += r["dur"]
            cnt[r["name"]] += 1
    n = len(steps) or 1
    main = {}
    for s in res["streams"]:
        if s["gpu"] not in main or s["compute"] > main[s["gpu"]]["compute"]:
            main[s["gpu"]] = s
    verdicts = {}
    for g, s in main.items():
        sh = {k: s[k] / s["span"] if s["span"] else 0.0 for k in skip_stats.VERDICT}
        verdicts[g] = (skip_stats.verdict(sh), sh)
    return dict(n_gpu_steps=len(steps), gpus=len(main), il=[s["il"] / 1e3 for s in steps],
                idle=[s["idle_merged"] / 1e3 for s in steps], kernels=[s["kernels"] for s in steps],
                time={k: v / n / 1e3 for k, v in per.items()}, count={k: v / n for k, v in cnt.items()},
                verdicts=verdicts, checks=check_tracer.check(trace))


def rng(xs):
    return f"{statistics.median(xs):.1f} [{min(xs):.1f}-{max(xs):.1f}]" if xs else "-"


def changed(a, b):
    """'up' / 'down' if the ranges don't overlap, else 'same'."""
    if not a or not b:
        return "same"
    return "up" if min(b) > max(a) else "down" if max(b) < min(a) else "same"


def compare(A, B, top=10):
    out, summary = [], []
    p = out.append
    p(f"A: {A['gpus']} GPUs, {A['n_gpu_steps']} GPU-steps   B: {B['gpus']} GPUs, {B['n_gpu_steps']} GPU-steps")
    p("\n== Same work? (per GPU-step) ==")
    p(f"kernels: A {rng(A['kernels'])}, B {rng(B['kernels'])}")
    only_a = sorted(set(A["time"]) - set(B["time"]), key=lambda k: -A["time"][k])
    only_b = sorted(set(B["time"]) - set(A["time"]), key=lambda k: -B["time"][k])
    for tag, names, S in (("only in A", only_a, A), ("only in B", only_b, B)):
        p(f"kernel names {tag}: {len(names)}" + "".join(f"\n  {S['time'][k]:8.3f} ms  {k[:90]}" for k in names[:5]))
    same_work = not only_a and not only_b and changed(A["kernels"], B["kernels"]) == "same"
    summary.append("same kernels in both" if same_work else
                   f"the work differs: {len(only_a)} kernel names only in A, {len(only_b)} only in B")

    p("\n== Step latency and idle (ms per GPU-step, median [min-max]) ==")
    for name, key in (("step latency (IL)", "il"), ("idle (IL - busy)", "idle")):
        c = changed(A[key], B[key])
        d = 100 * (statistics.median(B[key]) - statistics.median(A[key])) / (statistics.median(A[key]) or 1)
        p(f"{name:<18} A {rng(A[key]):<24} B {rng(B[key]):<24} median {d:+.1f}%  "
          + ("ranges overlap: within run-to-run noise" if c == "same" else f"ranges don't overlap: {c}"))
        if key == "il":
            summary.append(f"step latency {d:+.1f}% (median), " + ("within run-to-run noise" if c == "same" else
                                                                   "a real change: the ranges don't overlap"))

    p("\n== Verdict on each GPU's main compute stream ==")
    flips = 0
    for g in sorted(set(A["verdicts"]) | set(B["verdicts"]), key=lambda g: int(g.rsplit(":", 1)[1])):
        va, vb = A["verdicts"].get(g), B["verdicts"].get(g)
        fmt = lambda v: "-" if not v else f"{v[0].split(' (')[0]} (compute {100 * v[1]['compute']:.0f}%, host-idle {100 * v[1]['host_idle']:.0f}%)"
        flips += bool(va and vb and va[0] != vb[0])
        p(f"{g}: A {fmt(va)}   B {fmt(vb)}")
    summary.append("verdicts unchanged" if not flips else f"verdict changed on {flips} GPUs")

    p(f"\n== Biggest kernel-time changes (ms per GPU-step) ==")
    names = set(A["time"]) | set(B["time"])
    delta = sorted(names, key=lambda k: -abs(B["time"].get(k, 0) - A["time"].get(k, 0)))
    p(skip_stats.table(["kernel", "A ms", "B ms", "change ms", "A count", "B count"],
                       [[k[:70], f"{A['time'].get(k, 0):.3f}", f"{B['time'].get(k, 0):.3f}",
                         f"{B['time'].get(k, 0) - A['time'].get(k, 0):+.3f}", f"{A['count'].get(k, 0):.1f}",
                         f"{B['count'].get(k, 0):.1f}"] for k in delta[:top]]))

    p("\n== Tracer checks (count of / GPU time %) ==")
    fixed, new, both = [], [], []
    for k in A["checks"]:
        a, b = A["checks"][k], B["checks"][k]
        cell = lambda r: (f"{r['status'] if r['status'] == 'skipped' else r['count']:>7} of {r['of'] if r['of'] is not None else '-':<7} "
                          f"{r['share_gpu_time'] if r['share_gpu_time'] is not None else '':>6}")
        p(f"{k:<14} A {cell(a)}   B {cell(b)}")
        if (a["status"], b["status"]) == ("defect", "ok"):
            fixed.append(f"{k} {a['count']} -> 0")
        elif (a["status"], b["status"]) == ("ok", "defect"):
            new.append(f"{k} 0 -> {b['count']}")
        elif a["status"] == b["status"] == "defect":
            both.append(k)
    parts = [f"fixed in B: {', '.join(fixed)}"] * bool(fixed) + [f"new in B: {', '.join(new)}"] * bool(new)
    summary.append(f"tracer checks: {'; '.join(parts) or 'no status change'}" + (f"; failing in both: {', '.join(both)}" if both else ""))
    return "== Summary ==\n" + "\n".join(f"- {s}" for s in summary) + "\n\n" + "\n".join(out)


def self_test():
    def trace(gemm_us, extra, kd_same, step_us=100):
        def M(pid, tid, kind, name):
            return {"ph": "M", "pid": pid, "tid": tid, "name": kind, "args": {"name": name}}

        def X(pid, tid, ev, ts, dur, **a):
            return {"ph": "X", "pid": pid, "tid": tid, "name": ev, "ts": ts, "dur": dur, "args": a}
        ev, corr = [M(1, 0, "process_name", "/device:GPU:0"), M(9, 0, "process_name", "/host:CPU"),
                    M(1, 0, "thread_name", "Stream #0"), M(9, 1, "thread_name", "python3")], 0
        for s in range(3):
            t0 = s * step_us
            ev.append(X(9, 1, "train", t0, step_us, step_num=str(s)))
            corr += 1
            ev.append(X(9, 2, "hipGraphLaunch", t0 + 1, 1, correlation_id=corr, device_id=0))
            ks = [("fusion_1.kd", 10, "regs:8 static_shared:0"), ("Cijk_gemm.kd", gemm_us, "regs:8 static_shared:0" if kd_same else "regs:96 static_shared:4096")]
            ks += [("fusion_9.kd", 5, "regs:8 static_shared:0")] if extra else []
            t = t0 + 5
            for name, d, kd in ks:
                ev.append(X(1, 0, name, t, d, correlation_id=corr, hlo_op="command_buffer_1", hlo_module="m", kernel_details=kd))
                t += d
        return {"traceEvents": ev}
    A = summarize(trace(gemm_us=40, extra=False, kd_same=True))
    B = summarize(trace(gemm_us=20, extra=True, kd_same=False))
    text = compare(A, B)
    assert A["time"]["Cijk_gemm.kd"] == 0.04 and B["time"]["Cijk_gemm.kd"] == 0.02, (A["time"], B["time"])
    assert "kernel names only in B: 1" in text and "fusion_9.kd" in text, text
    assert "the work differs: 0 kernel names only in A, 1 only in B" in text
    assert "fixed in B: kd_collapse 3 -> 0" in text, text  # A: each graph's 2 names share one kernel_details
    # IL = last kernel end - step start: A 5 + 10 + 40 = 55 us, B 5 + 10 + 20 + 5 = 40 us -> -27.3%
    assert "step latency -27.3% (median), a real change" in text, text
    same = compare(A, summarize(trace(gemm_us=40, extra=False, kd_same=True)))
    assert "same kernels in both" in same and "within run-to-run noise" in same, same
    assert "\n- tracer checks: no status change; failing in both: kd_collapse\n" in same, same
    print("self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", nargs="?", help="trace.json.gz of capture A (before)")
    ap.add_argument("b", nargs="?", help="trace.json.gz of capture B (after)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--self-test", action="store_true")
    x = ap.parse_args()
    if x.self_test:
        return self_test()
    if not (x.a and x.b):
        ap.error("two traces are required")
    print(f"A: {x.a}\nB: {x.b}\n")
    print(compare(summarize(skip_stats.load(x.a)), summarize(skip_stats.load(x.b)), x.top))


if __name__ == "__main__":
    main()
