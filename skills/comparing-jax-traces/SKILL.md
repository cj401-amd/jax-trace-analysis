---
name: comparing-jax-traces
description: Use when comparing two jax.profiler.trace() / XProf captures of the same workload - before and after a code, compiler, flag, profiler, or driver change, or two builds or runs - to tell whether step time, idle time, the CPU/GPU-bound verdict, kernels, or trace labels actually changed or are within run-to-run noise.
---

# Comparing JAX traces

## Overview

`scripts/compare_traces.py A B` puts two captures side by side, per GPU-step, in about
5 seconds, using only the standard library. It reuses this plugin's `skip_stats.py`
and `check_tracer.py`, so the numbers mean what they mean in those skills. A summary
comes first, then the evidence. Like the rest of the plugin, it uses the trace files
only.

```bash
python3 -P <this skill's base directory>/scripts/compare_traces.py <A>/<host>.trace.json.gz <B>/<host>.trace.json.gz [--top N]
python3 -P <this skill's base directory>/scripts/compare_traces.py --self-test
```

## What it reports

| Section | Question it answers |
|---|---|
| Summary | Same work? Did step time change beyond noise? Did a verdict flip? Which tracer checks moved? |
| Same work? | kernels per GPU-step, and kernel names found in only one capture (top by time) |
| Step latency and idle | IL and idle (IL − busy) per GPU-step, median [min–max] |
| Verdict | each GPU's main compute stream: verdict, compute %, host-idle % |
| Biggest kernel-time changes | ms per GPU-step, and counts, for the kernels that moved most |
| Tracer checks | the checking-rocm-tracer-defects checks side by side |

A change is called real only when the two ranges over steps and GPUs don't overlap;
otherwise the report says "within run-to-run noise". With 3 profiled steps that bar
is low, so say how many steps each range rests on.

## How to read it

- **Different work:** kernel names in only one capture, or different kernels per
  step, mean the two runs didn't do the same work (different model, batch,
  sharding or compile). Compare timings only after explaining that.
- **Collectives in the kernel list:** RCCL/NCCL kernel time includes waiting for
  peers. A change there can be another GPU arriving later, not a faster or slower
  collective.
- **One capture each:** two captures are one run each. For a performance claim,
  ask the user for repeated captures; don't rerun anything yourself.

## Example (MaxText llama2-7b × 8, control vs kernel_details fix)

- **Work:** the same kernels, 3,847 per GPU-step.
- **Step latency:** −2.1%, within noise.
- **Verdicts:** unchanged, GPU-bound with compute 86–88%.
- **Tracer checks:** `kd_collapse` 5,592 → 0 and `kd_code` 129 → 0.

So the fix changed the labels, not the speed.
