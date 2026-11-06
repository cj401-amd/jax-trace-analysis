---
name: checking-rocm-tracer-defects
description: Use when deciding how far a jax.profiler / XProf trace from ROCm GPUs can be trusted, or checking whether a profiler or tracer fix worked - symptoms like kernels in HIP graphs all showing the same kernel_details, hlo_op=command_buffer_N labels, memcpy rows with kind:Unknown size:0, kernels on the wrong GPU, kernel rows missing on some GPUs, or GPU events with no matching host launch (correlation_id).
---

# Checking ROCm tracer defects

## Overview

`scripts/check_tracer.py` runs nine checks on one `trace.json.gz` in about a second,
using only the standard library. Each check gives a count, what it's out of, the share
of GPU time where that applies, and one example. Use it before trusting per-kernel
numbers from a trace, and before and after a tracer change to show that the change
worked.

```bash
python3 -P <this skill's base directory>/scripts/check_tracer.py <capture>/<host>.trace.json.gz [--json out.json]
python3 -P <this skill's base directory>/scripts/check_tracer.py --self-test
```

## The checks

| Check | Defect it catches | What a nonzero count means for analysis |
|---|---|---|
| `join` | GPU events with no host launch call, or starting before it | launch lag and host attribution are unknown for those events |
| `device` | the launch's host `device_id` ≠ the GPU it ran on | per-GPU tables misplace that work |
| `graph_labels` (info) | kernels inside HIP graphs labelled only `command_buffer_N` | their HLO op must be recovered from the HLO (investigating-jax-traces does this) |
| `kd_collapse` | a graph launch whose kernels all show one `kernel_details` | grid, block and registers of graph kernels are wrong |
| `kd_code` | one kernel name with several `(regs, static_shared)` | some rows carry another kernel's record; those fields can't be used |
| `kd_zero` | `kernel_details` with grid or block `0,0,0` (copy kernels) | launch dimensions of those kernels are missing |
| `occupancy` | `occ_pct:0` (the tracer doesn't fill it in) | occupancy can't come from the trace |
| `memcpy` | `kind:Unknown`, `size:0`, or `async:0` on an `*Async` call | copy sizes, directions and bandwidth can't come from the trace |
| `dropped` | SPMD only: kernel rows missing on some GPUs. Compares the largest group of GPUs that make the same kernel-launch sequence, and names any GPU left out | busy time and kernel counts on those GPUs are low; a GPU left out may itself have lost launch records |

## Measured reference (MaxText llama2-7b, 8 × MI355X, 2026-11-02 captures)

| Check | Control build | With the kernel_details fix |
|---|---|---|
| `kd_collapse` | 5,592 of 5,592 multi-kernel graph launches (55% of GPU time) | 0 of 5,592 |
| `kd_code` | 129 of 179 kernel names | 0 |
| `kd_zero` | 9,333 copy kernels (1.4% of GPU time) | 0 |
| `occupancy` | 100% of rows | 100% of rows |
| `memcpy` | 9,333 of 9,333 | 9,333 of 9,333 (not part of that fix) |
| `graph_labels` | 55% of GPU time | 56% (not part of that fix) |
| `dropped` | 3 rows (GPU:0 left out) | 1 row (GPU:0 and GPU:5 left out) |
| `join`, `device` | 0 | 0 |

## XProf views to distrust on such traces (measured on the control capture)

- **`hlo_stats`, `op_profile`, roofline, overview top ops:** they cover only the
  kernels launched outside graphs. `hlo_stats` total equals the non-graph kernel
  time, about 45% of GPU time here.
- **`framework_op_stats`:** reports 48.7% IDLE, while the GPUs are about 96% busy
  within a step.
- **Overview page:** "No step time measured", even with step annotations present.
- **Device peaks:** XProf's `memory_bandwidth` for gfx950 is 3.89 TB/s, against the
  8 TB/s spec, so its HBM-utilisation percentages are about twice too high.

## Using it to verify a fix

1. Run it on a capture without the fix and one with it, from the same workload and
   config.
2. The fixed check must go to 0. The other checks must not change, except `dropped`,
   which varies from run to run (3 vs 1 rows above). If they do change, the two
   captures differ in more than the fix.
3. For the performance side (step time, kernels, idle), compare the two captures with
   the comparing-jax-traces skill.

## Limits

- **One file only:** it checks the `trace.json.gz`, not the `.xplane.pb` (XProf's
  conversion keeps these fields).
- **Dropped rows:** these show up only where GPUs make the same launches. A row lost
  on every GPU at once, or on a non-SPMD capture, stays invisible.
- **memcpy:** `async:0` on an `*Async` call is flagged as a defect, because the API
  name says async.
