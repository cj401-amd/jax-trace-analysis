---
name: analyzing-jax-launch-overhead
description: Use when a jax.profiler.trace() or XProf capture (trace.json.gz) needs a quick CPU-bound vs GPU-bound verdict, kernel launch or queuing time (TKLQT, the SKIP paper arXiv 2504.11750), GPU idle time and the host work behind it (e.g. command_buffer::update), or kernel-fusion candidates, including traces with HIP/CUDA graphs (XLA command buffers) and multi-GPU RCCL/NCCL collectives.
---

# Analyzing JAX launch overhead

## Overview

`scripts/skip_stats.py` turns a `jax.profiler.trace()` capture into the SKIP
paper's metrics, plus a per-stream time budget that gives the verdict. The
budget splits each stream's time into compute, collective, host-idle, launch
and wait, and names the host event running during host-idle time. The
definitions are fixed and self-tested, so numbers compare across traces, and
it runs in seconds.

**Core rule:** take the verdict from the stream budget, not from TKLQT. With
async dispatch and graph launches, TKLQT is mostly queuing. For idle time, use
the `idle ms (IL - busy)` column, not the paper's IL − Σt_k.

## Run it

`S` is `scripts/skip_stats.py` under this skill's base directory, the one shown
when the skill loads.

```bash
S=<this skill's base directory>/scripts/skip_stats.py
python3 -P $S --self-test
python3 -P $S <logdir>/plugins/profile/<run>/<host>.trace.json.gz [--dump PREFIX] [--out DIR]
```

- **Input:** `jax.profiler.trace()` writes `trace.json.gz` next to
  `.xplane.pb`. The script needs only the standard library.
- **Output:** the summary on stdout and in `DIR/summary.txt` (default
  `skip_out/<run>/` in the current directory), plus CSVs: `rows` (one per GPU
  event), `steps`, `streams`, `chains`, `chain_list` and `ops`.
- **`--dump`:** the XLA dump prefix (from `--xla_dump_to`) of the same
  compile, e.g. `dump/module_0123.jit_train_step`. It names kernels inside
  command buffers, which the ROCm tracer labels `command_buffer_N`. A dump
  from a rerun can be a different compile.

## Read the output in this order

| Section | Take from it |
|---|---|
| First line | Join rate (GPU events linked by `correlation_id`), negative lags (clock trouble), overlaps (concurrent graph nodes) |
| Verdict | Per GPU, the main compute stream's split. `mixed` means the top two shares are within 5 points. The indented line names the innermost host event during host-idle time. |
| Stream time | The same split for every stream |
| Paper metrics | Kernels per step must be the same in every step (this checks the step assignment); then TKLQT, AKD, IL |
| TKLQT split | Queue, launch and wait shares |
| Top kernels, chains, attribution | The paper's top-k and fusion candidates; attribution classes exact, name, ambiguous and none |

## Traps in the numbers

- **TKLQT far larger than the trace:** kernels are queued behind earlier work,
  because the host runs ahead or a graph launch arrives as a burst. That is
  not GPU saturation.
- **The paper's idle, IL − Σt_k,** is wrong in two ways. It double-counts
  streams that overlap (a collective stream next to the compute stream): on
  llama2-7b × 8 it came out at −186 to −281 ms per step. And when the host runs
  ahead, it counts other steps' kernels as idle: 306 ms where the GPU was 98%
  busy. The `idle ms (IL - busy)` column fixes both; busy is the time with any
  kernel running.
- **RCCL/NCCL kernel time** includes waiting for peers, so it inflates busy
  time and AKD.
- **PS = 1 chains are rare:** XLA reuses one kernel for identical fusions,
  and all RCCL kernels share one name. The paper's fusion speedup counts
  launches, not time; inside graphs the gaps are about 0.
- **The launch floor** is the smallest lag on an idle stream in the same
  trace, not a null-kernel benchmark.
- **The Python tracer** inflates host time, so traced host-idle overstates
  the untraced share.

## Report

Give each GPU's verdict with its split and the host-idle cause, or a range
across GPUs. Then give the join rate and the kernels-per-step check, the
numbers the user asked for, and the traps above that apply to this trace.

## Before trusting a verdict

- The self-test passes, the join rate is about 100%, and kernels per step are
  constant.
- Work from the given trace only. A cause stays *inferred* until the user
  captures a new trace with the cause removed. Give the estimate, e.g. step −
  host-idle, and the change to try; don't rerun the workload.

## Going deeper

If the user wants what to change, the operators on each stream of each GPU, or the Python
lines behind them, use the investigating-jax-traces skill. It works from the given trace
files only.

## Not covered

- **The paper's batch sweep:** one trace is one batch size. The sweep needs one
  capture per batch size from the user.
- **Command-buffer GEMMs** (hipBLASLt/cuBLAS): `--dump` can't name them.
- **CUDA:** tested on ROCm traces (MI355X) only. CUDA should work, since
  CUPTI API events carry `correlation_id`, but it hasn't been run.
