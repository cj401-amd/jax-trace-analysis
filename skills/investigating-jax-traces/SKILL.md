---
name: investigating-jax-traces
description: Use when a jax.profiler.trace() or XProf capture (xplane.pb, trace.json.gz) needs a deep investigation rather than a quick verdict - what to change to make a step faster, the operators on each stream of each GPU, GPU kernels tied back to HLO ops and Python source lines, critical path and stragglers across GPUs, or where the trace can't be trusted - including HIP/CUDA graph (XLA command buffer) kernels labelled command_buffer_N.
---

# Investigating JAX traces

## Overview

A seven-step investigation of one capture: the verdict, reconciliation, swimlanes,
kernel → HLO → Python line, roofline, what to change, and the trust list.

**Everything comes from the capture the user gave**: `<host>.xplane.pb` and/or
`<host>.trace.json.gz`. Don't rerun the workload, dump XLA or run another profiler.
Where the trace can't answer, put it on the "can't be trusted" list instead.

## Goal

After the report, the reader knows:

- **(a)** what to change to make the step faster, as an estimate with its assumptions;
- **(b)** where this trace can't support the attribution, with counts.

## Tools

Script paths are relative to this skill's base directory, the one shown when the
skill loads. Run from a scratch directory: both scripts write their outputs into
the current directory.

```bash
B=<this skill's base directory>
python3 -P $B/../analyzing-jax-launch-overhead/scripts/skip_stats.py <capture>/<host>.trace.json.gz
python3 -P $B/scripts/attribute_kernels.py <capture dir> [--program ID] [--user-re REGEX]
```

- **`skip_stats.py`** (standard library only): the verdict, the per-stream time budget,
  the host event during idle time, and the SKIP-paper metrics. How to read it:
  the analyzing-jax-launch-overhead skill.
- **`attribute_kernels.py`** (needs the `jax` and `xprof` packages): every GPU
  kernel → HLO instruction → op_name → Python line. It reads only the `.xplane.pb`,
  copying it to `./attribution/xplane/` first, because XProf writes the HLO protos
  next to it. It writes `kernels.csv`, `ops.csv`, `lines.csv`, `how.csv`,
  `summary.txt` and the program's HLO text to `./attribution/`.
  - **exact:** the tracer's own `hlo_op`, for kernels launched one per call.
  - **name:** a graph kernel whose name is the only candidate in its window.
  - **order:** shared kernels, library GEMMs, copies and collectives, matched in
    schedule order.
  - **none:** no candidate left.
  - Windows are the instructions between neighbouring exact kernels, in the
    program's schedule with while loops expanded by their known trip count.
  - **Validation:** on a MiniGPT capture it chose the same instruction for all
    147,840 kernels as a pipeline that had agreed 100% with a command-buffers-off
    run of the same program.

## Five techniques, each with its tool

| Field | Technique | Tool here |
|---|---|---|
| Performance engineering | Critical path; roofline | The stream budget from `skip_stats.py`; a roofline only with the right peaks (see traps) |
| Distributed tracing | Join host and GPU events by correlation ID; find stragglers | `skip_stats.py` → `rows.csv`; the spread of step end times across GPUs |
| Compilers | Source provenance through compiler passes | `attribute_kernels.py` → `lines.csv` |
| Forensic accounting | Totals must reconcile across independent records | The checks in step 2 |
| Information design | Aligned swimlanes, one per GPU stream | One step, every GPU × stream, on one time axis |

## Steps

The plan is these seven steps, all run on the given files. A question only a new capture
could answer (what a host call does inside, the effect of a fix, a batch sweep) goes in
step 7's "needs a new capture" list, and whether to capture is the user's call.

1. **Verdict and host-side cause:** run `skip_stats.py`.
2. **Reconcile.** Each check either holds or goes on the trust list:
   - GPU events per device: json = xplane;
   - every GPU event joins one host launch call;
   - kernels per step are constant;
   - all GPUs do the same work (equal kernel counts per step, or the difference
     explained);
   - for each stream, compute + collective + memcpy + idle shares add to the span.
3. **Swimlanes.** For the median step, put every GPU × stream on one time axis, plus
   the launch thread's host events. This shows the critical path and the stragglers:
   which GPU finishes last, and what that GPU waits on.
4. **Kernel → HLO → Python line:** run `attribute_kernels.py`.
   - Report each match class's share of GPU time, and the kernels left without a
     candidate.
   - Use `lines.csv` and `ops.csv` for the per-line and per-op tables.
   - `--user-re` picks which frames count as the user's code. By default it's the
     first frame outside site-packages and jax.
5. **Roofline,** only for ops whose FLOPs and bytes you can get from their HLO
   shapes: the `<program>(<id>).hlo.txt` that `attribute_kernels.py` writes to `./attribution/`.
   Use the peak for the op's data type, cited from the vendor spec and labelled
   *read*; the trace can't measure peaks.
6. **What to change.** Rank the causes by their share of the critical path. For
   each, estimate the step time if it went away, e.g. step − host-idle on the
   critical stream. Label it *inferred, not run*, and name the knob as a hypothesis
   for the user to test (`donate_argnums`, command buffers off, fewer or larger
   collectives).
7. **Where this trace can't be trusted:** list each item with counts or shares. Then add the
   "needs a new capture" list: each open question, and what capture would answer it.

## Traps (measured on ROCm MI355X captures)

- **XProf's op views leave out graph kernels.** `hlo_stats` covered 37.6% of GPU
  kernel time on a 4-GPU capture. The other 62.4% (command-buffer kernels) shows
  in `kernel_stats` as op "XlaModule". Check coverage before quoting `hlo_stats`,
  `op_profile` or `roofline_model`.
- **XProf's roofline peaks for gfx950** are 2,517 TFLOP/s and 3,624 GiB/s, whatever
  the data type. FP32 GEMMs need the FP32 peak.
- **RCCL/NCCL kernel time includes waiting for peers.** It is not transfer time.
- **Async collectives:** their `hlo_op` (e.g. `all-gather.34`) isn't in the HLO stored
  in the capture. `attribute_kernels.py` keeps the label and takes the Python line
  from the matching collective-start instruction, by order.
- **ROCm memcpy rows** may show `kind:Unknown size:0`.
- **The Python tracer** inflates host time.
- **One program can appear twice** (two `.hlo_proto.pb` files with different IDs).
  `attribute_kernels.py` uses the program ID with the most GPU time, from the
  xplane; the json has no `program_id`.

## Report

- **Markdown:** findings first.
- **Numbers:** each comes with its range across steps and GPUs, labelled
  *measured*, *read* or *inferred*.
- **Sections:** verdict and cause, reconciliation, per-GPU/stream table, top ops
  and Python lines, what to change (estimates), and the trust list.
- **Fact check:** before sending, have a fresh subagent check every claim against
  the tables.
- **HTML page:** if the user wants one, the writing-trace-report-pages skill
  builds it from this report and its tables.
