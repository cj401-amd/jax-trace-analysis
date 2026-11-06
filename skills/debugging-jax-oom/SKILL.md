---
name: debugging-jax-oom
description: Use when a JAX / XLA program ran out of GPU memory (RESOURCE_EXHAUSTED, "Out of memory while trying to allocate", OOM on the first step or only after N steps) or uses more GPU memory than expected, and a jax.profiler / XProf capture (.xplane.pb) of the run is available.
---

# Debugging JAX OOM from a capture

## Overview

A jitted JAX step gets all of its scratch memory as one planned temp block. So the
step's memory is XLA's plan, and the `.xplane.pb` holds both that plan and the
allocator's runtime events. `scripts/oom_report.py` reads both in 2–5 seconds, using
only the standard library:

- **plan:** each module's buffer assignment, from `/host:metadata`. It gives the
  parameters (and how much is donated), the non-aliased outputs, the temp block, and the
  heap peak. For the buffers live at that peak it gives size, shape, role and Python line.
  The role comes from the JAX name stack: `jvp(` is forward (kept for backward),
  `transpose(` is backward, `s8` custom-call outputs are library workspace.
- **runtime:** the BFC allocator's events. They give the capacity, bytes in use at the
  start of each execution, the peak, `AllocatorRetry` waits, executions that never
  launched a kernel, and the host events between the last two executions. They also
  give what was live when the last execution started, grouped by the host event that
  allocated it: `outputs of jit_step execution 3`, `BufferFromHostBuffer(s32[…])`
  (a `device_put`), or "allocated before the capture started".

The report ends with diagnosis lines, each tagged measured or inferred. Run it first:

```bash
python3 -P <this skill's base directory>/scripts/oom_report.py <capture dir or .xplane.pb> [--module 'jit_train_step(123)'] [--top 25] [--json out.json]
python3 -P <this skill's base directory>/scripts/oom_report.py --self-test
```

By default it breaks down the module whose last execution never launched. If there is
none, it takes the biggest module that ran. Use `--module` to pick another, for example
an eval step.

## From diagnosis line to cause and fix

| Diagnosis line | Cause | Fix to propose; the saving comes from the report's bytes |
|---|---|---|
| `the step itself doesn't fit` | the plan alone exceeds the pool | Follow the role split at the peak (below) |
| ↳ mostly `forward, kept for the backward pass` | activations: they grow with batch × seq; naive attention adds B·H·S·S per layer | `jax.checkpoint` / remat on the block (saves most of the forward-kept bytes, costs about one extra forward pass); a smaller micro-batch with gradient accumulation; a shorter sequence (S² buffers shrink by 4× when S halves). Offer fused attention only as "check that the B·H·S·S buffers leave the plan" |
| ↳ mostly `backward` | gradients and their temporaries | Shard (FSDP) or accumulate; check for f32 copies of bf16 gradients |
| ↳ `workspace` | GEMM / conv library scratch | Usually small; on ROCm it is 64 MiB per hipBLASLt GEMM on gfx950 |
| `grow by X MiB per execution`, `= 1.00x the non-aliased outputs` | the program keeps each step's outputs alive (a leak across steps): the live list shows `outputs of … execution 1, 2, …` | Find the holder in the host events between executions (e.g. `$<unknown> append` = `list.append`). Then drop the reference, reduce inside the step (a norm, not the tree), or `jax.device_get` it. A bigger pool only delays the OOM |
| `no parameter is donated … can save up to X` | old and new params / optimizer state are live together | `donate_argnums` on the state arguments |
| `other live arrays take the room; the largest … came from X` | arrays outside the step: a dataset or eval copy put on the GPU, prefetch, other jit outputs | Keep it on the host and stream it, free it, or shrink its dtype. "Allocated before the capture started" means profile earlier to name it |
| `AllocatorRetry … longest N s`, `launched none` | the OOM happened inside the capture | Evidence only. The failed request is usually that module's temp block, or one output |
| `XLA already rematerialized N instructions` | XLA's own remat already ran against the limit | Expect the fix to come from the program, not from XLA |

Give each fix its estimated saving and its cost, labelled inferred. Check the estimate
against the capacity. "Raise `XLA_PYTHON_CLIENT_MEM_FRACTION`" fixes a plan that
doesn't fit only if the GPU has the room. Capacity well below HBM suggests the fraction
is set; the capture doesn't record environment variables.

## The playbook's memory anatomy in JAX terms

The Ultra-Scale Playbook ("Profiling the memory usage") splits training memory into
weights, gradients, optimizer state and activations. In the report:

- **Weights + optimizer state** = the step's parameters. Adam in fp32 is 12 B per
  parameter (p, m, v). Donated means no second copy.
- **Gradients** = the `backward` role at the peak, plus outputs if the step returns them.
- **Activations** = `forward, kept for the backward pass`. Scan / `nn.scan` models keep
  them as stacked `[layers, …]` buffers on a `while` loop.
- **"Step 1 fits, step N fails":** in PyTorch this is optimizer state created at the
  first step. In JAX the state exists before step 1, so look for the growth line instead.
  The first call also compiles and autotunes, so a peak before the first execution is
  compile-time.

## Rules

- **Use only the capture.** Don't rebuild the model or rerun the workload to test a
  fix: estimate it from the plan's bytes, and say how to confirm it
  (`jit(...).lower(...).compile().memory_analysis()` on the user's side).
- **Don't write into the capture directory.** XProf's converters
  (`xspace_to_tool_names`, `memory_viewer`, `xprof` CLI) write `*.hlo_proto.pb` cache
  files next to the `.xplane.pb`. If you need XProf views, copy the `.xplane.pb` to a
  scratch directory first.
- **Label every number** measured, read or inferred.

## Measured reference (single MI355X, small pools, 2026-11-06)

| Capture | What the report shows |
|---|---|
| attention plan too big | need 3.72 GiB > 2.79 GiB; heap peak 81.6% forward-kept; `f32[8,8,1024,1024]` ×5 at the softmax line; temp block 3,495 MiB = the error's "3.41GiB"; AllocatorRetry 10 s; 89 remat instructions |
| grads kept every step | in use at step start 316 → 1,252 MiB, +104.0 MiB per step = 1.00× the non-aliased outputs; plan 788 MiB fits; `$<unknown> append` between steps |
| no donation | need 3.88 GiB > 2.79 GiB; 1,902 MiB params + 1,902 MiB outputs, 0 donated; temp only 172 MiB |
| MaxText llama2-7b, 8 GPUs, FSDP (fits) | temp 61.9 GiB; heap peak 57% forward-kept, the scan stacks `bf16[32,4,4096,11008]` ×2 via `while.50` |

## Limits

- **One host per file:** multi-host runs have one `.xplane.pb` per host, so run each.
- **HBM only:** only memory space 0 is broken down. Host offload shows as `memory space 5`.
- **Contiguity:** "fits" compares bytes. BFC also needs a contiguous chunk, so a
  near-fit can still fail through fragmentation.
- **"other" role:** buffers defined by XLA-created ops with no JAX name stack, and with
  no alias that has one.
- **CUDA:** not tried. It uses the same BFC and XLA events, but the names are unverified.
- **TPU / eager:** TPU captures give plan totals only, with no heap peak and no executions. Eager
  (op-by-op) programs show as many small modules, so read the runtime section there.
