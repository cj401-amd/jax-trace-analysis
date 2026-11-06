# jax-trace-analysis

A Claude Code plugin with skills for analyzing `jax.profiler.trace()` / XProf captures on GPUs.
Everything works from the capture's own files (`<host>.xplane.pb`, `<host>.trace.json.gz`):
nothing reruns the workload. Each skill bundles a script you can also run without Claude.

| Skill | Use it for | Script (runtime) |
|---|---|---|
| `analyzing-jax-launch-overhead` | Quick CPU- vs GPU-bound verdict per GPU; launch / queue time (SKIP paper, arXiv 2504.11750); idle time and the host event behind it | `skip_stats.py`: stdlib, a few seconds |
| `investigating-jax-traces` | Deep, trace-only investigation: reconciliation, swimlanes, every kernel → HLO → Python line, what to change, trust list | `attribute_kernels.py`: needs `jax` + `xprof`, a few seconds |
| `writing-trace-report-pages` | A shareable, self-contained HTML page of the investigation | `build_page.py`: stdlib, under 1 s |
| `comparing-jax-traces` | Before vs after: same work? step time beyond noise? verdict? tracer checks? | `compare_traces.py`: stdlib, a few seconds |
| `checking-rocm-tracer-defects` | How far to trust a ROCm trace: kernel_details collapse, memcpy rows, wrong device, graph labels, joins | `check_tracer.py`: stdlib, about 1 s |
| `debugging-jax-oom` | Why a run ran out of GPU memory (or uses more than expected): plan vs pool, what is live at the peak (forward / backward / workspace, Python line), growth between steps, donation | `oom_report.py`: stdlib, 2–5 s; reads the `.xplane.pb` |

## Install

```bash
claude plugin marketplace add <this repo: local path or git URL>
claude plugin install jax-trace-analysis@jax-trace-analysis
```

Then start a new Claude Code session. The skills load on their own when a request
fits them. To force one, start the request with `/jax-trace-analysis:<skill>`, e.g.
`/jax-trace-analysis:investigating-jax-traces`.

## Try it

A capture is the directory `jax.profiler.trace(logdir)` writes:
`<logdir>/plugins/profile/<run>/` with `<host>.xplane.pb` and `<host>.trace.json.gz`.

| Ask Claude | Skill that should fire |
|---|---|
| "Is this CPU-bound or GPU-bound, and what is the GPU waiting on? `<capture>/<host>.trace.json.gz`" | analyzing-jax-launch-overhead |
| "What would a top professional do with this trace to gain deep insights into the performance, the operators on each stream of each GPU, and tie it back to the Python code? `<capture>`" | investigating-jax-traces |
| "Write an HTML report page for this capture: `<capture>`" | writing-trace-report-pages |
| "Did my change do anything except the labels? A: `<capture before>`, B: `<capture after>`" | comparing-jax-traces |
| "How far can I trust this trace? `<capture>`" | checking-rocm-tracer-defects |
| "My JAX training job died with an out-of-memory error. Why? `<capture>/<host>.xplane.pb`" | debugging-jax-oom |

The scripts on their own, from the repo root:

```bash
C=<logdir>/plugins/profile/<run>    # one capture
python3 -P skills/analyzing-jax-launch-overhead/scripts/skip_stats.py $C/*.trace.json.gz --out skip_out
python3 -P skills/investigating-jax-traces/scripts/attribute_kernels.py $C --out attribution
python3 -P skills/checking-rocm-tracer-defects/scripts/check_tracer.py $C/*.trace.json.gz
python3 -P skills/comparing-jax-traces/scripts/compare_traces.py <before>/*.trace.json.gz <after>/*.trace.json.gz
python3 -P skills/writing-trace-report-pages/scripts/build_page.py --keys   # narrative fields
python3 -P skills/debugging-jax-oom/scripts/oom_report.py $C
```

Every script has `--self-test`.

## Requirements

- **Python:** 3.11 or newer (the commands use `python3 -P`).
- **Packages:** `attribute_kernels.py` needs `jax` and `xprof` (it reads the HLO
  stored in the `.xplane.pb`). All other scripts use only the standard library
  (`oom_report.py` decodes the `.xplane.pb` itself).
- **Platform:** tested on ROCm captures (MI355X, TheRock ROCm 10, rocprofiler-sdk
  1.3.5). CUDA captures should work, since they use the same event and argument
  names, but haven't been tried.

## Tests

Each script's `--self-test` checks it on synthetic input built inside the script, so
it needs no captures. The agent tests (with and without each skill, on MI355X
captures) and the `claude plugin eval` cases depend on local captures, so they are
kept out of this repo.

## Layout

```
.claude-plugin/{plugin.json, marketplace.json}
skills/<skill>/{SKILL.md, scripts/}
PLAN.md          build plan and status
```
