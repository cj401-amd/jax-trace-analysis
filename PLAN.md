# jax-trace-analysis plugin: build plan (2026-11-06)

Decisions (from the user):

- **v1 skills:** analyzing-jax-launch-overhead (existing), investigating-jax-traces,
  writing-trace-report-pages, comparing-jax-traces, and the tracer-defect checks.
- **Hosting:** local repo first. Pushing to a remote is the user's call.
- **Report page:** plain self-contained HTML, no house style. Sections 1–6 and
  10–15 of the MiniGPT page; 7–9 aren't needed.
- **Everything works from the given trace files:** no reruns.

Each skill: RED (no plugin) → write → GREEN, plus the script's own self-test. The
agent tests and `claude plugin eval` cases use local captures, so they are kept out
of this repo.

| # | Step | Check | Status |
|---|---|---|---|
| 1 | Repo scaffold; move in analyzing-jax-launch-overhead | local marketplace installs; skill listed; self-test passes | done: 0.2.0 installed as jax-trace-analysis@jax-trace-analysis, 5 skills listed; old ~/.claude/skills copy removed |
| 2 | investigating-jax-traces + attribute_kernels.py | script vs the MiniGPT pipeline; GREEN on llama2 | done: 147,840/147,840 same as the validated pipeline; llama2 loops 99.98%; GREEN passed 5/5 (RED had failed 3/5) |
| 3 | writing-trace-report-pages (plain HTML) | render check; numbers from tables; RED/GREEN | done: both pass; GREEN uses the fixed 12 sections, 22% cheaper; IL label fix |
| 4 | comparing-jax-traces | llama2 ctrl vs fix; clean test MiniGPT default vs donate | done: GREEN 5/5 on both; RED passed too at 2.6× the cost; RED's extra finding added as `kd_zero` |
| 5 | checking-rocm-tracer-defects | llama2 ctrl; clean test MiniGPT | done: GREEN 5/5 on both; tests added `dropped` (SPMD), `occupancy`, `kd_zero` and an XProf-views section |
| 6 | debugging-jax-oom + oom_report.py (2026-11-06; from the Ultra-Scale Playbook's "Profiling the memory usage") | four local OOM captures with known causes; RED/GREEN | done: RED passed A and B on content, but wrote into the capture dir and (A) built a model replica, at 1.7–2.5× the tokens; GREEN 5/5 on A–C and on the clean D (re-run after the refactor: 11 calls, 68k tokens) |

Open (not done):

- Commit and push to github.com/cj401-amd/jax-trace-analysis (public, MIT).
- `claude plugin eval` cases are written, but they need a sandbox (bubblewrap +
  socat) that this container doesn't have.
- No CUDA capture has been tried.

## Notes

- `claude plugin eval` needs a sandbox (bubblewrap + socat) to give the agent a
  shell. It isn't installed in this container, so the local eval cases are for
  machines that have it. Here, tests run as fresh subagents.
- The verdict skill's install path moved from `~/.claude/skills/…` to the skill's
  base directory. Installed plugins live in
  `~/.claude/plugins/cache/<marketplace>/<plugin>/<version>/`.
