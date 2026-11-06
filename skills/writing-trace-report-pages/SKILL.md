---
name: writing-trace-report-pages
description: Use when the findings of a jax.profiler.trace() / XProf trace analysis should become a standalone HTML page or report to share - a write-up with a per-step swimlane of every GPU stream, per-stream time budgets, the bottleneck, kernels tied back to Python lines, and what to change.
---

# Writing trace report pages

## Overview

One self-contained HTML page per capture. A script draws the figures and tables from
the investigation's CSVs, so every number on the page comes from a table; you write the
prose. The page has sections 1–6 and 10–15:

| # | Section | Comes from |
|---|---|---|
| 1 | The core idea | you: the verdict and main cause, 2–4 sentences |
| 2 | The players | you: programs, GPUs, streams, host threads |
| 3 | A picture to hold on to | script: step-flow diagram + main-stream split per GPU |
| 4 | One step, frame by frame | script: swimlane of the median step, every GPU × stream |
| 5 | Per GPU, per stream | script: time-budget bars + full table |
| 6 | The bottleneck | you, plus the script's host-idle cause table |
| 10 | From kernels back to Python | script: GPU time per Python line + top ops |
| 11 | The real compute, against the roofline | you: rows you computed, or "not computed" |
| 12 | Evidence, and how strong it is | you: the reconciliation checks, labelled |
| 13 | Where this trace can't be trusted | you: items with sizes |
| 14 | What to change | you: estimates, labelled *inferred, not run* |
| 15 | If you remember 3 things | you: three sentences |

Like the investigation, everything comes from the trace files: no reruns.

## Steps

1. **Get the tables.** If the user hasn't run the investigating-jax-traces skill on
   this capture, run its two scripts first, from one scratch directory:

   ```bash
   B=<this skill's base directory>
   python3 -P $B/../analyzing-jax-launch-overhead/scripts/skip_stats.py <capture>/<host>.trace.json.gz --out skip_out
   python3 -P $B/../investigating-jax-traces/scripts/attribute_kernels.py <capture dir> --out attribution
   ```

2. **Write `narrative.json`.** `python3 -P $B/scripts/build_page.py --keys` lists the
   keys. Write the prose from the tables and the investigation's findings:
   - Text fields take a tiny Markdown subset: paragraphs, `- ` bullets,
     `` `code` `` and `**bold**`.
   - Label numbers *measured*, *read* or *inferred*.
3. **Build the page:**

   ```bash
   python3 -P $B/scripts/build_page.py --skip skip_out --attr attribution --narrative narrative.json --out <name>.html
   ```

   It refuses a narrative with a missing key, or `remember` without exactly three
   items, and checks that the HTML is well formed.
4. **Look at it.** Open it or screenshot it (headless Chromium if available) and check
   the figures for collisions, clipped labels and empty lanes before sending it.
5. **Fact check:** have a fresh subagent compare every number in your prose with
   the CSVs.

## What the script guarantees

- **Single file:** no network requests, light and dark modes (the system setting).
- **Charts:** a `<title>` tooltip on every mark, and a table view under every chart.
- **Palette:** the validated categorical palette, in a fixed order: compute,
  collective, memcpy, host-idle, launch, wait.
- **Swimlane:** the median step by latency. Kernels closer together than a pixel
  merge into one mark. Lanes busy under 0.5% of the step appear in the table only.

## Common mistakes

- **Wrong idle number:** quoting the paper's idle (IL − Σt_k) in the prose. Use
  `idle_merged` from `steps.csv`, or the stream budget.
- **Overclaiming the roofline:** filling in §11 without a cited peak for the op's
  data type. If you didn't compute it, leave the list empty: the page then says
  "Not computed".
- **New numbers in the prose:** writing numbers that aren't in the tables. Every
  number in the prose must be in a CSV or labelled *inferred*.
