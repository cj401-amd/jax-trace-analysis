#!/usr/bin/env python3
"""Build one self-contained HTML page from a trace investigation.

usage: build_page.py --skip SKIP_OUT_DIR --attr ATTRIBUTION_DIR --narrative NARRATIVE.json --out PAGE.html
       build_page.py --self-test

Figures and tables come from the CSVs written by skip_stats.py (--skip) and attribute_kernels.py
(--attr); the prose comes from NARRATIVE.json (keys in NARRATIVE_KEYS, written by the agent). Times in
skip_stats tables are microseconds. Plain HTML: no external files, light and dark modes, every
mark has a <title> tooltip, and every chart has a table view.
"""
import argparse
import ast
import collections
import csv
import html
import json
import os
import re
import statistics
from html.parser import HTMLParser

NARRATIVE_KEYS = {
    "title": "page title",
    "capture": "capture path or name",
    "core_idea": "1. the core idea: verdict and main cause, 2-4 sentences",
    "players": "2. what runs where: programs, GPUs, streams, host threads",
    "bottleneck": "6. the bottleneck, with the host event behind idle time",
    "roofline": "11. list of {op, time_ms, achieved, peak, peak_source, note}, or [] if not computed",
    "evidence": "12. list of {check, result, label} with label measured/read/inferred",
    "untrusted": "13. list of {item, size}",
    "changes": "14. list of {change, estimate, assumptions}",
    "remember": "15. exactly three short sentences",
}
CATS = ["compute", "collective", "memcpy", "host_idle", "launch", "wait"]
LABEL = {"compute": "compute", "collective": "collective", "memcpy": "memcpy", "host_idle": "host-idle",
         "launch": "launch", "wait": "wait"}
CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f3f2ef;--text-primary:#0b0b0b;--text-secondary:#52514e;
--rule:#d9d8d4;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;
--surface-2:#252523;--text-primary:#fff;--text-secondary:#c3c2b7;--rule:#3a3a37;--s1:#3987e5;--s2:#d95926;--s3:#199e70;
--s4:#c98500;--s5:#d55181;--s6:#008300}}
body{margin:0;background:var(--surface-1);color:var(--text-primary);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px}h1{font-size:1.7em;margin:.4em 0}h2{margin-top:2em;border-top:1px solid var(--rule);padding-top:.8em}
.muted{color:var(--text-secondary)}code{font-size:.9em;background:var(--surface-2);padding:0 .25em;border-radius:3px}
table{border-collapse:collapse;font-size:.88em;margin:.6em 0}td,th{border-bottom:1px solid var(--rule);padding:3px 8px;text-align:left;vertical-align:top}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}figure{margin:1em 0}figcaption{color:var(--text-secondary);font-size:.88em}
svg text{fill:var(--text-secondary);font:11px system-ui,sans-serif}.legend span{display:inline-block;margin-right:14px;font-size:.85em}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.tbl{overflow-x:auto}details{margin:.4em 0 1em}summary{cursor:pointer;color:var(--text-secondary);font-size:.88em}
.label{font-size:.75em;border:1px solid var(--rule);border-radius:3px;padding:0 4px;color:var(--text-secondary)}
"""
COLOR = {c: f"var(--s{i + 1})" for i, c in enumerate(CATS)}
E = html.escape


def md(text):
    """Tiny Markdown: paragraphs, '- ' bullets, `code`, **bold**."""
    out, items = [], []

    def inline(s):
        s = E(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)

    for block in re.split(r"\n\s*\n", (text or "").strip()):
        lines = block.splitlines()
        if lines and all(l.lstrip().startswith("- ") for l in lines):
            out.append("<ul>" + "".join(f"<li>{inline(l.lstrip()[2:])}</li>" for l in lines) + "</ul>")
        elif block.strip():
            out.append(f"<p>{inline(' '.join(lines))}</p>")
    return "\n".join(out)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def table(head, rows, num=()):
    th = "".join(f'<th class="n">{E(h)}</th>' if i in num else f"<th>{E(h)}</th>" for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td class="n">{E(str(c))}</td>' if i in num else f"<td>{E(str(c))}</td>"
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def legend(cats):
    return '<div class="legend">' + "".join(f'<span><i style="background:{COLOR[c]}"></i>{LABEL[c]}</span>' for c in cats) + "</div>"


def short_gpu(g):
    return g.replace("/device:", "")


def budget_figure(streams):
    """Stacked bars, one per GPU stream: shares of the stream's span."""
    W, L, H = 900, 150, 18
    rows = [s for s in streams if float(s["span"]) > 0 and float(s["busy"]) / float(s["span"]) >= 0.01]
    svg = [f'<svg viewBox="0 0 {W} {len(rows) * (H + 6) + 24}" width="100%" role="img" aria-label="time budget per GPU stream">']
    trs = []
    for i, s in enumerate(rows):
        y, x, span = i * (H + 6), L, float(s["span"])
        svg.append(f'<text x="{L - 6}" y="{y + 13}" text-anchor="end">{E(short_gpu(s["gpu"]) + " " + s["stream"])}</text>')
        shares = {c: max(0.0, float(s[c])) / span for c in CATS}
        for c in CATS:
            w = shares[c] * (W - L - 10)
            if w > 0.5:
                svg.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w - 2, 0.5):.1f}" height="{H}" rx="2" fill="{COLOR[c]}">'
                           f'<title>{E(short_gpu(s["gpu"]))} {E(s["stream"])}: {LABEL[c]} {100 * shares[c]:.1f}%</title></rect>')
                if w > 34:
                    ink = "#0b0b0b" if c in ("memcpy", "host_idle", "launch") else "#fff"
                    svg.append(f'<text x="{x + 4:.1f}" y="{y + 13}" style="fill:{ink}">{100 * shares[c]:.0f}%</text>')
            x += w
        trs.append([short_gpu(s["gpu"]), s["stream"], f"{span / 1e3:.1f}"] + [f"{100 * shares[c]:.1f}" for c in CATS])
    svg.append(f'<text x="{L}" y="{len(rows) * (H + 6) + 16}">0%</text><text x="{W - 10}" y="{len(rows) * (H + 6) + 16}" text-anchor="end">100% of the stream\'s span</text></svg>')
    tab = table(["GPU", "stream", "span ms"] + [LABEL[c] + " %" for c in CATS], trs, num=tuple(range(2, 9)))
    return legend(CATS) + "\n".join(svg), tab


def swimlane_figure(rows, steps):
    """The median step (by IL) on every GPU x stream; kernels closer than a pixel merge into one mark."""
    il = collections.defaultdict(list)
    for s in steps:
        il[int(s["step"])].append(float(s["il"]))
    med = statistics.median(statistics.median(v) for v in il.values())
    step = min(sorted(il), key=lambda k: abs(statistics.median(il[k]) - med))  # ties: the earlier step
    evs = [r for r in rows if r["step"] and int(r["step"]) == step]
    t0 = min(float(r["start"]) for r in evs)
    t1 = max(float(r["start"]) + float(r["dur"]) for r in evs)
    W, L, H = 900, 150, 14
    px = (t1 - t0) / (W - L - 10)
    num = lambda s: int(re.sub(r"\D", "", s) or 0)
    lanes = sorted({(r["gpu"], r["stream"]) for r in evs}, key=lambda k: (num(k[0]), num(k[1])))
    lane_busy = collections.Counter()
    for r in evs:
        lane_busy[(r["gpu"], r["stream"])] += float(r["dur"])
    drawn = [l for l in lanes if lane_busy[l] >= 0.005 * (t1 - t0)]  # near-empty lanes: table only
    svg = [f'<svg viewBox="0 0 {W} {len(drawn) * (H + 4) + 26}" width="100%" role="img" aria-label="one step, every GPU stream">']
    trs = []
    for lane in lanes:
        i = drawn.index(lane) if lane in drawn else None
        y = (i or 0) * (H + 4)
        if i is not None:
            svg.append(f'<text x="{L - 6}" y="{y + 11}" text-anchor="end">{E(short_gpu(lane[0]) + " " + lane[1])}</text>')
        busy = collections.Counter()
        for cat in ("compute", "collective", "memcpy"):
            iv = sorted((float(r["start"]), float(r["start"]) + float(r["dur"]), r["name"]) for r in evs
                        if (r["gpu"], r["stream"]) == lane and r["cat"] == cat)
            merged = []
            for a, b, n in iv:
                if merged and a - merged[-1][1] < px:
                    merged[-1][1] = max(merged[-1][1], b)
                    merged[-1][2] += 1
                else:
                    merged.append([a, b, 1, n])
            for a, b, n, first in merged:
                busy[cat] += b - a
                if i is None:
                    continue
                x = L + (a - t0) / px
                what = first[:60] if n == 1 else f"{n} kernels"
                svg.append(f'<rect x="{x:.1f}" y="{y}" width="{max((b - a) / px, 0.8):.1f}" height="{H}" fill="{COLOR[cat]}">'
                           f'<title>{E(short_gpu(lane[0]))} {E(lane[1])}: {cat}, {E(what)}, '
                           f'{(a - t0) / 1e3:.2f}-{(b - t0) / 1e3:.2f} ms</title></rect>')
        trs.append([short_gpu(lane[0]), lane[1]] + [f"{busy[c] / 1e3:.2f}" for c in ("compute", "collective", "memcpy")])
    yb = len(drawn) * (H + 4) + 14
    for k in range(6):
        x = L + k * (W - L - 10) / 5
        anchor = "start" if k == 0 else "end" if k == 5 else "middle"
        svg.append(f'<text x="{x:.0f}" y="{yb}" text-anchor="{anchor}">{k * (t1 - t0) / 5e3:.0f} ms</text>')
    svg.append("</svg>")
    tab = table(["GPU", "stream", "compute ms", "collective ms", "memcpy ms"], trs, num=(2, 3, 4))
    cap = (f"Step {step}, the median step by latency: {(t1 - t0) / 1e3:.1f} ms from its first kernel to its last, "
           f"on {len({l[0] for l in lanes})} GPUs. Gaps are idle time. Lanes busy under 0.5% of the step are in the table only.")
    return step, legend(["compute", "collective", "memcpy"]) + "\n".join(svg), tab, cap


def bars_figure(items, unit="% of GPU time"):
    """Horizontal bars, one series: label, share."""
    W, L, H = 900, 330, 16
    top = max((v for _, v in items), default=1) or 1
    svg = [f'<svg viewBox="0 0 {W} {len(items) * (H + 6)}" width="100%" role="img" aria-label="{E(unit)}">']
    for i, (k, v) in enumerate(items):
        y = i * (H + 6)
        w = v / top * (W - L - 60)
        svg.append(f'<text x="{L - 6}" y="{y + 12}" text-anchor="end">{E((lambda b: b if len(b) <= 45 else b[:44] + "…")(os.path.basename(k)))}</text>'
                   f'<rect x="{L}" y="{y}" width="{max(w, 1):.1f}" height="{H}" rx="2" fill="var(--s1)"><title>{E(k)}: {v:.2f}{E(" " + unit)}</title></rect>'
                   f'<text x="{L + w + 6:.1f}" y="{y + 12}">{v:.1f}%</text>')
    return "\n".join(svg) + "</svg>"


def build(skip, attr, nar):
    missing = [k for k in NARRATIVE_KEYS if k not in nar]
    assert not missing, f"narrative is missing {missing}"
    assert len(nar["remember"]) == 3, "remember needs exactly three sentences"
    streams, steps, rows = read(os.path.join(skip, "streams.csv")), read(os.path.join(skip, "steps.csv")), read(os.path.join(skip, "rows.csv"))
    lines, ops, how = (read(os.path.join(attr, f)) for f in ("lines.csv", "ops.csv", "how.csv"))
    first = open(os.path.join(skip, "summary.txt")).read().splitlines()[1]
    main = {}
    for s in streams:
        if s["gpu"] not in main or float(s["compute"]) > float(main[s["gpu"]]["compute"]):
            main[s["gpu"]] = s
    step, lane_svg, lane_tab, lane_cap = swimlane_figure(rows, steps)
    bud_svg, bud_tab = budget_figure(streams)
    causes = []
    for g, s in sorted(main.items(), key=lambda x: int(x[0].rsplit(":", 1)[1])):
        c = ast.literal_eval(s["causes"]) if s["causes"] else {}
        tot = float(s["host_idle"]) or 1
        top = sorted(c.items(), key=lambda x: -x[1])[:2]
        causes.append([short_gpu(g), s["stream"], f"{100 * float(s['host_idle']) / float(s['span']):.1f}",
                       "; ".join(f"{n} {100 * v / tot:.0f}%" for n, v in top)])
    med_il = statistics.median(float(s["il"]) for s in steps) / 1e3
    pic = (f'<svg viewBox="0 0 900 120" width="100%" role="img" aria-label="how a step flows from Python to the GPUs">'
           + "".join(f'<rect x="{x}" y="30" width="190" height="56" rx="6" fill="none" stroke="var(--rule)" stroke-width="2"/>'
                     f'<text x="{x + 95}" y="54" text-anchor="middle" style="fill:var(--text-primary);font-size:13px">{a}</text>'
                     f'<text x="{x + 95}" y="72" text-anchor="middle">{b}</text>'
                     for x, a, b in ((10, "Python step loop", "dispatches jitted steps"),
                                     (240, "XLA runtime (host)", "prepares, launches"),
                                     (470, "launch calls", "kernels, graphs, copies"),
                                     (700, "GPU streams", f"{len(main)} GPUs, median IL {med_il:.0f} ms")))
           + "".join(f'<path d="M{x} 58 h30" stroke="var(--text-secondary)" stroke-width="2" marker-end="url(#a)"/>' for x in (202, 432, 662))
           + '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
             '<path d="M0 0L10 5L0 10z" fill="var(--text-secondary)"/></marker></defs></svg>')
    budget_main = table(["GPU", "main stream", "compute %", "host-idle %", "wait %"],
                        [[short_gpu(g), s["stream"]] + [f"{100 * float(s[c]) / float(s['span']):.1f}" for c in ("compute", "host_idle", "wait")]
                         for g, s in sorted(main.items(), key=lambda x: int(x[0].rsplit(":", 1)[1]))], num=(2, 3, 4))
    top_lines = [(r["key"], float(r["share"])) for r in lines[:12]]
    roof = nar["roofline"]
    sec = []
    add = sec.append
    add(f"<h2>1. The core idea</h2>{md(nar['core_idea'])}")
    add(f"<h2>2. The players</h2>{md(nar['players'])}")
    add(f'<h2>3. A picture to hold on to</h2><figure>{pic}<figcaption>Where a step goes. The numbers come from the '
        f"trace; the boxes are XLA's usual path, not measured. IL is a step's latency: from its step marker to its last "
        f"GPU event, so it differs from the interval between steps.</figcaption></figure>{budget_main}")
    add(f'<h2>4. One step, frame by frame</h2><figure>{lane_svg}<figcaption>{E(lane_cap)} Hover a mark for its kernels '
        f"and times.</figcaption></figure><details><summary>Table: busy time per lane in step {step}</summary>{lane_tab}</details>")
    add(f'<h2>5. Per GPU, per stream</h2><figure>{bud_svg}<figcaption>Each stream\'s span (first event to last over the '
        f"whole capture), split by what the stream was doing. Streams under 1% busy are left out of the chart, not the table.</figcaption></figure>"
        f"<details><summary>Table: every stream</summary>{bud_tab}</details>")
    add(f"<h2>6. The bottleneck</h2>{md(nar['bottleneck'])}"
        + table(["GPU", "main stream", "host-idle %", "host event during host-idle (share of it)"], causes, num=(2,)))
    add(f'<h2>10. From kernels back to Python</h2><figure>{bars_figure(top_lines)}<figcaption>Share of GPU kernel time per '
        f'Python line (the innermost frame of the user\'s code), all GPUs. Matching: '
        + E(", ".join(f"{r['key']} {float(r['share']):.1f}%" for r in how)) + "</figcaption></figure>"
        f"<details><summary>Table: top operations</summary>"
        + table(["op_name | HLO instruction", "kernels", "GPU ms", "share %"],
                [[r["key"][:120], r["kernels"], r["gpu_ms"], r["share"]] for r in ops[:20]], num=(1, 2, 3)) + "</details>")
    add("<h2>11. The real compute, against the roofline</h2>" + (
        table(["op", "time ms", "achieved", "peak", "peak source", "note"],
              [[r.get(k, "") for k in ("op", "time_ms", "achieved", "peak", "peak_source", "note")] for r in roof], num=(1,))
        if roof else '<p class="muted">Not computed for this capture.</p>'))
    add("<h2>12. Evidence, and how strong it is</h2>"
        + table(["check", "result", "label"], [[r["check"], r["result"], r["label"]] for r in nar["evidence"]])
        + f'<p class="muted">From the trace: {E(first)}</p>')
    add("<h2>13. Where this trace can't be trusted</h2>" + table(["item", "size"], [[r["item"], r["size"]] for r in nar["untrusted"]]))
    add("<h2>14. What to change</h2>" + table(["change", "estimated effect (inferred, not run)", "assumptions"],
                                               [[r["change"], r["estimate"], r["assumptions"]] for r in nar["changes"]]))
    add("<h2>15. If you remember 3 things</h2><ol>" + "".join(f"<li>{E(s)}</li>" for s in nar["remember"]) + "</ol>")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{E(nar['title'])}</title><style>{CSS}</style></head><body class=\"viz-root\"><main>"
            f"<h1>{E(nar['title'])}</h1><p class=\"muted\">Capture: <code>{E(nar['capture'])}</code>. Built from its trace "
            f"files only, with skip_stats.py and attribute_kernels.py.</p>" + "\n".join(sec) + "</main></body></html>")


class _Check(HTMLParser):
    VOID = {"meta", "br", "img", "input", "link", "hr", "path", "rect", "i", "circle", "line"}

    def __init__(self):
        super().__init__()
        self.stack = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        assert self.stack and self.stack[-1] == tag, f"unbalanced </{tag}>, open {self.stack[-3:]}"
        self.stack.pop()


def check_html(page):
    c = _Check()
    c.feed(page)
    assert not c.stack, f"unclosed {c.stack}"


def self_test():
    import tempfile
    d = tempfile.mkdtemp(prefix="build_page_test_", dir=".")
    skip, attr = os.path.join(d, "skip"), os.path.join(d, "attr")
    os.makedirs(skip)
    os.makedirs(attr)

    def w(path, head, rows):
        with open(path, "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(head)
            cw.writerows(rows)
    # 1 GPU, 2 streams, 2 steps; times in us
    w(os.path.join(skip, "streams.csv"), ["gpu", "stream", "events", "span", "busy", "compute", "collective", "memcpy", "host_idle",
                                           "launch", "wait", "other", "causes", "cause_line"],
      [["/device:GPU:0", "Stream #0", 3, 1000, 600, 550, 0, 50, 300, 20, 80, 0, "{'command_buffer::update': 270, 'x': 30}", "t"],
       ["/device:GPU:0", "Stream #1", 1, 900, 200, 0, 200, 0, 0, 0, 700, 0, "", ""]])
    w(os.path.join(skip, "steps.csv"), ["gpu", "step", "kernels", "il", "kernel_time", "idle", "busy", "idle_merged", "tklqt", "akd"],
      [["/device:GPU:0", 1, 2, 400, 300, 100, 300, 100, 10, 150], ["/device:GPU:0", 2, 2, 500, 300, 200, 300, 200, 10, 150]])
    w(os.path.join(skip, "rows.csv"), ["gpu", "stream", "step", "name", "cat", "start", "dur"],
      [["/device:GPU:0", "Stream #0", 1, "fusion_1.kd", "compute", 0, 100], ["/device:GPU:0", "Stream #0", 1, "fusion_2.kd", "compute", 100, 200],
       ["/device:GPU:0", "Stream #1", 2, "ncclDevKernel", "collective", 600, 200], ["/device:GPU:0", "Stream #0", 2, "copy", "memcpy", 900, 50],
       ["/device:GPU:0", "Stream #10", 1, "tiny.kd", "compute", 50, 1]])
    open(os.path.join(skip, "summary.txt"), "w").write("trace: x\nGPU events 4: linked to a host launch call 4\n")
    w(os.path.join(attr, "lines.csv"), ["key", "kernels", "gpu_ms", "share"], [["/x/site-packages/m/train.py:12 step", 2, 0.3, 75.0],
                                                                              ["(no user frame)", 1, 0.1, 25.0]])
    w(os.path.join(attr, "ops.csv"), ["key", "kernels", "gpu_ms", "share"], [["jit(f)/add | fusion.1", 2, 0.3, 75.0]])
    w(os.path.join(attr, "how.csv"), ["key", "kernels", "gpu_ms", "share"], [["exact", 3, 0.3, 75.0], ["name", 1, 0.1, 25.0]])
    nar = {k: "Text with `code` and **bold**.\n\n- one\n- two" for k in NARRATIVE_KEYS}
    nar.update(title="T <&>", capture="/cap", roofline=[], evidence=[{"check": "join", "result": "4/4", "label": "measured"}],
               untrusted=[{"item": "RCCL waits", "size": "20%"}], changes=[{"change": "donate", "estimate": "-30%", "assumptions": "a"}],
               remember=["a", "b", "c"])
    page = build(skip, attr, nar)
    check_html(page)
    for want in ("T &lt;&amp;&gt;", "<code>code</code>", "<strong>bold</strong>", "<li>one</li>",
                 "compute 55.0%", "host-idle 30.0%", "command_buffer::update 90%",  # from streams.csv
                 "m/train.py:12 step", "exact 75.0%", "Not computed", "Step 1, the median step",
                 "GPU events 4: linked", "4/4", "RCCL waits", "-30%", "<li>c</li>"):
        assert want in page, want
    assert "Stream #1</text>" in page and page.count("<h2>") == 12
    lane = page.split("<h2>4.")[1].split("<h2>5.")[0]
    assert "GPU:0 Stream #10</text>" not in lane and "<td>Stream #10</td>" in lane, "tiny lane"
    assert "train.py:12 step</text>" in page, "short line label"
    try:
        build(skip, attr, {k: v for k, v in nar.items() if k != "changes"})
        raise SystemExit("missing narrative key not caught")
    except AssertionError as e:
        assert "changes" in str(e)
    import shutil
    shutil.rmtree(d)
    print("self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", help="skip_stats.py output directory")
    ap.add_argument("--attr", help="attribute_kernels.py output directory")
    ap.add_argument("--narrative", help="JSON with the keys listed by --keys")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--keys", action="store_true", help="print the narrative keys and exit")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.keys:
        return print(json.dumps(NARRATIVE_KEYS, indent=1))
    if not (a.skip and a.attr and a.narrative):
        ap.error("--skip, --attr and --narrative are required")
    page = build(a.skip, a.attr, json.load(open(a.narrative)))
    check_html(page)
    open(a.out, "w").write(page)
    print(f"wrote {a.out} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
