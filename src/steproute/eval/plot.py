"""The frontier plot as a standalone SVG, written with the standard library only.

Score (success rate) against spend (mean cost per unit). Single configurations are neutral
circles with direct labels; the frontier is a dashed line; each policy has its own marker
shape, a direct label, its confidence intervals as whiskers, and a legend entry with its
verdict. Identity never rests on colour alone: shapes, labels and the legend carry it, so
the plot reads in greyscale and for colour-blind readers. Light and dark themes follow the
viewer's setting. Every mark has a ``<title>`` tooltip.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from xml.sax.saxutils import escape

from steproute.eval.frontier import Point, envelope
from steproute.eval.results import BEATS, DOES_NOT_BEAT, Evaluation

__all__ = ["render_svg", "write_svg"]

_W, _H = 820, 560
_L, _R, _T, _B = 76, 24, 64, 150  # plot margins; the legend sits under the axes

_STYLE = """
.viz{color-scheme:light;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;
--grid:#e6e5e0;--cfg:#6b6a65;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#52514e}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])) .viz{
color-scheme:dark;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;
--grid:#33332f;--cfg:#a6a59c;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c3c2b7}}
:root[data-theme="dark"] .viz{color-scheme:dark;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#8f8e86;--grid:#33332f;--cfg:#a6a59c;--s1:#3987e5;--s2:#d95926;
--s3:#199e70;--s4:#c3c2b7}
.viz text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;fill:var(--ink)}
.bg{fill:var(--surface)} .title{font-size:17px;font-weight:600}
.sub{font-size:12px;fill:var(--ink2)} .tick{font-size:11px;fill:var(--ink2)}
.axis-label{font-size:12px;fill:var(--ink2)} .grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--muted);stroke-width:1} .cfg{fill:var(--cfg);stroke:var(--surface);stroke-width:2}
.cfg-label{font-size:10.5px;fill:var(--ink2)} .frontier{fill:none;stroke:var(--ink2);
stroke-width:2;stroke-dasharray:6 4} .pol-label{font-size:12px;font-weight:600}
.legend{font-size:12px} .legend-note{font-size:11px;fill:var(--ink2)}
.whisker{stroke-width:1.5}
"""

_SHAPES = ("diamond", "triangle", "square", "star", "triangle-down", "cross")


def _nice_step(span: float, target: int = 6) -> float:
    if span <= 0:
        return 1.0
    raw = span / target
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * mag >= raw:
            return float(m * mag)
    return float(10 * mag)


def _fmt_usd(x: float) -> str:
    if x == 0:
        return "$0"
    if abs(x) >= 1:
        return f"${x:,.2f}"
    return f"${x:.3g}"


def _short(name: str, limit: int = 40) -> str:
    """Display label: a path in a spec is reduced to its file name, then the text is cut."""
    kind, sep, rest = name.partition(":")
    if sep and ("/" in rest or "\\" in rest):
        name = f"{kind}:{os.path.basename(rest)}"
    return name if len(name) <= limit else name[: limit - 1] + "…"


class _Placer:
    """Greedy label placement: the first of four positions around a mark that stays inside
    the plot and overlaps no label or mark placed before it."""

    def __init__(self) -> None:
        self.boxes: list[tuple[float, float, float, float]] = []

    def block(self, x: float, y: float, r: float) -> None:
        self.boxes.append((x - r, y - r, x + r, y + r))

    def label(self, x: float, y: float, text: str, cls: str, char_px: float, size: float) -> str:
        w = char_px * len(text)
        options = ((9, size + 3, "start"), (9, -7, "start"), (-9, size + 3, "end"), (-9, -7, "end"))
        chosen = None
        for dx, dy, anchor in options:
            x0 = x + dx if anchor == "start" else x + dx - w
            box = (x0, y + dy - size, x0 + w, y + dy + 2)
            inside = box[0] >= _L - 4 and box[2] <= _W - _R
            clear = all(
                box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in self.boxes
            )
            if inside and clear:
                chosen = (dx, dy, anchor, box)
                break
            if chosen is None and inside:
                chosen = (dx, dy, anchor, box)
        if chosen is None:
            dx, dy, anchor = options[2]
            chosen = (dx, dy, anchor, (x + dx - w, y + dy - size, x + dx, y + dy + 2))
        dx, dy, anchor, box = chosen
        self.boxes.append(box)
        end = ' text-anchor="end"' if anchor == "end" else ""
        return f'<text class="{cls}" x="{x + dx:.1f}" y="{y + dy:.1f}"{end}>{escape(text)}</text>'


def _marker(shape: str, x: float, y: float, r: float = 7.0) -> str:
    if shape == "diamond":
        d = f"M{x},{y - r}L{x + r},{y}L{x},{y + r}L{x - r},{y}Z"
    elif shape == "triangle":
        d = f"M{x},{y - r}L{x + r * 0.95},{y + r * 0.7}L{x - r * 0.95},{y + r * 0.7}Z"
    elif shape == "triangle-down":
        d = f"M{x},{y + r}L{x + r * 0.95},{y - r * 0.7}L{x - r * 0.95},{y - r * 0.7}Z"
    elif shape == "square":
        s = r * 0.8
        d = f"M{x - s},{y - s}H{x + s}V{y + s}H{x - s}Z"
    elif shape == "star":
        pts = []
        for k in range(10):
            rr = r if k % 2 == 0 else r * 0.45
            a = -math.pi / 2 + k * math.pi / 5
            pts.append(f"{x + rr * math.cos(a):.2f},{y + rr * math.sin(a):.2f}")
        d = "M" + "L".join(pts) + "Z"
    else:  # cross
        s = r * 0.35
        d = (
            f"M{x - s},{y - r}H{x + s}V{y - s}H{x + r}V{y + s}H{x + s}V{y + r}"
            f"H{x - s}V{y + s}H{x - r}V{y - s}H{x - s}Z"
        )
    return d


def render_svg(ev: Evaluation, title: str | None = None) -> str:
    unit = "task" if ev.level == "task" else "step"
    cfg_pts = ev.frontier_points()
    included = [p for p in cfg_pts if p.label not in ev.frontier_excluded]
    env = envelope(included)
    pols = [
        p
        for p in ev.policies
        if p.metrics["success_rate"].value is not None
        and p.metrics.get(f"cost_per_{unit}") is not None
        and p.metrics[f"cost_per_{unit}"].value is not None
    ]
    xs = [p.cost for p in cfg_pts] + [0.0]
    for p in pols:
        c = p.metrics[f"cost_per_{unit}"]
        xs.append(c.value or 0.0)
        if c.ci is not None and c.ci.estimable and math.isfinite(c.ci.hi):
            xs.append(c.ci.hi)
    x_max = max(xs) * 1.08 or 1.0
    step = _nice_step(x_max)
    x_max = math.ceil(x_max / step) * step
    pw, ph = _W - _L - _R, _H - _T - _B
    height = _T + ph + 62 + 20 * (len(pols) + 2)

    def sx(c: float) -> float:
        return _L + pw * max(0.0, c) / x_max

    def sy(s: float) -> float:
        return _T + ph * (1.0 - min(max(s, 0.0), 1.0))

    out: list[str] = []
    add = out.append
    head = title or f"Success rate against spend, per {unit}"
    src = os.path.basename(str(ev.provenance.get("source", "")).rstrip("/")) or "data"
    sub = f"{src} · {ev.kind} data, {ev.level} level"
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" class="viz" viewBox="0 0 {_W} {height}" '
        f'width="{_W}" height="{height}" role="img" aria-label="{escape(head)}">'
    )
    add(f"<title>{escape(head)}</title><style>{_STYLE}</style>")
    add(f'<rect class="bg" width="{_W}" height="{height}"/>')
    add(f'<text class="title" x="{_L}" y="28">{escape(head)}</text>')
    add(f'<text class="sub" x="{_L}" y="46">{escape(sub)}</text>')
    # grid and ticks
    for k in range(0, 11, 2):
        y = sy(k / 10)
        add(f'<line class="grid" x1="{_L}" x2="{_L + pw}" y1="{y:.1f}" y2="{y:.1f}"/>')
        add(f'<text class="tick" x="{_L - 8}" y="{y + 4:.1f}" text-anchor="end">{k * 10}%</text>')
    n_ticks = int(round(x_max / step))
    for k in range(n_ticks + 1):
        x = sx(k * step)
        add(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{_T}" y2="{_T + ph}"/>')
        add(
            f'<text class="tick" x="{x:.1f}" y="{_T + ph + 18}" text-anchor="middle">'
            f"{escape(_fmt_usd(k * step))}</text>"
        )
    add(f'<line class="axis" x1="{_L}" x2="{_L + pw}" y1="{_T + ph}" y2="{_T + ph}"/>')
    add(f'<line class="axis" x1="{_L}" x2="{_L}" y1="{_T}" y2="{_T + ph}"/>')
    add(
        f'<text class="axis-label" x="{_L + pw / 2}" y="{_T + ph + 38}" text-anchor="middle">'
        f"Spend: mean cost per {unit} (USD)</text>"
    )
    add(
        f'<text class="axis-label" transform="translate({_L - 52},{_T + ph / 2}) rotate(-90)" '
        f'text-anchor="middle">Score: success rate</text>'
    )
    # frontier, extended flat to the right edge
    fpts = [*env, Point("", x_max, env[-1].success)]
    path = "M" + "L".join(f"{sx(p.cost):.1f},{sy(p.success):.1f}" for p in fpts)
    add(
        f'<path class="frontier" d="{path}"><title>Single-configuration frontier: best success '
        f"reachable at each spend by single configurations or random mixes of them</title></path>"
    )
    placer = _Placer()
    pol_xy = []
    for p in pols:
        c_val = p.metrics[f"cost_per_{unit}"].value
        s_val = p.metrics["success_rate"].value
        assert c_val is not None and s_val is not None
        pol_xy.append((sx(c_val), sy(s_val)))
    for x, y in [(sx(p.cost), sy(p.success)) for p in cfg_pts] + pol_xy:
        placer.block(x, y, 7)
    labels = [
        placer.label(x, y, _short(p.name), "pol-label", 7.2, 12)
        for p, (x, y) in zip(pols, pol_xy, strict=True)
    ]
    # configurations
    for cp in cfg_pts:
        x, y = sx(cp.cost), sy(cp.success)
        tip = f"{cp.label}: {100 * cp.success:.1f}% at {_fmt_usd(cp.cost)} per {unit}"
        if cp.label in ev.frontier_excluded:
            tip += f" (not on the frontier: {ev.frontier_excluded[cp.label]})"
        add(
            f'<circle class="cfg" cx="{x:.1f}" cy="{y:.1f}" r="5">'
            f"<title>{escape(tip)}</title></circle>"
        )
        add(placer.label(x, y, _short(cp.label), "cfg-label", 6.0, 10.5))
    # policies
    for i, p in enumerate(pols):
        color = f"var(--s{min(i + 1, 4)})"
        shape = _SHAPES[i % len(_SHAPES)]
        s_est = p.metrics["success_rate"]
        c_est = p.metrics[f"cost_per_{unit}"]
        assert s_est.value is not None and c_est.value is not None
        x, y = sx(c_est.value), sy(s_est.value)
        if s_est.ci is not None and s_est.ci.estimable:
            add(
                f'<line class="whisker" stroke="{color}" x1="{x:.1f}" x2="{x:.1f}" '
                f'y1="{sy(s_est.ci.lo):.1f}" y2="{sy(s_est.ci.hi):.1f}"/>'
            )
        if c_est.ci is not None and c_est.ci.estimable:
            hi = c_est.ci.hi if math.isfinite(c_est.ci.hi) else x_max
            add(
                f'<line class="whisker" stroke="{color}" x1="{sx(c_est.ci.lo):.1f}" '
                f'x2="{sx(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
            )
        tip = (
            f"{p.name}: {100 * s_est.value:.1f}% at {_fmt_usd(c_est.value)} per {unit}; "
            f"verdict {p.verdict.status}"
        )
        add(
            f'<path d="{_marker(shape, x, y)}" fill="{color}" stroke="var(--surface)" '
            f'stroke-width="2"><title>{escape(tip)}</title></path>'
        )
        add(labels[i])
    # legend
    ly = _T + ph + 62
    add(f'<circle class="cfg" cx="{_L + 7}" cy="{ly - 4}" r="5"/>')
    add(f'<text class="legend" x="{_L + 20}" y="{ly}">single configuration</text>')
    add(f'<path class="frontier" d="M{_L + 200},{ly - 4}H{_L + 232}"/>')
    add(
        f'<text class="legend" x="{_L + 240}" y="{ly}">frontier: envelope of single '
        "configurations and the origin</text>"
    )
    words = {BEATS: "beats the frontier", DOES_NOT_BEAT: "does not beat it"}
    for i, p in enumerate(pols):
        yy = ly + 20 * (i + 1)
        color = f"var(--s{min(i + 1, 4)})"
        add(f'<path d="{_marker(_SHAPES[i % len(_SHAPES)], _L + 7, yy - 4, 6)}" fill="{color}"/>')
        verdict = words.get(p.verdict.status, "not shown")
        tag = " (hindsight reference)" if p.hindsight else ""
        add(
            f'<text class="legend" x="{_L + 20}" y="{yy}">'
            f"{escape(_short(p.name, 60))}{tag}: {verdict}</text>"
        )
    missing = [_short(p.name, 30) for p in ev.policies if p not in pols]
    notes = []
    if missing:
        notes.append("not plotted (estimate withheld): " + ", ".join(missing))
    if ev.frontier_excluded:
        notes.append(f"{len(ev.frontier_excluded)} configurations could not be estimated")
    notes.append("whiskers: confidence intervals of success and spend")
    add(
        f'<text class="legend-note" x="{_L}" y="{ly + 20 * (len(pols) + 1)}">'
        f"{escape('; '.join(notes))}</text>"
    )
    add("</svg>")
    return "\n".join(out) + "\n"


def write_svg(path: str | os.PathLike[str], evaluations: Sequence[Evaluation]) -> list[str]:
    """One SVG per evaluation: ``path`` itself for one, ``stem-N.svg`` for several."""
    base = os.fspath(path)
    written = []
    for i, ev in enumerate(evaluations):
        target = base if len(evaluations) == 1 else _numbered(base, i + 1)
        with open(target, "w", encoding="utf-8") as f:
            f.write(render_svg(ev))
        written.append(target)
    return written


def _numbered(path: str, n: int) -> str:
    stem, ext = os.path.splitext(path)
    return f"{stem}-{n}{ext or '.svg'}"
