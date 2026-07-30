#!/usr/bin/env python3
"""Render the public Mage-VL reproduction figures from aggregated measurements."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts/mage-vl-codec-reproduction/results.json"
OUT = ROOT / "reports/mage-vl-codec-reproduction/images"

COLORS = {"codec": "#146C94", "uniform": "#D95F02", "dcvc": "#6A3D9A"}


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_start(title: str, width: int = 1000, height: int = 520) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#17202A}.title{font-size:24px;font-weight:700}.axis{font-size:14px}.note{font-size:13px;fill:#4D5656}.grid{stroke:#D5D8DC;stroke-width:1}.legend{font-size:14px;font-weight:600}</style>",
        f'<rect width="{width}" height="{height}" fill="#FCFCFA"/>',
        f'<text x="55" y="38" class="title">{esc(title)}</text>',
    ]


def finish(lines: list[str], path: Path) -> None:
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def scatter(data: dict, x_key: str, title: str, xlabel: str, filename: str) -> None:
    points = []
    for row in data["budget_sweep"]:
        for kind in ("codec", "uniform"):
            points.append((row[kind][x_key], row[kind]["accuracy_pct"], kind, str(row["budget"])))
    md = data["matched_duration"]
    if x_key != "tokens":
        points.append((md["codec"][x_key], md["codec"]["accuracy_pct"], "codec", "tc8/64f"))
    points.append((md["uniform"][x_key], md["uniform"]["accuracy_pct"], "uniform", "64f"))
    tm = data["approximately_token_matched"]
    points.append((tm["codec"][x_key], tm["codec"]["accuracy_pct"], "codec", "tc84*"))
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = 66, 83
    x0, x1, y0, y1 = 90, 790, 445, 82
    lx0, lx1 = math.log10(xmin), math.log10(xmax)
    xp = lambda x: x0 + (math.log10(x) - lx0) / (lx1 - lx0) * (x1 - x0)
    yp = lambda y: y0 - (y - ymin) / (ymax - ymin) * (y0 - y1)
    lines = svg_start(title)
    for y in (68, 72, 76, 80):
        lines += [f'<line x1="{x0}" y1="{yp(y):.1f}" x2="{x1}" y2="{yp(y):.1f}" class="grid"/>',
                  f'<text x="{x0-12}" y="{yp(y)+5:.1f}" text-anchor="end" class="axis">{y}%</text>']
    for kind in ("uniform", "codec"):
        seq = sorted([p for p in points if p[2] == kind], key=lambda p: p[0])
        coords = " ".join(f"{xp(p[0]):.1f},{yp(p[1]):.1f}" for p in seq)
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{COLORS[kind]}" stroke-width="2" opacity=".55"/>')
    for x, y, kind, label in points:
        lines.append(f'<circle cx="{xp(x):.1f}" cy="{yp(y):.1f}" r="7" fill="{COLORS[kind]}" stroke="#fff" stroke-width="2"/>')
        lines.append(f'<text x="{xp(x)+9:.1f}" y="{yp(y)-7:.1f}" class="note">{esc(label)}</text>')
    lines += [
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#17202A"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#17202A"/>',
        f'<text x="{(x0+x1)/2}" y="495" text-anchor="middle" class="axis">{esc(xlabel)} (log scale)</text>',
        '<circle cx="590" cy="60" r="6" fill="#146C94"/><text x="602" y="65" class="legend">codec-guided</text>',
        '<circle cx="730" cy="60" r="6" fill="#D95F02"/><text x="742" y="65" class="legend">uniform</text>',
        '<text x="90" y="510" class="note">Labels are nominal N; *tc84 is the approximately token-matched stress test.</text>',
    ]
    finish(lines, OUT / filename)


def token_reduction(data: dict) -> None:
    labels = ["tc4", "tc8", "tc16", "tc32", "tc8 vs 64f", "BBB 60s", "BBB 120s"]
    vals = [100 * (1 - r["codec"]["tokens"] / r["uniform"]["tokens"]) for r in data["budget_sweep"]]
    vals += [data["matched_duration"]["token_reduction_pct"], 96.41, 96.41]
    lines = svg_start("Codec selection removed at least 90% of measured visual tokens")
    x0, y0, top, bw, gap = 90, 435, 75, 75, 29
    for y in (75, 80, 90, 100):
        yy = y0 - (y - 70) / 30 * (y0 - top)
        lines += [f'<line x1="{x0}" y1="{yy:.1f}" x2="850" y2="{yy:.1f}" class="grid"/>',
                  f'<text x="78" y="{yy+5:.1f}" text-anchor="end" class="axis">{y}%</text>']
    for i, (lab, val) in enumerate(zip(labels, vals)):
        x = x0 + 25 + i * (bw + gap)
        h = (val - 70) / 30 * (y0 - top)
        lines += [f'<rect x="{x}" y="{y0-h:.1f}" width="{bw}" height="{h:.1f}" rx="5" fill="#146C94"/>',
                  f'<text x="{x+bw/2}" y="{y0-h-9:.1f}" text-anchor="middle" class="legend">{val:.1f}%</text>',
                  f'<text x="{x+bw/2}" y="{y0+23}" text-anchor="middle" class="axis">{esc(lab)}</text>']
    threshold = y0 - (75 - 70) / 30 * (y0 - top)
    lines.append(f'<line x1="{x0}" y1="{threshold:.1f}" x2="850" y2="{threshold:.1f}" stroke="#C0392B" stroke-width="2" stroke-dasharray="8 6"/>')
    lines.append('<text x="675" y="416" class="note" fill="#C0392B">paper headline threshold</text>')
    finish(lines, OUT / "token_reduction.svg")


def cross_codec(data: dict) -> None:
    cc = data["alternate_codec"]
    lines = svg_start("Alternate codec: same accuracy, different selected patches")
    for i, (label, key, color) in enumerate([("HEVC", "hevc", COLORS["codec"]), ("DCVC-RT", "dcvc_rt", COLORS["dcvc"]), ("Uniform", "uniform", COLORS["uniform"])]):
        x = 105 + i * 180
        val = cc[key]["accuracy_pct"]
        h = val / 100 * 315
        lines += [f'<rect x="{x}" y="{430-h:.1f}" width="115" height="{h:.1f}" rx="6" fill="{color}"/>',
                  f'<text x="{x+57.5}" y="{420-h:.1f}" text-anchor="middle" class="title" font-size="20">{val:.1f}%</text>',
                  f'<text x="{x+57.5}" y="456" text-anchor="middle" class="legend">{label}</text>',
                  f'<text x="{x+57.5}" y="477" text-anchor="middle" class="note">{cc[key]["tokens"]:,} tokens</text>']
    lines += [
        '<rect x="650" y="105" width="195" height="235" rx="10" fill="#EEF2F7"/>',
        '<text x="748" y="145" text-anchor="middle" class="legend">HEVC ↔ DCVC top-k</text>',
        f'<text x="748" y="210" text-anchor="middle" class="title">{100*cc["median_topk_jaccard"]:.1f}%</text>',
        '<text x="748" y="234" text-anchor="middle" class="note">median Jaccard</text>',
        f'<text x="748" y="292" text-anchor="middle" class="title">{100*cc["median_smaller_set_coverage"]:.1f}%</text>',
        '<text x="748" y="316" text-anchor="middle" class="note">smaller-set coverage</text>',
        '<text x="650" y="382" class="note">36 questions; no retraining.</text>'
    ]
    finish(lines, OUT / "cross_codec.svg")


def memory(data: dict) -> None:
    rows = data["budget_sweep"]
    lines = svg_start("Sparse selection kept peak GPU memory nearly flat")
    x0, x1, y0, y1 = 100, 840, 430, 80
    xp = lambda i: x0 + i / 3 * (x1 - x0)
    yp = lambda y: y0 - (y - 8) / 34 * (y0 - y1)
    for y in (10, 20, 30, 40):
        lines += [f'<line x1="{x0}" y1="{yp(y):.1f}" x2="{x1}" y2="{yp(y):.1f}" class="grid"/>',
                  f'<text x="88" y="{yp(y)+5:.1f}" text-anchor="end" class="axis">{y} GiB</text>']
    for kind in ("codec", "uniform"):
        coords = " ".join(f"{xp(i):.1f},{yp(r[kind]['peak_gib']):.1f}" for i, r in enumerate(rows))
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{COLORS[kind]}" stroke-width="4"/>')
        for i, r in enumerate(rows):
            lines.append(f'<circle cx="{xp(i):.1f}" cy="{yp(r[kind]["peak_gib"]):.1f}" r="7" fill="{COLORS[kind]}"/>')
    for i, r in enumerate(rows):
        lines.append(f'<text x="{xp(i):.1f}" y="462" text-anchor="middle" class="axis">N={r["budget"]}</text>')
    lines += [
        '<circle cx="640" cy="35" r="6" fill="#146C94"/><text x="652" y="40" class="legend">codec-guided</text>',
        '<circle cx="770" cy="35" r="6" fill="#D95F02"/><text x="782" y="40" class="legend">uniform</text>',
        '<text x="470" y="500" text-anchor="middle" class="note">Peak allocated memory reported by PyTorch; single-sample inference.</text>',
    ]
    finish(lines, OUT / "memory.svg")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS.read_text())
    scatter(data, "tokens", "Accuracy versus measured visual tokens", "retained visual tokens", "accuracy_tokens.svg")
    scatter(data, "e2e_s", "Accuracy versus preprocessing-inclusive wall time", "warm end-to-end seconds", "accuracy_time.svg")
    token_reduction(data)
    cross_codec(data)
    memory(data)
    print(f"Rendered 5 figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
