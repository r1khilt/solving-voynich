#!/usr/bin/env python3
"""Render the audited TEACH-0006 causal-geometry summary as standalone SVG."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "TEACH-0006" / "report.json"
OUTPUT = ROOT / "results" / "TEACH-0006" / "geometry-summary.svg"
BLUE, ORANGE, GREEN, GRAY = "#2458A6", "#E05A33", "#2B8A66", "#9A9388"


def text(x, y, value, size=14, weight="normal", anchor="start", fill="#24221F"):
    return (f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
            f'fill="{fill}">{value}</text>')


def panel(parts, x, y, width, height, title):
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="16" '
                 'fill="#FCFAF6" stroke="#DED8CD"/>')
    parts.append(text(x + 24, y + 34, title, 17, "bold"))


def axes(parts, x, y, width, height):
    left, top, right, bottom = x + 64, y + 60, x + width - 24, y + height - 56
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#776F65"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#776F65"/>')
    for fraction in (0, .5, 1):
        yy = bottom - fraction * (bottom - top)
        parts.append(f'<line x1="{left}" y1="{yy}" x2="{right}" y2="{yy}" '
                     'stroke="#D8D2C8" stroke-dasharray="3 5"/>')
        parts.append(text(left - 10, yy + 5, f"{fraction:.1f}", 11, anchor="end", fill="#6F685F"))
    return left, top, right, bottom


def main():
    report = json.loads(REPORT.read_text())
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1200" viewBox="0 0 1200 1200">',
             '<rect width="1200" height="1200" fill="#F7F4EE"/>']
    parts.append(text(600, 42, "TEACH-0006 · A distributed 11D causal key code", 25,
                      "bold", "middle"))
    boxes = ((28, 66), (614, 66), (28, 438), (614, 438))
    titles = ("Causal transfer rises with subspace rank",
              "The key is low-rank but not neuron-local",
              "Key geometry transfers between models",
              "Moving through latent space flips the answer")
    for (x, y), title_value in zip(boxes, titles, strict=True):
        panel(parts, x, y, 558, 342, title_value)

    x, y = boxes[0]
    left, top, right, bottom = axes(parts, x, y, 558, 342)
    dimensions = [1, 2, 4, 8, 11]
    for rep, color in zip(("0", "1"), (BLUE, ORANGE), strict=True):
        curve = report["metrics"][rep]["dimension_scores"]
        values = [curve[str(d)]["donor_items"]["accuracy"] for d in dimensions]
        points = []
        for index, (dimension, value) in enumerate(zip(dimensions, values, strict=True)):
            xx = left + index * (right - left) / 4
            yy = bottom - value * (bottom - top)
            points.append(f"{xx},{yy}")
            parts.append(f'<circle cx="{xx}" cy="{yy}" r="5" fill="{color}"/>')
            parts.append(text(xx, bottom + 20, str(dimension), 11, anchor="middle", fill="#6F685F"))
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
    parts.append(text((left + right) / 2, bottom + 42, "key-subspace dimensions", 12, anchor="middle"))
    parts.append(text(right - 90, top + 18, "seed 0", 12, fill=BLUE))
    parts.append(text(right - 90, top + 38, "seed 1", 12, fill=ORANGE))

    x, y = boxes[1]
    left, top, right, bottom = axes(parts, x, y, 558, 342)
    labels = ["full", "11D span", "117D rest", "top 11", "top 32"]
    for index, label in enumerate(labels):
        center = left + (index + .5) * (right - left) / len(labels)
        for offset, (rep, color) in enumerate(zip(("0", "1"), (BLUE, ORANGE), strict=True)):
            metric = report["metrics"][rep]
            values = [metric["scores"]["full"]["donor_items"]["accuracy"],
                      metric["scores"]["key_span"]["donor_items"]["accuracy"],
                      metric["scores"]["complement"]["donor_items"]["accuracy"],
                      metric["native_neuron_scores"]["11"]["donor_items"]["accuracy"],
                      metric["native_neuron_scores"]["32"]["donor_items"]["accuracy"]]
            bar_width = 34
            xx = center + (offset - .5) * bar_width
            height = values[index] * (bottom - top)
            parts.append(f'<rect x="{xx - bar_width / 2}" y="{bottom - height}" width="{bar_width}" '
                         f'height="{height}" fill="{color}"/>')
        parts.append(text(center, bottom + 22, label, 10, anchor="middle", fill="#6F685F"))

    x, y = boxes[2]
    left, top, right, bottom = axes(parts, x, y, 558, 342)
    cross = report["cross_seed"]
    for index, name in enumerate(("0 → 1", "1 → 0")):
        center = left + (index + .5) * (right - left) / 2
        key = "0_to_1" if index == 0 else "1_to_0"
        for offset, (value, color) in enumerate(((cross[key]["donor_accuracy"], GREEN),
                                                 (cross[key]["shuffled_donor_accuracy"], GRAY))):
            bar_width = 64
            xx = center + (offset - .5) * bar_width
            height = value * (bottom - top)
            parts.append(f'<rect x="{xx - bar_width / 2}" y="{bottom - height}" width="{bar_width}" '
                         f'height="{height}" fill="{color}"/>')
        parts.append(text(center, bottom + 25, name, 12, anchor="middle"))
    parts.append(text(right - 150, top + 18, "matched alignment", 12, fill=GREEN))
    parts.append(text(right - 150, top + 38, "shuffled alignment", 12, fill=GRAY))

    x, y = boxes[3]
    left, top, right, bottom = axes(parts, x, y, 558, 342)
    fractions = [0, .25, .5, .75, 1]
    for rep, color in zip(("0", "1"), (BLUE, ORANGE), strict=True):
        trajectory = report["metrics"][rep]["trajectory"]
        values = [trajectory[str(float(value))]["mean_donor_probability"] for value in fractions]
        points = []
        for fraction, value in zip(fractions, values, strict=True):
            xx = left + fraction * (right - left)
            yy = bottom - value * (bottom - top)
            points.append(f"{xx},{yy}")
            parts.append(f'<circle cx="{xx}" cy="{yy}" r="5" fill="{color}"/>')
            parts.append(text(xx, bottom + 20, str(fraction), 10, anchor="middle", fill="#6F685F"))
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
    midpoint = left + .5 * (right - left)
    parts.append(f'<line x1="{midpoint}" y1="{top}" x2="{midpoint}" y2="{bottom}" '
                 'stroke="#776F65" stroke-dasharray="5 5"/>')
    parts.append(text((left + right) / 2, bottom + 42, "fraction from base to donor", 12,
                      anchor="middle"))
    parts.append('</svg>')
    OUTPUT.write_text("\n".join(parts) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
