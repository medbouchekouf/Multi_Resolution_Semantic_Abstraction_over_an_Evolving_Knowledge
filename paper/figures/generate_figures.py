"""Generate publication figures from TRAIL-Hyper experiment artifacts.

Usage (from repository root):
  PYTHONPATH=src python paper/figures/generate_figures.py \
    --hypergraph PATH --experiments artifacts/experiments.json --out paper/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from trail_hyper.data import load_hypergraph


W, H, M = 1120, 610, 90
FONT = ImageFont.load_default()


def canvas(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)
    draw.text((M, 26), title, fill="#111827", font=FONT)
    draw.line((M, H - M, W - M, H - M), fill="#374151", width=2)
    draw.line((M, 75, M, H - M), fill="#374151", width=2)
    return image, draw


def xy(values: list[float], low: float, high: float, left: int = M, right: int = W - M) -> list[float]:
    return [left + (right - left) * (v - low) / (high - low) for v in values]


def save_growth(summaries: list[dict], out: Path) -> None:
    years = [s["cutoff"] for s in summaries]; nodes = [s["nodes"] for s in summaries]; edges = [s["hyperedges"] for s in summaries]
    image, draw = canvas("Cumulative knowledge-hypergraph growth")
    xs = xy(years, min(years), max(years)); yn = xy(nodes, 0, max(nodes), H - M, 75); ye = xy(edges, 0, max(edges), H - M, 75)
    draw.line(list(zip(xs, yn)), fill="#2563eb", width=4); draw.line(list(zip(xs, ye)), fill="#dc2626", width=4)
    for year, x, n, e, a, b in zip(years, xs, nodes, edges, yn, ye):
        draw.ellipse((x - 6, a - 6, x + 6, a + 6), fill="#2563eb"); draw.rectangle((x - 6, b - 6, x + 6, b + 6), fill="#dc2626")
        draw.text((x - 12, H - M + 12), str(year), fill="#111827", font=FONT); draw.text((x + 8, a - 12), str(n), fill="#2563eb", font=FONT); draw.text((x + 8, b + 4), str(e), fill="#dc2626", font=FONT)
    draw.text((M, 55), "Blue circles: nodes    Red squares: hyperedges", fill="#374151", font=FONT)
    image.save(out / "snapshot_growth.png")


def save_arity(summaries: list[dict], out: Path) -> None:
    image, draw = canvas("Native hyperedge arity distribution")
    colors = ["#2563eb", "#d97706", "#16a34a"]; all_x = sorted({int(k) for s in summaries for k in s["arity_distribution"]}); maximum = max(int(v) for s in summaries for v in s["arity_distribution"].values())
    for summary, color in zip(summaries, colors):
        values = summary["arity_distribution"]; xs = xy(all_x, min(all_x), max(all_x)); ys = xy([int(values.get(str(a), 0)) for a in all_x], 0, maximum, H - M, 75)
        draw.line(list(zip(xs, ys)), fill=color, width=3)
        for x, y in zip(xs, ys): draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    for x in [2, 10, 30, 65]: draw.text((xy([x], min(all_x), max(all_x))[0] - 8, H - M + 12), str(x), fill="#111827", font=FONT)
    draw.text((M, 55), "Blue: 2022    Orange: 2024    Green: 2026", fill="#374151", font=FONT); draw.text((M, H - 45), "Hyperedge arity", fill="#111827", font=FONT)
    image.save(out / "arity_distribution.png")


def save_tradeoff(experiment: dict, out: Path) -> None:
    image, draw = canvas("Observed structural-temporal trade-off")
    colors = {"structural_only": "#2563eb", "semantic_structural": "#d97706", "semantic_emphasis": "#16a34a"}
    points = [p for rows in experiment.values() for p in rows[1:]]; x0 = min(p["stability"] for p in points) - .01; x1 = max(p["stability"] for p in points) + .01; y0 = min(p["hyperedge_retention"] for p in points) - .01; y1 = max(p["hyperedge_retention"] for p in points) + .01
    for name, rows in experiment.items():
        for point in rows[1:]:
            x = xy([point["stability"]], x0, x1)[0]; y = xy([point["hyperedge_retention"]], y0, y1, H - M, 75)[0]
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colors[name]); draw.text((x + 8, y - 10), str(point["cutoff"]), fill="#111827", font=FONT)
    draw.text((M, 55), "Blue: structural-only   Orange: GRU semantic-structural   Green: semantic emphasis", fill="#374151", font=FONT)
    draw.text((M, H - 45), "Temporal membership agreement", fill="#111827", font=FONT)
    image.save(out / "retention_stability.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypergraph", required=True)
    parser.add_argument("--experiments", required=True)
    parser.add_argument("--out", default="paper/figures")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    graph = load_hypergraph(args.hypergraph)
    cutoffs = [2022, 2024, 2026]
    summaries = [graph.snapshot(c).describe() for c in cutoffs]

    experiment = json.loads(Path(args.experiments).read_text(encoding="utf-8"))["baselines"]
    save_growth(summaries, out); save_arity(summaries, out); save_tradeoff(experiment, out)


if __name__ == "__main__":
    main()
