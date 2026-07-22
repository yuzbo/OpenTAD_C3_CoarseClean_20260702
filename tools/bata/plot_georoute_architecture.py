#!/usr/bin/env python3
"""Render the GeoRoute-AdaTAD architecture FigureSpec outside the repository.

The schematic is generated from a checked-in JSON source rather than hand
drawn after results are known.  It is an explanatory figure only: numerical
claims remain tied to ``georoute-paper-result-v1`` and its analyzer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _load_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "georoute-architecture-figurespec-v1":
        raise ValueError("unexpected GeoRoute architecture FigureSpec schema")
    if not isinstance(payload.get("canvas"), Mapping) or not isinstance(payload.get("nodes"), list):
        raise ValueError("architecture FigureSpec lacks canvas or nodes")
    return payload


def _center(node: Mapping[str, Any]) -> tuple[float, float]:
    return float(node["x"]), float(node["y"])


def render(spec: Mapping[str, Any], *, output: Path, title: str | None = None) -> Path:
    canvas = spec["canvas"]
    width, height = float(canvas["width"]), float(canvas["height"])
    figure, axis = plt.subplots(figsize=(width / 190.0, height / 190.0))
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.axis("off")

    nodes = {str(node["id"]): node for node in spec["nodes"]}
    for group in spec.get("groups", []):
        group_nodes = [nodes[node_id] for node_id in group["node_ids"]]
        left = min(float(node["x"]) - float(node["width"]) / 2 for node in group_nodes) - 24
        right = max(float(node["x"]) + float(node["width"]) / 2 for node in group_nodes) + 24
        top = min(float(node["y"]) - float(node["height"]) / 2 for node in group_nodes) - 42
        bottom = max(float(node["y"]) + float(node["height"]) / 2 for node in group_nodes) + 26
        axis.add_patch(
            FancyBboxPatch(
                (left, top),
                right - left,
                bottom - top,
                boxstyle="round,pad=0.02,rounding_size=12",
                facecolor=group["fill"],
                edgecolor=group["stroke"],
                linewidth=1.1,
                alpha=0.52,
                zorder=0,
            )
        )
        axis.text(left + 10, top + 18, group["label"], fontsize=8.5, weight="bold", color=group["stroke"], zorder=1)

    for edge in spec.get("edges", []):
        source = nodes[str(edge["from"])]
        target = nodes[str(edge["to"])]
        source_xy, target_xy = _center(source), _center(target)
        style = "--" if edge.get("style") == "dashed" else "-"
        connectionstyle = "arc3,rad=-0.26" if edge.get("curve") else "arc3,rad=0.0"
        arrow = FancyArrowPatch(
            source_xy,
            target_xy,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.4,
            linestyle=style,
            color=edge.get("color", "#333333"),
            connectionstyle=connectionstyle,
            shrinkA=36,
            shrinkB=42,
            zorder=2,
        )
        axis.add_patch(arrow)
        if edge.get("label"):
            mid_x = 0.5 * (source_xy[0] + target_xy[0])
            mid_y = 0.5 * (source_xy[1] + target_xy[1]) - (28 if edge.get("curve") else 10)
            axis.text(mid_x, mid_y, edge["label"], fontsize=6.9, ha="center", va="center", color=edge.get("color", "#333333"), zorder=3)

    for node in spec["nodes"]:
        x, y = _center(node)
        node_width, node_height = float(node["width"]), float(node["height"])
        axis.add_patch(
            FancyBboxPatch(
                (x - node_width / 2, y - node_height / 2),
                node_width,
                node_height,
                boxstyle="round,pad=0.02,rounding_size=8",
                facecolor="white",
                edgecolor="#3A3A3A",
                linewidth=0.9,
                zorder=4,
            )
        )
        axis.text(x, y, node["label"], fontsize=7.6, ha="center", va="center", zorder=5)
    if title:
        axis.set_title(title, fontsize=11, pad=9, weight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    output = args.output.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    if repository_root in output.parents:
        parser.error("generated figure must be written outside the repository")
    render(_load_spec(args.spec), output=output, title=args.title)
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
