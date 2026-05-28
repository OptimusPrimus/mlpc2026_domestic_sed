from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_PDF = Path(__file__).with_name("resnet_prenorm_block_figure.pdf")
BOX_FACE_COLOR = "white"
BOX_EDGE_COLOR = "black"
LINE_WIDTH = 1.5
ARROW_STYLE = "-|>"
FONT_SIZE = 16


def _draw_box(ax: plt.Axes, x: float, y: float, width: float, height: float, label: str) -> None:
    box = patches.Rectangle(
        (x, y),
        width,
        height,
        linewidth=LINE_WIDTH,
        edgecolor=BOX_EDGE_COLOR,
        facecolor=BOX_FACE_COLOR,
    )
    ax.add_patch(box)
    ax.text(x + width / 2.0, y + height / 2.0, label, ha="center", va="center", fontsize=FONT_SIZE)


def _draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": ARROW_STYLE, "linewidth": LINE_WIDTH, "color": "black", "mutation_scale": 18},
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    box_width = 0.10
    box_height = 0.18
    main_y = 0.58
    left_margin = 0.14
    gap = 0.035
    box_xs = [left_margin + index * (box_width + gap) for index in range(6)]
    labels = ["BN", "ReLU", "Conv", "BN", "ReLU", "Conv"]

    ax.text(0.05, main_y + box_height / 2.0, "x", ha="center", va="center", fontsize=FONT_SIZE)
    ax.text(0.95, main_y + box_height / 2.0, "y", ha="center", va="center", fontsize=FONT_SIZE)

    for x, label in zip(box_xs, labels, strict=True):
        _draw_box(ax, x, main_y, box_width, box_height, label)

    _draw_arrow(ax, (0.065, main_y + box_height / 2.0), (box_xs[0], main_y + box_height / 2.0))
    for left_x, right_x in zip(box_xs[:-1], box_xs[1:], strict=True):
        _draw_arrow(
            ax,
            (left_x + box_width, main_y + box_height / 2.0),
            (right_x, main_y + box_height / 2.0),
        )

    sum_center = (0.88, main_y + box_height / 2.0)
    sum_radius = 0.035
    sum_circle = patches.Circle(sum_center, radius=sum_radius, linewidth=LINE_WIDTH, edgecolor="black", facecolor="white")
    ax.add_patch(sum_circle)
    ax.text(sum_center[0], sum_center[1], "+", ha="center", va="center", fontsize=FONT_SIZE)

    _draw_arrow(
        ax,
        (box_xs[-1] + box_width, main_y + box_height / 2.0),
        (sum_center[0] - sum_radius, sum_center[1]),
    )
    _draw_arrow(ax, (sum_center[0] + sum_radius, sum_center[1]), (0.935, sum_center[1]))

    skip_y = 0.24
    ax.plot([0.05, 0.05], [main_y + box_height / 2.0, skip_y], color="black", linewidth=LINE_WIDTH)
    ax.plot([0.05, sum_center[0]], [skip_y, skip_y], color="black", linewidth=LINE_WIDTH)
    _draw_arrow(ax, (sum_center[0], skip_y), (sum_center[0], sum_center[1] - sum_radius))

    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUTPUT_PDF.resolve()}")


if __name__ == "__main__":
    main()
