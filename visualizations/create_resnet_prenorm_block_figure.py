from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_PDF = Path(__file__).with_name("resnet_prenorm_block_figure.pdf")
BOX_FACE_COLOR = "white"
BOX_EDGE_COLOR = "black"
LINE_WIDTH = 1.5
ARROW_STYLE = "-|>"
FONT_SIZE = 20


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
    fig, ax = plt.subplots(figsize=(4.7, 5.6))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_xlim(0.24, 0.86)
    ax.set_ylim(-0.02, 0.97)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    box_width = 0.36
    box_height = 0.06
    main_x = 0.32
    top_margin = 0.78
    gap = 0.055
    box_ys = [top_margin - index * (box_height + gap) for index in range(6)]
    labels = ["BatchNorm", "ReLU", "Conv", "BatchNorm", "ReLU", "Conv"]

    ax.text(main_x + box_width / 2.0, 0.93, "x", ha="center", va="center", fontsize=FONT_SIZE)
    ax.text(0.50, 0.00, "y", ha="center", va="center", fontsize=FONT_SIZE)

    for y, label in zip(box_ys, labels, strict=True):
        _draw_box(ax, main_x, y, box_width, box_height, label)

    _draw_arrow(ax, (main_x + box_width / 2.0, 0.905), (main_x + box_width / 2.0, box_ys[0] + box_height))
    for upper_y, lower_y in zip(box_ys[:-1], box_ys[1:], strict=True):
        _draw_arrow(
            ax,
            (main_x + box_width / 2.0, upper_y),
            (main_x + box_width / 2.0, lower_y + box_height),
        )

    sum_center = (0.50, 0.12)
    sum_radius = 0.035
    sum_circle = patches.Circle(sum_center, radius=sum_radius, linewidth=LINE_WIDTH, edgecolor="black", facecolor="white")
    ax.add_patch(sum_circle)
    ax.text(sum_center[0], sum_center[1], "+", ha="center", va="center", fontsize=FONT_SIZE)

    _draw_arrow(
        ax,
        (main_x + box_width / 2.0, box_ys[-1]),
        (sum_center[0], sum_center[1] + sum_radius),
    )
    _draw_arrow(ax, (sum_center[0], sum_center[1] - sum_radius), (0.50, 0.015))

    skip_x = 0.82
    x_center = main_x + box_width / 2.0
    ax.plot([x_center + 0.05, skip_x], [0.93, 0.93], color="black", linewidth=LINE_WIDTH)
    ax.plot([skip_x, skip_x], [0.93, sum_center[1]], color="black", linewidth=LINE_WIDTH)
    _draw_arrow(ax, (skip_x, sum_center[1]), (sum_center[0] + sum_radius, sum_center[1]))

    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figure to {OUTPUT_PDF.resolve()}")


if __name__ == "__main__":
    main()
