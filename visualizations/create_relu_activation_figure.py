from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PDF = Path(__file__).with_name("relu_activation_figure.pdf")
ACTIVATION_COLOR = "#1f77b4"
FONT_SIZE = 16


def main() -> None:
    x = np.linspace(-10.0, 10.0, 601)
    y = np.maximum(0.0, x)

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(x, y, color=ACTIVATION_COLOR, linewidth=3.0)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlim(-10.0, 10.0)
    ax.set_ylim(-1.0, 10.0)
    ax.set_xlabel("Input", fontsize=FONT_SIZE)
    ax.set_ylabel("Activation", fontsize=FONT_SIZE)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-1, 0, 2, 4, 6, 8, 10])
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, color="#D0D0D0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figure to {OUTPUT_PDF.resolve()}")


if __name__ == "__main__":
    main()
