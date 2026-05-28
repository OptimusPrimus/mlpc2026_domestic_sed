from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _spectrogram_convolution_common import AXIS_LABEL_FONTSIZE, hide_axis, load_example, plot_arrow


def main() -> None:
    sample, kernel, _, _, _ = load_example()
    output_pdf = Path(__file__).with_name("spectrogram_convolution_kernel_figure.pdf")

    fig = plt.figure(figsize=(8, 4), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.5, 1.0, 1.0])
    kernel_ax = fig.add_subplot(gs[0, 0])
    middle_ax = fig.add_subplot(gs[1, 0])
    lower_ax = fig.add_subplot(gs[2, 0])

    hide_axis(kernel_ax)
    hide_axis(middle_ax)
    hide_axis(lower_ax)

    kernel_inset = [0.28, 0.06, 0.44, 1.0]
    kernel_image_ax = middle_ax.inset_axes(kernel_inset)
    kernel_limit = float(np.max(np.abs(kernel)))
    kernel_image_ax.imshow(
        kernel,
        origin="lower",
        aspect="equal",
        cmap="coolwarm",
        vmin=-kernel_limit,
        vmax=kernel_limit,
        extent=(0.0, float(kernel.shape[1]), 0.0, float(kernel.shape[0])),
    )
    kernel_image_ax.set_title("2D-kernel", fontsize=AXIS_LABEL_FONTSIZE)
    kernel_image_ax.set_xticks(np.arange(kernel.shape[1]) + 0.5, labels=[str(index) for index in range(kernel.shape[1])])
    kernel_image_ax.set_yticks(np.arange(kernel.shape[0]) + 0.5, labels=[str(index) for index in range(kernel.shape[0])])
    kernel_image_ax.set_xticks(np.arange(0.0, kernel.shape[1] + 1.0, 1.0), minor=True)
    kernel_image_ax.set_yticks(np.arange(0.0, kernel.shape[0] + 1.0, 1.0), minor=True)
    kernel_image_ax.set_xlim(0.0, float(kernel.shape[1]))
    kernel_image_ax.set_ylim(0.0, float(kernel.shape[0]))
    kernel_image_ax.grid(which="minor", color="#7A7A7A", linestyle="-", linewidth=0.6)
    kernel_image_ax.tick_params(axis="x", labelbottom=False, bottom=False)
    kernel_image_ax.tick_params(which="minor", bottom=False, left=False)
    for spine in kernel_image_ax.spines.values():
        spine.set_visible(False)

    kernel_left = kernel_inset[0]
    kernel_right = kernel_inset[0] + kernel_inset[2]
    kernel_center = kernel_inset[0] + 0.5 * kernel_inset[2]
    plot_arrow(middle_ax, (kernel_left, 0.68), (0.04, 0.68))
    plot_arrow(middle_ax, (kernel_right, 0.68), (0.96, 0.68))
    plot_arrow(kernel_ax, (kernel_center, 0.10), (kernel_center, 0.90))
    plot_arrow(lower_ax, (kernel_center, 0.92), (kernel_center, 0.18))

    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_pdf.resolve()}")
    print(f"Selected file: {sample['filename']}")


if __name__ == "__main__":
    main()
