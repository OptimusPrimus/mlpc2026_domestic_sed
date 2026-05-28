from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from _spectrogram_convolution_common import (
    AXIS_LABEL_FONTSIZE,
    N_MELS,
    TRANSPOSE_KERNEL,
    apply_stride,
    hide_axis,
    load_example,
    strided_valid_extent,
)

CONV_STRIDE = 2


def main() -> None:
    sample, _, duration_seconds, logmel, signed_convolution = load_example()
    signed_convolution = signed_convolution
    signed_convolution = apply_stride(signed_convolution, CONV_STRIDE)

    filename = "spectrogram_convolution_result_figure"
    if CONV_STRIDE > 1:
        filename += f"_stride_{CONV_STRIDE}"
    output_pdf = Path(__file__).with_name(f"{filename}.pdf")

    fig = plt.figure(figsize=(8, 4), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0])
    spectrogram_ax = fig.add_subplot(gs[0, 0])
    feature_map_parent_ax = fig.add_subplot(gs[1, 0])

    extent = (0.0, duration_seconds, 0.0, float(N_MELS))
    spectrogram_ax.imshow(logmel, origin="lower", aspect="auto", extent=extent, cmap="magma")
    spectrogram_ax.set_ylabel("Mel bin", fontsize=AXIS_LABEL_FONTSIZE)
    spectrogram_ax.tick_params(axis="x", labelbottom=False)

    hide_axis(feature_map_parent_ax)
    feature_height = signed_convolution.shape[0] / float(logmel.shape[0])
    feature_width = signed_convolution.shape[1] / float(logmel.shape[1])
    feature_map_ax = feature_map_parent_ax.inset_axes(
        [
            0.5 - 0.5 * feature_width,
            0.5 - 0.5 * feature_height,
            feature_width,
            feature_height,
        ]
    )

    signed_limit = float(abs(signed_convolution).max())
    feature_map_ax.imshow(
        signed_convolution,
        origin="lower",
        aspect="auto",
        extent=strided_valid_extent(logmel, duration_seconds, CONV_STRIDE),
        cmap="coolwarm",
        vmin=-signed_limit,
        vmax=signed_limit,
    )
    feature_map_ax.set_ylabel("Feature Map", fontsize=AXIS_LABEL_FONTSIZE)
    feature_map_ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)

    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_pdf.resolve()}")
    print(f"Selected file: {sample['filename']}")
    print(f"Kernel transposed: {TRANSPOSE_KERNEL}")
    print(f"Convolution stride: {CONV_STRIDE}")


if __name__ == "__main__":
    main()
