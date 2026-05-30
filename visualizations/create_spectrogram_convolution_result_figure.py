from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _spectrogram_convolution_common import (
    AXIS_LABEL_FONTSIZE,
    N_MELS,
    TRANSPOSE_KERNEL,
    apply_stride,
    hide_axis,
    load_example,
    strided_valid_extent,
)

CONV_STRIDE = 1
SHOW_AXIS_LABELS = False
FIGURE_SIZE = (12, 3)


def normalize_feature_map(feature_map: np.ndarray) -> np.ndarray:
    mean = float(feature_map.mean())
    std = float(feature_map.std())
    if std == 0.0:
        return feature_map - mean
    return (feature_map - mean) / std


def apply_axis_labels(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    if SHOW_AXIS_LABELS:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)


def save_spectrogram_figure(logmel: np.ndarray, duration_seconds: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    extent = (0.0, duration_seconds, 0.0, float(N_MELS))
    ax.imshow(logmel, origin="lower", aspect="auto", extent=extent, cmap="magma")
    apply_axis_labels(ax, "Time (s)", "Mel bin")
    save_figure(fig, output_path)
    plt.close(fig)


def save_feature_map_figure(
    feature_map: np.ndarray,
    logmel: np.ndarray,
    duration_seconds: float,
    output_path: Path,
    ylabel: str,
    cmap: str,
    symmetric_limits: bool,
) -> None:
    fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
    parent_ax = fig.add_subplot(1, 1, 1)
    hide_axis(parent_ax)

    feature_height = feature_map.shape[0] / float(logmel.shape[0])
    feature_width = feature_map.shape[1] / float(logmel.shape[1])
    feature_map_ax = parent_ax.inset_axes(
        [
            0.5 - 0.5 * feature_width,
            0.5 - 0.5 * feature_height,
            feature_width,
            feature_height,
        ]
    )

    imshow_kwargs = {
        "origin": "lower",
        "aspect": "auto",
        "extent": strided_valid_extent(logmel, duration_seconds, CONV_STRIDE),
        "cmap": cmap,
    }
    if symmetric_limits:
        limit = float(np.abs(feature_map).max())
        imshow_kwargs["vmin"] = -limit
        imshow_kwargs["vmax"] = limit
    else:
        imshow_kwargs["vmin"] = 0.0

    feature_map_ax.imshow(feature_map, **imshow_kwargs)
    apply_axis_labels(feature_map_ax, "Time (s)", ylabel)

    save_figure(fig, output_path)
    plt.close(fig)


def save_figure(fig: plt.Figure, pdf_path: Path) -> None:
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(pdf_path.with_suffix(".svg"), format="svg")


def main() -> None:
    sample, _, duration_seconds, logmel, signed_convolution = load_example()
    signed_convolution = apply_stride(signed_convolution, CONV_STRIDE)
    normalized_convolution = normalize_feature_map(signed_convolution)
    relu_convolution = np.maximum(normalized_convolution, 0.0)

    stride_suffix = f"_stride_{CONV_STRIDE}" if CONV_STRIDE > 1 else ""
    output_dir = Path(__file__).parent
    spectrogram_pdf = output_dir / "spectrogram_convolution_input_spectrogram.pdf"
    unnormalized_pdf = output_dir / f"spectrogram_convolution_result_unnormalized{stride_suffix}.pdf"
    normalized_pdf = output_dir / f"spectrogram_convolution_result_normalized{stride_suffix}.pdf"
    relu_pdf = output_dir / f"spectrogram_convolution_result_normalized_relu{stride_suffix}.pdf"

    save_spectrogram_figure(logmel, duration_seconds, spectrogram_pdf)
    save_feature_map_figure(
        signed_convolution,
        logmel,
        duration_seconds,
        unnormalized_pdf,
        ylabel="Feature Map",
        cmap="coolwarm",
        symmetric_limits=True,
    )
    save_feature_map_figure(
        normalized_convolution,
        logmel,
        duration_seconds,
        normalized_pdf,
        ylabel="Normalized Map",
        cmap="coolwarm",
        symmetric_limits=True,
    )
    save_feature_map_figure(
        relu_convolution,
        logmel,
        duration_seconds,
        relu_pdf,
        ylabel="ReLU Map",
        cmap="magma",
        symmetric_limits=False,
    )

    print(f"Saved figure to {spectrogram_pdf.resolve()}")
    print(f"Saved figure to {spectrogram_pdf.with_suffix('.svg').resolve()}")
    print(f"Saved figure to {unnormalized_pdf.resolve()}")
    print(f"Saved figure to {unnormalized_pdf.with_suffix('.svg').resolve()}")
    print(f"Saved figure to {normalized_pdf.resolve()}")
    print(f"Saved figure to {normalized_pdf.with_suffix('.svg').resolve()}")
    print(f"Saved figure to {relu_pdf.resolve()}")
    print(f"Saved figure to {relu_pdf.with_suffix('.svg').resolve()}")
    print(f"Selected file: {sample['filename']}")
    print(f"Kernel transposed: {TRANSPOSE_KERNEL}")
    print(f"Convolution stride: {CONV_STRIDE}")


if __name__ == "__main__":
    main()
