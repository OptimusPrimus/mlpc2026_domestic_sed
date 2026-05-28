from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from domestic_sed.dataset import MLPC2026SoundEventDataset
from domestic_sed.metrics.segment_based_metrics import aggregate_ground_truth_annotations

DATASET_ROOT = Path(os.environ.get("DOMESTIC_SED_ROOT", "~/data/mlpc2026_dataset/MLPC2026_challenge_dataset_raw")).expanduser()
TRAIN_SAMPLE_INDEX = 11
TARGET_FILENAME: str | None = None
OUTPUT_PDF = Path(__file__).with_name("sed_system_overview.pdf")
MAX_CLASSES = 3
ALTERNATIVE_PREDICTION_MATRIX = False
WAVEFORM_COLOR = "#0078aa"
BOX_FACE_COLOR = "#D9D9D9"
ACTIVE_CELL_COLOR = "#525252"
TRUE_POSITIVE_CELL_COLOR = "#2E8B57"
FALSE_POSITIVE_CELL_COLOR = "#C62828"
INACTIVE_CELL_COLOR = "#F5F5F5"


def _select_sample(dataset: MLPC2026SoundEventDataset) -> dict[str, object]:
    if TARGET_FILENAME is not None:
        for sample in dataset:
            if sample["filename"] == TARGET_FILENAME:
                return sample
        raise ValueError(f"Could not find {TARGET_FILENAME!r} in the training split.")

    if not 0 <= TRAIN_SAMPLE_INDEX < len(dataset):
        raise IndexError(f"TRAIN_SAMPLE_INDEX={TRAIN_SAMPLE_INDEX} is out of range for {len(dataset)} training files.")
    return dataset[TRAIN_SAMPLE_INDEX]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []

    merged = [sorted(intervals, key=lambda interval: interval[0])[0]]
    for start, end in sorted(intervals, key=lambda interval: interval[0])[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _prepare_annotations(sample: dict[str, object]) -> pd.DataFrame:
    raw_annotations = sample["annotations"]
    if not isinstance(raw_annotations, pd.DataFrame):
        raise TypeError("Sample annotations are not a pandas DataFrame.")
    if raw_annotations.empty:
        raise ValueError(f"No annotations available for {sample['filename']}.")

    aggregated = aggregate_ground_truth_annotations(raw_annotations)
    if aggregated.empty:
        grouped_rows: list[dict[str, object]] = []
        for annotation, group in raw_annotations.groupby("annotation", sort=False):
            merged = _merge_intervals(list(zip(group["onset"], group["offset"])))
            for onset, offset in merged:
                grouped_rows.append(
                    {
                        "filename": sample["filename"],
                        "annotation": annotation,
                        "onset": onset,
                        "offset": offset,
                    }
                )
        annotations = pd.DataFrame(grouped_rows, columns=["filename", "annotation", "onset", "offset"])
    else:
        annotations = aggregated.copy()

    annotations = annotations.loc[annotations["offset"] > annotations["onset"]].copy()
    if annotations.empty:
        raise ValueError(f"No valid annotation intervals available for {sample['filename']}.")
    return annotations


def _build_annotation_matrix(annotations: pd.DataFrame, clip_duration_seconds: float) -> tuple[np.ndarray, list[str], int]:
    per_class = annotations.groupby("annotation", sort=False).agg(first_onset=("onset", "min"))
    per_class["total_duration"] = (
        annotations.groupby("annotation", sort=False)
        .apply(lambda group: float((group["offset"] - group["onset"]).sum()), include_groups=False)
        .reindex(per_class.index)
    )
    per_class = per_class.sort_values(["first_onset", "total_duration"], ascending=[True, False])
    classes = per_class.index.tolist()[:MAX_CLASSES]
    if not classes:
        raise ValueError("No annotated classes found for the selected recording.")

    total_seconds = max(1, math.ceil(clip_duration_seconds))
    matrix = np.zeros((len(classes), total_seconds), dtype=int)

    for row_index, annotation in enumerate(classes):
        class_rows = annotations.loc[annotations["annotation"] == annotation]
        for row in class_rows.itertuples(index=False):
            start_second = max(0, math.floor(float(row.onset)))
            end_second = min(total_seconds, math.ceil(float(row.offset)))
            if end_second > start_second:
                matrix[row_index, start_second:end_second] = 1

    return matrix, classes, total_seconds


def _plot_waveform(ax: plt.Axes, waveform: np.ndarray, sample_rate: int, total_seconds: int) -> None:
    time_axis = np.arange(waveform.shape[0], dtype=np.float64) / float(sample_rate)
    ax.plot(time_axis, waveform, color=WAVEFORM_COLOR, linewidth=1.0)
    ax.set_xlim(0.0, float(total_seconds))
    max_abs = np.max(np.abs(waveform))
    ylim = max(max_abs * 1.05, 1e-3)
    ax.set_ylim(-ylim, ylim)
    # ax.text(-0.02, 0.5, "waveform", transform=ax.transAxes, ha="right", va="center", fontsize=17)
    ax.axis("off")


def _plot_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "",
        xy=(0.5, 0.1),
        xytext=(0.5, 1.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "linewidth": 2.0, "color": "black", "mutation_scale": 28},
    )
    ax.axis("off")


def _plot_sed_box(ax: plt.Axes) -> None:
    box = patches.FancyBboxPatch(
        (0.03, 0.14),
        0.94,
        0.72,
        boxstyle="round,pad=0.00,rounding_size=0.08",
        linewidth=1.5,
        edgecolor="#808080",
        facecolor=BOX_FACE_COLOR,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(0.5, 0.5, "SED System", ha="center", va="center", fontsize=20)
    ax.axis("off")


def _plot_annotation_matrix(ax: plt.Axes, matrix: np.ndarray, classes: list[str], total_seconds: int) -> None:
    rows, cols = matrix.shape
    x_edges = np.arange(cols + 1)
    y_edges = np.arange(rows + 1)
    display_matrix = matrix.copy()
    cmap_colors = [INACTIVE_CELL_COLOR, ACTIVE_CELL_COLOR]

    if ALTERNATIVE_PREDICTION_MATRIX:
        display_matrix = display_matrix.astype(int)
        cmap_colors = [INACTIVE_CELL_COLOR, TRUE_POSITIVE_CELL_COLOR, FALSE_POSITIVE_CELL_COLOR]
        false_positive_row_index = min(1, rows - 1)
        false_positive_col_index = 13
        if 0 <= false_positive_row_index < rows and 0 <= false_positive_col_index < cols:
            display_matrix[false_positive_row_index, false_positive_col_index] = 2

    ax.pcolormesh(
        x_edges,
        y_edges,
        display_matrix,
        shading="flat",
        cmap=ListedColormap(cmap_colors),
        vmin=0,
        vmax=len(cmap_colors) - 1,
        edgecolors="#BDBDBD",
        linewidth=1.0,
    )
    ax.set_xlim(0, total_seconds)
    ax.set_ylim(rows, 0)
    ax.set_yticks(np.arange(rows) + 0.5, labels=classes)
    ax.set_xticks(np.arange(total_seconds) + 0.5, labels=[str(second) for second in range(total_seconds)])
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16, length=0)
    ax.set_xlabel("Time (s)", fontsize=16)

    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    dataset = MLPC2026SoundEventDataset(DATASET_ROOT, "train", load_audio=True)
    sample = _select_sample(dataset)
    waveform = sample["waveform"]
    sample_rate = sample["sample_rate"]
    if waveform is None or sample_rate is None:
        raise RuntimeError("Selected sample did not include audio data.")

    waveform_np = waveform.mean(dim=0).numpy()
    clip_duration_seconds = waveform_np.shape[0] / float(sample_rate)
    annotations = _prepare_annotations(sample)
    matrix, classes, total_seconds = _build_annotation_matrix(annotations, clip_duration_seconds)

    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=[3.2, 0.6, 1.7, 0.6, 1.0])

    waveform_ax = fig.add_subplot(gs[0, 0])
    arrow_top_ax = fig.add_subplot(gs[1, 0])
    box_ax = fig.add_subplot(gs[2, 0])
    arrow_bottom_ax = fig.add_subplot(gs[3, 0])
    matrix_ax = fig.add_subplot(gs[4, 0])

    _plot_waveform(waveform_ax, waveform_np, sample_rate, total_seconds)
    _plot_arrow(arrow_top_ax)
    _plot_sed_box(box_ax)
    _plot_arrow(arrow_bottom_ax)
    _plot_annotation_matrix(matrix_ax, matrix, classes, total_seconds)

    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUTPUT_PDF.resolve()}")
    print(f"Selected file: {sample['filename']}")


if __name__ == "__main__":
    main()
