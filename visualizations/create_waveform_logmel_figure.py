from __future__ import annotations

import os
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
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
OUTPUT_PDF = Path(__file__).with_name("waveform_logmel_figure.pdf")
INCLUDE_ANNOTATION_MATRIX = False
INCLUDE_SPECTROGRAM = True

SAMPLE_RATE = 44_100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
TOP_DB = 80.0
WAVEFORM_COLOR = "#0078aa"
INACTIVE_CELL_COLOR = "#F5F5F5"
ACTIVE_CELL_COLOR = "#525252"
AXIS_LABEL_FONTSIZE = 15


def _select_sample(dataset: MLPC2026SoundEventDataset) -> dict[str, object]:
    if TARGET_FILENAME is not None:
        for sample in dataset:
            if sample["filename"] == TARGET_FILENAME:
                return sample
        raise ValueError(f"Could not find {TARGET_FILENAME!r} in the training split.")

    if not 0 <= TRAIN_SAMPLE_INDEX < len(dataset):
        raise IndexError(f"TRAIN_SAMPLE_INDEX={TRAIN_SAMPLE_INDEX} is out of range for {len(dataset)} training files.")
    return dataset[TRAIN_SAMPLE_INDEX]


def _prepare_waveform(sample: dict[str, object]) -> tuple[np.ndarray, int]:
    waveform = sample["waveform"]
    sample_rate = sample["sample_rate"]
    if waveform is None or sample_rate is None:
        raise RuntimeError("Selected sample did not include audio data.")

    waveform_np = waveform.mean(dim=0).numpy()
    if sample_rate != SAMPLE_RATE:
        waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        sample_rate = SAMPLE_RATE

    return waveform_np.astype(np.float32, copy=False), int(sample_rate)


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
    classes = annotations.groupby("annotation", sort=False)["onset"].min().sort_values().index.tolist()
    total_seconds = max(1, int(np.ceil(clip_duration_seconds)))
    matrix = np.zeros((len(classes), total_seconds), dtype=int)

    for row_index, annotation in enumerate(classes):
        class_rows = annotations.loc[annotations["annotation"] == annotation]
        for row in class_rows.itertuples(index=False):
            start_second = max(0, int(np.floor(float(row.onset))))
            end_second = min(total_seconds, int(np.ceil(float(row.offset))))
            if end_second > start_second:
                matrix[row_index, start_second:end_second] = 1

    return matrix, classes, total_seconds


def _compute_logmel(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=0.0,
        fmax=sample_rate / 2,
        power=2.0,
        center=True,
        norm="slaney",
        htk=False,
    )
    return librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)


def main() -> None:
    dataset = MLPC2026SoundEventDataset(DATASET_ROOT, "train", load_audio=True)
    sample = _select_sample(dataset)
    waveform, sample_rate = _prepare_waveform(sample)
    duration_seconds = waveform.shape[0] / float(sample_rate)
    time_axis = np.arange(waveform.shape[0], dtype=np.float64) / float(sample_rate)
    total_seconds = max(1, int(np.ceil(duration_seconds)))
    label_matrix: np.ndarray | None = None
    class_names: list[str] = []
    logmel: np.ndarray | None = None
    if INCLUDE_SPECTROGRAM:
        logmel = _compute_logmel(waveform, sample_rate)

    fig = plt.figure(figsize=(8, 3.5), constrained_layout=True)
    if INCLUDE_ANNOTATION_MATRIX:
        annotations = _prepare_annotations(sample)
        label_matrix, class_names, total_seconds = _build_annotation_matrix(annotations, duration_seconds)
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.8, 0.55])
        waveform_ax = fig.add_subplot(gs[0, 0])
        spec_ax = fig.add_subplot(gs[1, 0], sharex=waveform_ax)
        matrix_ax = fig.add_subplot(gs[2, 0], sharex=waveform_ax)
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.8])
        waveform_ax = fig.add_subplot(gs[0, 0])
        spec_ax = fig.add_subplot(gs[1, 0], sharex=waveform_ax)
        matrix_ax = None

    waveform_ax.plot(time_axis, waveform, color=WAVEFORM_COLOR, linewidth=0.8)
    waveform_ax.set_xlim(0.0, duration_seconds)
    waveform_ax.tick_params(axis="x", labelbottom=False)

    spec_ax.set_xlim(0.0, duration_seconds)
    if INCLUDE_SPECTROGRAM and logmel is not None:
        spec_ax.imshow(
            logmel,
            origin="lower",
            aspect="auto",
            extent=(0.0, duration_seconds, 0.0, float(N_MELS)),
            cmap="magma",
        )
        spec_ax.tick_params(axis="x", labelbottom=not INCLUDE_ANNOTATION_MATRIX)
    else:
        spec_ax.set_facecolor("white")
        spec_ax.set_xticks([])
        spec_ax.set_yticks([])
        spec_ax.tick_params(axis="x", bottom=False, labelbottom=False)
        for spine in spec_ax.spines.values():
            spine.set_visible(False)
    if not INCLUDE_ANNOTATION_MATRIX and INCLUDE_SPECTROGRAM:
        spec_ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)

    if INCLUDE_ANNOTATION_MATRIX and matrix_ax is not None:
        matrix_ax.set_xlim(0.0, duration_seconds)
        if INCLUDE_SPECTROGRAM and label_matrix is not None:
            x_edges = np.arange(total_seconds + 1, dtype=np.float64)
            y_edges = np.arange(len(class_names) + 1, dtype=np.float64)
            matrix_ax.pcolormesh(
                x_edges,
                y_edges,
                label_matrix,
                shading="flat",
                cmap=ListedColormap([INACTIVE_CELL_COLOR, ACTIVE_CELL_COLOR]),
                vmin=0,
                vmax=1,
                edgecolors="#BDBDBD",
                linewidth=0.8,
            )
            matrix_ax.set_ylim(len(class_names), 0)
            matrix_ax.set_yticks(np.arange(len(class_names)) + 0.5, labels=class_names)
            matrix_ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
            matrix_ax.tick_params(axis="y", length=0, labelsize=15)
            for spine in matrix_ax.spines.values():
                spine.set_visible(False)
        else:
            matrix_ax.set_facecolor("white")
            matrix_ax.set_xticks([])
            matrix_ax.set_yticks([])
            matrix_ax.tick_params(axis="x", bottom=False, labelbottom=False)
            for spine in matrix_ax.spines.values():
                spine.set_visible(False)

    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUTPUT_PDF.resolve()}")
    print(f"Selected file: {sample['filename']}")


if __name__ == "__main__":
    main()
