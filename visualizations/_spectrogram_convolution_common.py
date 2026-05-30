from __future__ import annotations

import os
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from domestic_sed.dataset import MLPC2026SoundEventDataset

DATASET_ROOT = Path(os.environ.get("DOMESTIC_SED_ROOT", "~/data/mlpc2026_dataset/MLPC2026_challenge_dataset_raw")).expanduser()
TRAIN_SAMPLE_INDEX = 11
TARGET_FILENAME: str | None = None

SAMPLE_RATE = 44_100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
TOP_DB = 80.0
AXIS_LABEL_FONTSIZE = 15
KERNEL_RADIUS = 2
TRANSPOSE_KERNEL = True
SIGNED_KERNEL = np.array(
    [
        [0.0, 0.5, 1.0, 0.5, 0.0],
        [0.5, 1.0, 2.0, 1.0, 0.5],
        [1.0, 2.0, 3.0, 2.0, 1.0],
        [0.5, 1.0, 2.0, 1.0, 0.5],
        [0.0, 0.5, 1.0, 0.5, 0.0],
    ],
    dtype=np.float32,
) / 3.0


def select_sample(dataset: MLPC2026SoundEventDataset) -> dict[str, object]:
    if TARGET_FILENAME is not None:
        for sample in dataset:
            if sample["filename"] == TARGET_FILENAME:
                return sample
        raise ValueError(f"Could not find {TARGET_FILENAME!r} in the training split.")

    if not 0 <= TRAIN_SAMPLE_INDEX < len(dataset):
        raise IndexError(f"TRAIN_SAMPLE_INDEX={TRAIN_SAMPLE_INDEX} is out of range for {len(dataset)} training files.")
    return dataset[TRAIN_SAMPLE_INDEX]


def prepare_waveform(sample: dict[str, object]) -> tuple[np.ndarray, int]:
    waveform = sample["waveform"]
    sample_rate = sample["sample_rate"]
    if waveform is None or sample_rate is None:
        raise RuntimeError("Selected sample did not include audio data.")

    waveform_np = waveform.mean(dim=0).numpy()
    if sample_rate != SAMPLE_RATE:
        waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        sample_rate = SAMPLE_RATE

    return waveform_np.astype(np.float32, copy=False), int(sample_rate)


def compute_logmel(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
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


def compute_convolution(logmel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return convolve2d(logmel, kernel, mode="valid")


def apply_stride(feature_map: np.ndarray, stride: int) -> np.ndarray:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return feature_map[::stride, ::stride]


def display_kernel() -> np.ndarray:
    return SIGNED_KERNEL.T if TRANSPOSE_KERNEL else SIGNED_KERNEL


def load_example() -> tuple[dict[str, object], np.ndarray, float, np.ndarray, np.ndarray]:
    dataset = MLPC2026SoundEventDataset(DATASET_ROOT, "train", load_audio=True)
    sample = select_sample(dataset)
    waveform, sample_rate = prepare_waveform(sample)
    duration_seconds = waveform.shape[0] / float(sample_rate)
    logmel = compute_logmel(waveform, sample_rate)
    kernel = display_kernel()
    signed_convolution = compute_convolution(logmel, kernel)
    return sample, kernel, duration_seconds, logmel, signed_convolution


def valid_extent(logmel: np.ndarray, duration_seconds: float) -> tuple[float, float, float, float]:
    num_bins, num_frames = logmel.shape
    seconds_per_frame = duration_seconds / float(num_frames)
    return (
        KERNEL_RADIUS * seconds_per_frame,
        duration_seconds - KERNEL_RADIUS * seconds_per_frame,
        float(KERNEL_RADIUS),
        float(num_bins - KERNEL_RADIUS),
    )


def strided_valid_extent(logmel: np.ndarray, duration_seconds: float, stride: int) -> tuple[float, float, float, float]:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    num_bins, num_frames = logmel.shape
    valid_rows = num_bins - 2 * KERNEL_RADIUS
    valid_cols = num_frames - 2 * KERNEL_RADIUS
    row_indices = np.arange(0, valid_rows, stride)
    col_indices = np.arange(0, valid_cols, stride)
    seconds_per_frame = duration_seconds / float(num_frames)

    return (
        (KERNEL_RADIUS + float(col_indices[0])) * seconds_per_frame,
        (KERNEL_RADIUS + float(col_indices[-1]) + 1.0) * seconds_per_frame,
        float(KERNEL_RADIUS + row_indices[0]),
        float(KERNEL_RADIUS + row_indices[-1] + 1),
    )


def hide_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords="axes fraction",
        textcoords="axes fraction",
        annotation_clip=False,
        arrowprops={"arrowstyle": "-|>", "linewidth": 2.0, "color": "black", "mutation_scale": 24},
    )
