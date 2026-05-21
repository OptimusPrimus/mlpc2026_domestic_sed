from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import scipy.io.wavfile
import torch
from torch.utils.data import Dataset

SplitName = str

_SPLIT_ALIASES: dict[SplitName, SplitName] = {
    "train": "train",
    "validation": "validation",
    "val": "validation",
    "test": "test",
}


def _load_wav_audio(audio_path: Path) -> tuple[torch.Tensor, int]:
    sample_rate, waveform_np = scipy.io.wavfile.read(audio_path)
    waveform = torch.from_numpy(waveform_np)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(1)

    if waveform.dtype == torch.uint8:
        waveform = (waveform.to(torch.float32) - 128.0) / 128.0
    elif waveform.dtype == torch.int16:
        waveform = waveform.to(torch.float32) / 32768.0
    elif waveform.dtype == torch.int32:
        waveform = waveform.to(torch.float32) / 2147483648.0
    elif waveform.dtype == torch.int64:
        waveform = waveform.to(torch.float32) / 9223372036854775808.0
    else:
        waveform = waveform.to(torch.float32)

    return waveform.transpose(0, 1).contiguous(), int(sample_rate)


class MLPC2026SoundEventDataset(Dataset[dict[str, Any]]):
    """PyTorch dataset for the MLPC 2026 sound event detection dataset.

    The dataset root is expected to contain `train`, `validation`, and `test`
    directories. Each split contains an `audio/` directory with `.wav` files.
    Annotated splits additionally contain `metadata.csv` and `annotations.csv`.

    Samples are returned as dictionaries with these keys:
    - `split`: normalized split name
    - `filename`: audio filename
    - `audio_path`: absolute path to the `.wav` file
    - `waveform`: audio tensor or `None`
    - `sample_rate`: sample rate or `None`
    - `metadata`: single-row pandas DataFrame for the file
    - `annotations`: pandas DataFrame with zero or more rows for the file
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: SplitName,
        *,
        load_audio: bool = False,
        audio_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.split = self._normalize_split(split)
        self.split_dir = self.root_dir / self.split
        self.audio_dir = self.split_dir / "audio"
        self.load_audio = load_audio
        self.audio_transform = audio_transform

        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

        self.audio_files = sorted(self.audio_dir.glob("*.wav"))
        self.filenames = [path.name for path in self.audio_files]
        self._audio_paths_by_name = {path.name: path for path in self.audio_files}

        metadata_path = self.split_dir / "metadata.csv"
        annotations_path = self.split_dir / "annotations.csv"

        if metadata_path.is_file():
            self.metadata = pd.read_csv(metadata_path)
            self._validate_metadata()
            self._metadata_by_filename = {
                filename: group.reset_index(drop=True)
                for filename, group in self.metadata.groupby("filename", sort=False)
            }
        else:
            self.metadata = pd.DataFrame(columns=["filename"])
            self._metadata_by_filename = {}

        if annotations_path.is_file():
            self.annotations = pd.read_csv(annotations_path)
            self._validate_annotations()
            self._annotations_by_filename = {
                filename: group.reset_index(drop=True)
                for filename, group in self.annotations.groupby("filename", sort=False)
            }
        else:
            self.annotations = pd.DataFrame(
                columns=["filename", "annotator_id", "annotation", "onset", "offset", "is_own_recording"]
            )
            self._annotations_by_filename = {}

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        audio_path = self.audio_files[index]
        filename = audio_path.name

        waveform: torch.Tensor | None = None
        sample_rate: int | None = None
        if self.load_audio:
            waveform, sample_rate = _load_wav_audio(audio_path)
            if self.audio_transform is not None:
                waveform = self.audio_transform(waveform)

        metadata = self._metadata_by_filename.get(filename, pd.DataFrame(columns=self.metadata.columns)).copy()
        annotations = self._annotations_by_filename.get(
            filename,
            pd.DataFrame(columns=self.annotations.columns),
        ).copy()

        return {
            "split": self.split,
            "filename": filename,
            "audio_path": audio_path,
            "waveform": waveform,
            "sample_rate": sample_rate,
            "metadata": metadata,
            "annotations": annotations,
        }

    @staticmethod
    def _normalize_split(split: SplitName) -> SplitName:
        normalized = _SPLIT_ALIASES.get(split.lower())
        if normalized is None:
            valid = ", ".join(sorted(_SPLIT_ALIASES))
            raise ValueError(f"Unsupported split '{split}'. Expected one of: {valid}")
        return normalized

    def _validate_metadata(self) -> None:
        if "filename" not in self.metadata.columns:
            raise ValueError(f"metadata.csv must contain a 'filename' column: {self.split_dir / 'metadata.csv'}")

        if self.metadata["filename"].duplicated().any():
            duplicates = self.metadata.loc[self.metadata["filename"].duplicated(), "filename"].tolist()
            raise ValueError(f"metadata.csv contains duplicate filenames: {duplicates}")

    def _validate_annotations(self) -> None:
        required_columns = {
            "filename",
            "annotator_id",
            "annotation",
            "onset",
            "offset",
            "is_own_recording",
        }
        missing = required_columns.difference(self.annotations.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(
                f"annotations.csv is missing required columns ({missing_list}): {self.split_dir / 'annotations.csv'}"
            )
