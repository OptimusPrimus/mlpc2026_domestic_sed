from __future__ import annotations

import argparse
import secrets
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import torch
import torchaudio
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader

from domestic_sed.augmentations import (
    RandomResizeCrop,
    SpectrogramAugmentationConfig,
    filter_augmentation,
    frame_shift,
    mixstyle,
    mixup,
    time_mask,
)
from domestic_sed.architectures import CRNN, CRNNBlockConfig, build_default_crnn_blocks
from domestic_sed.dataset import MLPC2026SoundEventDataset
from domestic_sed.metrics.segment_based_metrics import (
    build_segment_frame_from_intervals,
    calculate_map_score,
)


DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_MEL_BINS = 128
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_WIN_LENGTH = 2048
DEFAULT_MAX_DURATION_SECONDS = 35.0
MIN_INPUT_SAMPLE_RATE = DEFAULT_SAMPLE_RATE


def _resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return secrets.randbelow(2**32)


def _default_crnn_blocks(
    *,
    p1: int = 5,
    p2: int = 5,
    depth: int = 12,
    channel_multiplier: int = 1,
) -> list[CRNNBlockConfig]:
    return build_default_crnn_blocks(
        p1=p1,
        p2=p2,
        depth=depth,
        channel_multiplier=channel_multiplier,
    )


def _waveform_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform with shape (channels, samples), got {tuple(waveform.shape)}")
    if waveform.shape[0] == 1:
        return waveform.squeeze(0)
    return waveform.mean(dim=0)


def _prepare_class_names(
    annotations: pd.DataFrame,
    provided_class_names: list[str] | None,
) -> list[str]:
    if provided_class_names:
        class_names = provided_class_names
    else:
        class_names = sorted(str(label) for label in annotations["annotation"].dropna().unique().tolist())

    if len(class_names) != 15:
        raise ValueError(f"Expected 15 classes, got {len(class_names)}: {class_names}")
    return class_names


class SoundEventDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str | Path,
        *,
        batch_size: int,
        num_workers: int,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root).expanduser().resolve()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.max_num_samples = int(round(sample_rate * max_duration_seconds))
        self.class_names = class_names
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}

        self.train_dataset: MLPC2026SoundEventDataset | None = None
        self.validation_dataset: MLPC2026SoundEventDataset | None = None
        self._class_to_index: dict[str, int] | None = None

    @property
    def class_to_index(self) -> dict[str, int]:
        if self._class_to_index is None:
            raise RuntimeError("DataModule.setup() must be called before accessing class_to_index.")
        return self._class_to_index

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.train_dataset is not None and self.validation_dataset is not None:
            return

        self.train_dataset = MLPC2026SoundEventDataset(self.data_root, split="train", load_audio=True)
        self.validation_dataset = MLPC2026SoundEventDataset(self.data_root, split="validation", load_audio=True)
        class_names = _prepare_class_names(self.train_dataset.annotations, self.class_names)
        self.class_names = class_names
        self._class_to_index = {class_name: index for index, class_name in enumerate(class_names)}

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting a dataloader.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_batch,
        )

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        if self.validation_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting a dataloader.")
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_batch,
        )

    def _collate_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        waveforms: list[torch.Tensor] = []
        audio_num_samples: list[int] = []

        for sample in batch:
            waveform = sample["waveform"]
            sample_rate = sample["sample_rate"]
            if waveform is None or sample_rate is None:
                raise ValueError("Dataset samples must include loaded audio for training.")

            mono_waveform = _waveform_to_mono(waveform).float()
            if sample_rate != self.sample_rate:
                mono_waveform = self._resample_waveform(mono_waveform, sample_rate)
            if mono_waveform.numel() > self.max_num_samples:
                mono_waveform = mono_waveform[: self.max_num_samples]
            audio_num_samples.append(mono_waveform.numel())

            padded = torch.zeros(self.max_num_samples, dtype=mono_waveform.dtype)
            padded[: mono_waveform.numel()] = mono_waveform
            waveforms.append(padded)

        return {
            "waveform": torch.stack(waveforms, dim=0),
            "audio_num_samples": torch.tensor(audio_num_samples, dtype=torch.long),
            "filenames": [sample["filename"] for sample in batch],
            "annotations": [sample["annotations"] for sample in batch],
        }

    def _resample_waveform(self, waveform: torch.Tensor, input_sample_rate: int) -> torch.Tensor:
        resampler = self.resamplers.get(input_sample_rate)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(orig_freq=input_sample_rate, new_freq=self.sample_rate)
            self.resamplers[input_sample_rate] = resampler
        return resampler(waveform.unsqueeze(0)).squeeze(0)


class SoundEventLightningModule(L.LightningModule):
    def __init__(
        self,
        *,
        class_to_index: dict[str, int],
        learning_rate: float,
        weight_decay: float = 0.0,
        lr_linear_decay_epochs: int = 0,
        max_epochs: int = 30,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_mels: int = DEFAULT_MEL_BINS,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        win_length: int = DEFAULT_WIN_LENGTH,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.2,
        architecture_p1: int = 5,
        architecture_p2: int = 5,
        architecture_depth: int = 12,
        architecture_base_multiplier: int = 1,
        augmentation_config: SpectrogramAugmentationConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_to_index"])
        self.class_to_index = class_to_index
        self.num_classes = len(class_to_index)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_linear_decay_epochs = lr_linear_decay_epochs
        self.max_epochs = max_epochs
        self.hop_length = hop_length
        self.augmentation_config = augmentation_config or SpectrogramAugmentationConfig()
        self.class_names_by_index = [class_name for class_name, _ in sorted(class_to_index.items(), key=lambda item: item[1])]
        self.validation_predictions: list[dict[str, Any]] = []

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0.0,
            f_max=sample_rate / 2,
            power=2.0,
            center=True,
            norm="slaney",
            mel_scale="slaney",
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)
        self.freq_warp = RandomResizeCrop(
            virtual_crop_scale=self.augmentation_config.freq_warp_virtual_crop_scale,
            freq_scale=self.augmentation_config.freq_warp_freq_scale,
            time_scale=self.augmentation_config.freq_warp_time_scale,
        )
        self.model = CRNN(
            conv_blocks=_default_crnn_blocks(
                p1=architecture_p1,
                p2=architecture_p2,
                depth=architecture_depth,
                channel_multiplier=architecture_base_multiplier,
            ),
            input_bins=n_mels,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            lstm_dropout=dropout,
            dropout=dropout,
            output_size=self.num_classes,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.model(self._waveform_to_features(waveform))

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        spectrogram = self._waveform_to_features(batch["waveform"])
        output_frames = self.model.output_shape(spectrogram.shape[-1])[0]
        targets = self._build_targets(
            annotations_batch=batch["annotations"],
            audio_num_samples=batch["audio_num_samples"],
            output_frames=output_frames,
            device=spectrogram.device,
            dtype=spectrogram.dtype,
        )
        loss_mask = self._build_loss_mask(
            audio_num_samples=batch["audio_num_samples"],
            output_frames=output_frames,
            device=spectrogram.device,
            dtype=spectrogram.dtype,
        )
        spectrogram, targets = self._apply_training_augmentations(spectrogram, targets)
        logits = self.model(spectrogram)
        loss = self._compute_loss_from_targets(
            logits=logits,
            targets=targets,
            loss_mask=loss_mask,
        )
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=logits.shape[0])
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        spectrogram = self._waveform_to_features(batch["waveform"])
        logits = self.model(spectrogram)
        targets = self._build_targets(
            annotations_batch=batch["annotations"],
            audio_num_samples=batch["audio_num_samples"],
            output_frames=logits.shape[1],
            device=logits.device,
            dtype=logits.dtype,
        )
        loss_mask = self._build_loss_mask(
            audio_num_samples=batch["audio_num_samples"],
            output_frames=logits.shape[1],
            device=logits.device,
            dtype=logits.dtype,
        )
        loss = self._compute_loss_from_targets(
            logits=logits,
            targets=targets,
            loss_mask=loss_mask,
        )
        probabilities = torch.sigmoid(logits)
        valid_output_frames = loss_mask[:, 0, :].sum(dim=1).to(dtype=torch.long)

        for sample_index, filename in enumerate(batch["filenames"]):
            self.validation_predictions.append(
                {
                    "filename": filename,
                    "annotations": batch["annotations"][sample_index],
                    "audio_num_samples": int(batch["audio_num_samples"][sample_index].item()),
                    "valid_output_frames": int(valid_output_frames[sample_index].item()),
                    "probabilities": probabilities[sample_index].detach().cpu(),
                    "targets": targets[sample_index].detach().cpu(),
                }
            )

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=logits.shape[0])
        return loss

    def on_validation_epoch_start(self) -> None:
        self.validation_predictions = []

    def on_validation_epoch_end(self) -> None:
        if not self.validation_predictions:
            return

        ground_truth_rows: list[pd.DataFrame] = []
        prediction_frames: list[pd.DataFrame] = []
        for prediction in self.validation_predictions:
            annotations = prediction["annotations"]
            if not annotations.empty:
                ground_truth_rows.append(annotations.loc[:, ["filename", "annotation", "onset", "offset"]].copy())
            prediction_frames.append(self._prediction_segments_to_frame(prediction))

        if ground_truth_rows:
            ground_truth_df = pd.concat(ground_truth_rows, ignore_index=True)
            ground_truth_segments = build_segment_frame_from_intervals(ground_truth_df, name="validation_ground_truth")
        else:
            ground_truth_segments = pd.DataFrame()

        if prediction_frames:
            prediction_segments = pd.concat(prediction_frames, axis=0).sort_index()
            prediction_segments = prediction_segments.groupby(level=["filename", "segment_start"]).max()
        else:
            prediction_segments = pd.DataFrame()

        macro_map, per_class_map = calculate_map_score(ground_truth_segments, prediction_segments)
        self.log("val/map", macro_map, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        for row in per_class_map.itertuples(index=False):
            self.log(
                f"val_class/map_{row.annotation}",
                row.map,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def _lr_decay_factor(self, epoch: int) -> float:
        if self.lr_linear_decay_epochs <= 0:
            return 1.0
        decay_epochs = min(self.lr_linear_decay_epochs, self.max_epochs)
        decay_start_epoch = self.max_epochs - decay_epochs
        if epoch < decay_start_epoch:
            return 1.0
        remaining_epochs = self.max_epochs - epoch
        return max(0.0, remaining_epochs / decay_epochs)

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.lr_linear_decay_epochs <= 0:
            return optimizer
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=self._lr_decay_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _waveform_to_features(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrogram = self.mel_transform(waveform)
        return self.amplitude_to_db(spectrogram)

    def _apply_training_augmentations(
        self,
        spectrogram: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.augmentation_config
        augmented_spectrogram = spectrogram
        augmented_targets = targets

        if config.frame_shift_range > 0.0:
            augmented_spectrogram, augmented_targets = frame_shift(
                augmented_spectrogram,
                augmented_targets,
                shift_range=config.frame_shift_range,
            )

        if config.mixup_p > 0.0 and torch.rand(1).item() < config.mixup_p:
            augmented_spectrogram, augmented_targets = mixup(
                augmented_spectrogram,
                targets=augmented_targets,
                alpha=config.mixup_alpha,
                beta=config.mixup_beta,
            )

        if config.mixstyle_p > 0.0 and torch.rand(1).item() < config.mixstyle_p:
            augmented_spectrogram = mixstyle(
                augmented_spectrogram,
                alpha=config.mixstyle_alpha,
            )

        if config.max_time_mask_size > 0.0:
            augmented_spectrogram, augmented_targets = time_mask(
                augmented_spectrogram,
                augmented_targets,
                max_mask_ratio=config.max_time_mask_size,
            )

        if config.filter_augment_p > 0.0 and torch.rand(1).item() < config.filter_augment_p:
            augmented_spectrogram = filter_augmentation(
                augmented_spectrogram,
                filter_db_range=(config.filter_db_range_min, config.filter_db_range_max),
                filter_bands=(config.filter_bands_min, config.filter_bands_max),
                filter_minimum_bandwidth=config.filter_minimum_bandwidth,
            )

        if config.freq_warp_p > 0.0 and torch.rand(1).item() < config.freq_warp_p:
            augmented_spectrogram = self.freq_warp(augmented_spectrogram.squeeze(1)).unsqueeze(1)

        return augmented_spectrogram, augmented_targets

    def _compute_loss_from_targets(
        self,
        *,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        per_frame_loss = nn.functional.binary_cross_entropy_with_logits(
            logits.transpose(1, 2),
            targets,
            reduction="none",
        )
        return (per_frame_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

    def _build_loss_mask(
        self,
        *,
        audio_num_samples: torch.Tensor,
        output_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = torch.zeros(
            audio_num_samples.shape[0],
            self.num_classes,
            output_frames,
            device=device,
            dtype=dtype,
        )
        for batch_index, num_samples in enumerate(audio_num_samples.tolist()):
            spectrogram_frames = self._num_spectrogram_frames(num_samples)
            valid_output_frames = self.model.output_shape(spectrogram_frames)[0]
            valid_output_frames = max(1, min(valid_output_frames, output_frames))
            mask[batch_index, :, :valid_output_frames] = 1.0
        return mask

    def _build_targets(
        self,
        *,
        annotations_batch: list[pd.DataFrame],
        audio_num_samples: torch.Tensor,
        output_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        targets = torch.zeros(
            len(annotations_batch),
            self.num_classes,
            output_frames,
            device=device,
            dtype=dtype,
        )

        for batch_index, annotations in enumerate(annotations_batch):
            clip_duration_seconds = audio_num_samples[batch_index].item() / self.hparams.sample_rate
            if clip_duration_seconds <= 0.0 or annotations.empty:
                continue

            for row in annotations.itertuples(index=False):
                label = str(row.annotation)
                class_index = self.class_to_index.get(label)
                if class_index is None:
                    continue

                onset = max(0.0, float(row.onset))
                offset = min(clip_duration_seconds, float(row.offset))
                if offset <= onset:
                    continue

                start_frame = int(torch.floor(torch.tensor(onset / clip_duration_seconds * output_frames)).item())
                end_frame = int(torch.ceil(torch.tensor(offset / clip_duration_seconds * output_frames)).item())
                start_frame = max(0, min(start_frame, output_frames - 1))
                end_frame = max(start_frame + 1, min(end_frame, output_frames))
                targets[batch_index, class_index, start_frame:end_frame] = 1.0

        return targets

    def _num_spectrogram_frames(self, num_samples: int) -> int:
        return 1 + max(0, num_samples // self.hparams.hop_length)

    def _prediction_segments_to_frame(
        self,
        prediction: dict[str, Any],
        segment_duration_seconds: float = 1.0,
    ) -> pd.DataFrame:
        clip_duration_seconds = prediction["audio_num_samples"] / self.hparams.sample_rate
        valid_output_frames = prediction["valid_output_frames"]
        probabilities: torch.Tensor = prediction["probabilities"][:valid_output_frames]

        if clip_duration_seconds <= 0.0 or valid_output_frames <= 0:
            empty_index = pd.MultiIndex.from_tuples([], names=["filename", "segment_start"])
            return pd.DataFrame(columns=self.class_names_by_index, index=empty_index)

        num_segments = max(1, int(torch.ceil(torch.tensor(clip_duration_seconds / segment_duration_seconds)).item()))
        rows: list[dict[str, float | str]] = []
        for segment_index in range(num_segments):
            segment_start = segment_index * segment_duration_seconds
            segment_end = min(segment_start + segment_duration_seconds, clip_duration_seconds)
            frame_start = int(segment_start / clip_duration_seconds * valid_output_frames)
            frame_end = int(torch.ceil(torch.tensor(segment_end / clip_duration_seconds * valid_output_frames)).item())
            frame_start = max(0, min(frame_start, valid_output_frames - 1))
            frame_end = max(frame_start + 1, min(frame_end, valid_output_frames))
            segment_scores = probabilities[frame_start:frame_end].max(dim=0).values

            row: dict[str, float | str] = {
                "filename": prediction["filename"],
                "segment_start": segment_start,
            }
            for class_index, class_name in enumerate(self.class_names_by_index):
                row[class_name] = float(segment_scores[class_index].item())
            rows.append(row)

        segment_frame = pd.DataFrame(rows).set_index(["filename", "segment_start"])
        return segment_frame.loc[:, self.class_names_by_index]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a CRNN sound event detection model.")
    parser.add_argument("--data-root", type=Path, help="Dataset root containing the train split.", default="/home/paul/data/mlpc2026_dataset/MLPC2026_challenge_dataset_raw/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--lr-linear-decay-epochs",
        type=int,
        default=0,
        help="Linearly decay the learning rate to zero over the last N epochs of training.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop training if val/map does not improve for this many validation epochs.",
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--max-duration-seconds", type=float, default=DEFAULT_MAX_DURATION_SECONDS)
    parser.add_argument("--n-mels", type=int, default=DEFAULT_MEL_BINS)
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--win-length", type=int, default=DEFAULT_WIN_LENGTH)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--lstm-num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--architecture-p1", type=int, default=5)
    parser.add_argument("--architecture-p2", type=int, default=5)
    parser.add_argument("--architecture-depth", type=int, default=12)
    parser.add_argument("--architecture-base-multiplier", type=int, default=2)
    parser.add_argument("--augmentation-frame-shift-range", type=float, default=0.0)
    parser.add_argument("--augmentation-mixup-p", type=float, default=0.0)
    parser.add_argument("--augmentation-mixup-alpha", type=float, default=0.2)
    parser.add_argument("--augmentation-mixup-beta", type=float, default=0.2)
    parser.add_argument("--augmentation-mixstyle-p", type=float, default=0.0)
    parser.add_argument("--augmentation-mixstyle-alpha", type=float, default=0.4)
    parser.add_argument("--augmentation-max-time-mask-size", type=float, default=0.0)
    parser.add_argument("--augmentation-filter-p", type=float, default=0.0)
    parser.add_argument("--augmentation-filter-db-min", type=float, default=-6.0)
    parser.add_argument("--augmentation-filter-db-max", type=float, default=6.0)
    parser.add_argument("--augmentation-filter-bands-min", type=int, default=3)
    parser.add_argument("--augmentation-filter-bands-max", type=int, default=6)
    parser.add_argument("--augmentation-filter-min-bandwidth", type=int, default=6)
    parser.add_argument("--augmentation-freq-warp-p", type=float, default=0.0)
    parser.add_argument("--augmentation-freq-warp-virtual-scale-freq", type=float, default=1.0)
    parser.add_argument("--augmentation-freq-warp-virtual-scale-time", type=float, default=1.5)
    parser.add_argument("--augmentation-freq-warp-freq-scale-min", type=float, default=1.0)
    parser.add_argument("--augmentation-freq-warp-freq-scale-max", type=float, default=1.0)
    parser.add_argument("--augmentation-freq-warp-time-scale-min", type=float, default=1.0)
    parser.add_argument("--augmentation-freq-warp-time-scale-max", type=float, default=1.0)
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help="Comma-separated list of the 15 class names. Defaults to labels found in train/annotations.csv.",
    )
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. If omitted, choose a random seed for this run.",
    )
    parser.add_argument("--wandb-project", type=str, default="domestic-sed")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-save-dir", type=Path, default=Path("wandb"))
    return parser


def build_callbacks(*, early_stopping_patience: int | None) -> list[L.Callback]:
    callbacks: list[L.Callback] = [LearningRateMonitor(logging_interval="epoch")]
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val/map",
                mode="max",
                patience=early_stopping_patience,
            )
        )
    return callbacks


def main() -> None:
    args = build_arg_parser().parse_args()
    args.seed = _resolve_seed(args.seed)
    L.seed_everything(args.seed, workers=True)

    initial_class_names = None
    if args.class_names:
        initial_class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]

    augmentation_config = SpectrogramAugmentationConfig(
        frame_shift_range=args.augmentation_frame_shift_range,
        mixup_p=args.augmentation_mixup_p,
        mixup_alpha=args.augmentation_mixup_alpha,
        mixup_beta=args.augmentation_mixup_beta,
        mixstyle_p=args.augmentation_mixstyle_p,
        mixstyle_alpha=args.augmentation_mixstyle_alpha,
        max_time_mask_size=args.augmentation_max_time_mask_size,
        filter_augment_p=args.augmentation_filter_p,
        filter_db_range_min=args.augmentation_filter_db_min,
        filter_db_range_max=args.augmentation_filter_db_max,
        filter_bands_min=args.augmentation_filter_bands_min,
        filter_bands_max=args.augmentation_filter_bands_max,
        filter_minimum_bandwidth=args.augmentation_filter_min_bandwidth,
        freq_warp_p=args.augmentation_freq_warp_p,
        freq_warp_virtual_crop_scale=(
            args.augmentation_freq_warp_virtual_scale_freq,
            args.augmentation_freq_warp_virtual_scale_time,
        ),
        freq_warp_freq_scale=(
            args.augmentation_freq_warp_freq_scale_min,
            args.augmentation_freq_warp_freq_scale_max,
        ),
        freq_warp_time_scale=(
            args.augmentation_freq_warp_time_scale_min,
            args.augmentation_freq_warp_time_scale_max,
        ),
    )

    datamodule = SoundEventDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        max_duration_seconds=args.max_duration_seconds,
        class_names=initial_class_names,
    )
    datamodule.setup("fit")

    model = SoundEventLightningModule(
        class_to_index=datamodule.class_to_index,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_linear_decay_epochs=args.lr_linear_decay_epochs,
        max_epochs=args.max_epochs,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        dropout=args.dropout,
        architecture_p1=args.architecture_p1,
        architecture_p2=args.architecture_p2,
        architecture_depth=args.architecture_depth,
        architecture_base_multiplier=args.architecture_base_multiplier,
        augmentation_config=augmentation_config,
    )

    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        save_dir=str(args.wandb_save_dir),
        log_model=False,
    )

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        deterministic=False,
        log_every_n_steps=10,
        logger=logger,
        callbacks=build_callbacks(early_stopping_patience=args.early_stopping_patience),
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
