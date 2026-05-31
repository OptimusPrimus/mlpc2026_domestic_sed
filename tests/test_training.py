from types import SimpleNamespace

import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from domestic_sed.augmentations import SpectrogramAugmentationConfig, filter_augmentation
from domestic_sed.training import SoundEventLightningModule, _resolve_seed, build_arg_parser, build_callbacks


def test_build_arg_parser_accepts_lr_linear_decay_start_epoch() -> None:
    args = build_arg_parser().parse_args(["--lr-linear-decay-start-epoch", "5"])

    assert args.lr_linear_decay_start_epoch == 5


def test_build_arg_parser_defaults_lr_warmup_epochs_to_zero() -> None:
    args = build_arg_parser().parse_args([])

    assert args.lr_warmup_epochs == 0


def test_build_arg_parser_accepts_early_stopping_patience() -> None:
    args = build_arg_parser().parse_args(["--early-stopping-patience", "10"])

    assert args.early_stopping_patience == 10


def test_build_arg_parser_defaults_to_random_seed() -> None:
    args = build_arg_parser().parse_args([])

    assert args.seed is None


def test_resolve_seed_keeps_explicit_seed() -> None:
    assert _resolve_seed(1234) == 1234


def test_build_callbacks_includes_learning_rate_monitor() -> None:
    callbacks = build_callbacks(early_stopping_patience=None)

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], LearningRateMonitor)


def test_build_callbacks_adds_early_stopping_when_configured() -> None:
    callbacks = build_callbacks(early_stopping_patience=10)

    assert len(callbacks) == 2
    assert isinstance(callbacks[0], LearningRateMonitor)
    assert isinstance(callbacks[1], EarlyStopping)
    assert callbacks[1].patience == 10


def test_configure_optimizers_adds_stepwise_warmup_and_decay_scheduler() -> None:
    model = SoundEventLightningModule(
        class_to_index={f"class_{index}": index for index in range(15)},
        learning_rate=1e-3,
        weight_decay=0.0,
        lr_warmup_epochs=1,
        lr_linear_decay_start_epoch=3,
        max_epochs=6,
    )
    model._trainer = SimpleNamespace(estimated_stepping_batches=60)

    configured = model.configure_optimizers()

    assert isinstance(configured, dict)
    scheduler_config = configured["lr_scheduler"]
    scheduler = scheduler_config["scheduler"]
    optimizer = configured["optimizer"]
    lambda_fn = scheduler.lr_lambdas[0]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert scheduler_config["interval"] == "step"
    assert lambda_fn(0) == 0.0
    assert lambda_fn(5) == 0.5
    assert lambda_fn(10) == 1.0
    assert lambda_fn(20) == 1.0
    assert lambda_fn(30) == 0.75
    assert lambda_fn(40) == 0.5
    assert lambda_fn(50) == 0.25
    assert lambda_fn(60) == 0.0


def test_build_arg_parser_accepts_new_augmentation_arguments() -> None:
    args = build_arg_parser().parse_args(
        [
            "--augmentation-filter-p",
            "0.4",
            "--augmentation-filter-db-range",
            "4.0",
            "--augmentation-filter-n-band-min",
            "2",
            "--augmentation-filter-n-band-max",
            "5",
            "--augmentation-filter-min-bw",
            "7",
            "--augmentation-waveform-noise-max-level",
            "0.08",
        ]
    )

    assert args.augmentation_filter_p == 0.4
    assert args.augmentation_filter_db_range == 4.0
    assert args.augmentation_filter_n_band_min == 2
    assert args.augmentation_filter_n_band_max == 5
    assert args.augmentation_filter_min_bw == 7
    assert args.augmentation_waveform_noise_max_level == 0.08


def test_waveform_noise_augmentation_is_disabled_by_default() -> None:
    model = SoundEventLightningModule(
        class_to_index={f"class_{index}": index for index in range(15)},
        learning_rate=1e-3,
    )
    waveform = torch.ones(2, 32)

    augmented = model._apply_waveform_augmentations(
        waveform,
        audio_num_samples=torch.tensor([32, 32], dtype=torch.long),
    )

    assert torch.equal(augmented, waveform)


def test_filter_augmentation_accepts_three_dimensional_spectrograms() -> None:
    spectrogram = torch.ones(2, 128, 64)

    augmented = filter_augmentation(
        spectrogram,
        filter_db_range=3.0,
        filter_n_band_min=2,
        filter_n_band_max=4,
        filter_min_bw=4,
    )

    assert augmented.shape == spectrogram.shape


def test_training_filter_augmentation_is_applied_per_spectrogram(monkeypatch) -> None:
    model = SoundEventLightningModule(
        class_to_index={f"class_{index}": index for index in range(2)},
        learning_rate=1e-3,
        augmentation_config=SpectrogramAugmentationConfig(
            filter_augment_p=0.5,
            filter_db_range=3.0,
            filter_n_band_min=2,
            filter_n_band_max=4,
            filter_min_bw=4,
        ),
    )
    spectrogram = torch.zeros(3, 128, 16)
    targets = torch.zeros(3, 2, 4)

    monkeypatch.setattr("domestic_sed.training.torch.rand", lambda *args, **kwargs: torch.tensor([0.2, 0.7, 0.1]))

    def fake_filter_augmentation(features: torch.Tensor, **_kwargs) -> torch.Tensor:
        return features + 1.0

    monkeypatch.setattr("domestic_sed.training.filter_augmentation", fake_filter_augmentation)

    augmented_spectrogram, augmented_targets = model._apply_training_augmentations(spectrogram, targets)

    assert torch.equal(augmented_targets, targets)
    assert torch.all(augmented_spectrogram[0] == 1.0)
    assert torch.all(augmented_spectrogram[1] == 0.0)
    assert torch.all(augmented_spectrogram[2] == 1.0)


def test_on_validation_epoch_end_logs_macro_and_per_class_map(monkeypatch) -> None:
    model = SoundEventLightningModule(
        class_to_index={"class_a": 0, "class_b": 1},
        learning_rate=1e-3,
    )
    model.validation_predictions = [
        {
            "filename": "sample.wav",
            "annotations": pd.DataFrame(
                [
                    {
                        "filename": "sample.wav",
                        "annotation": "class_a",
                        "onset": 0.0,
                        "offset": 0.5,
                    }
                ]
            ),
        }
    ]

    logged_metrics: list[tuple[str, float, dict[str, object]]] = []

    def fake_prediction_segments_to_frame(_prediction):
        index = pd.MultiIndex.from_tuples(
            [("sample.wav", 0.0)],
            names=["filename", "segment_start"],
        )
        return pd.DataFrame({"class_a": [1], "class_b": [0]}, index=index)

    def fake_calculate_map_score(_ground_truth_segments, _prediction_segments):
        return 0.7, pd.DataFrame(
            [
                {"annotation": "class_a", "precision": 1.0, "recall": 1.0, "f1": 1.0, "map": 0.8},
                {"annotation": "class_b", "precision": 0.0, "recall": 0.0, "f1": 0.0, "map": 0.6},
            ]
        )

    def fake_log(name: str, value: float, **kwargs) -> None:
        logged_metrics.append((name, value, kwargs))

    monkeypatch.setattr(model, "_prediction_segments_to_frame", fake_prediction_segments_to_frame)
    monkeypatch.setattr("domestic_sed.training.calculate_map_score", fake_calculate_map_score)
    monkeypatch.setattr(model, "log", fake_log)

    model.on_validation_epoch_end()

    assert logged_metrics == [
        ("val/map", 0.7, {"prog_bar": True, "on_step": False, "on_epoch": True, "sync_dist": False}),
        ("val_class/map_class_a", 0.8, {"prog_bar": False, "on_step": False, "on_epoch": True, "sync_dist": False}),
        ("val_class/map_class_b", 0.6, {"prog_bar": False, "on_step": False, "on_epoch": True, "sync_dist": False}),
    ]


def test_prediction_segments_to_frame_accepts_custom_segment_duration() -> None:
    model = SoundEventLightningModule(
        class_to_index={"class_a": 0, "class_b": 1},
        learning_rate=1e-3,
        sample_rate=4,
    )
    prediction = {
        "filename": "sample.wav",
        "audio_num_samples": 8,
        "valid_output_frames": 4,
        "probabilities": torch.tensor(
            [
                [0.1, 0.2],
                [0.8, 0.4],
                [0.3, 0.9],
                [0.7, 0.5],
            ]
        ),
    }

    segment_frame = model._prediction_segments_to_frame(
        prediction,
        segment_duration_seconds=0.5,
    )

    expected_index = pd.MultiIndex.from_tuples(
        [
            ("sample.wav", 0.0),
            ("sample.wav", 0.5),
            ("sample.wav", 1.0),
            ("sample.wav", 1.5),
        ],
        names=["filename", "segment_start"],
    )
    expected = pd.DataFrame(
        {
            "class_a": [0.1, 0.8, 0.3, 0.7],
            "class_b": [0.2, 0.4, 0.9, 0.5],
        },
        index=expected_index,
    )

    pd.testing.assert_frame_equal(segment_frame, expected)
