from pathlib import Path

import pandas as pd
import pytest

from domestic_sed.dataset import MLPC2026SoundEventDataset


def test_train_split_returns_metadata_and_annotations(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    audio_dir = train_dir / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "000001.wav").touch()
    (audio_dir / "000002.wav").touch()

    metadata = pd.DataFrame(
        [
            {
                "filename": "000001.wav",
                "collector_id": "c1",
                "target_classes": "footsteps;keyboard_typing",
                "non_target_classes": "sigh",
                "recording_device": "phone",
                "device_placement": "mobile",
                "recording_environment": "office",
                "scene_description": "desc 1",
                "license": "CC0",
            },
            {
                "filename": "000002.wav",
                "collector_id": "c2",
                "target_classes": "door_open_close",
                "non_target_classes": "None",
                "recording_device": "phone",
                "device_placement": "static",
                "recording_environment": "living_room",
                "scene_description": "desc 2",
                "license": "CC0",
            },
        ]
    )
    metadata.to_csv(train_dir / "metadata.csv", index=False)

    annotations = pd.DataFrame(
        [
            {
                "filename": "000001.wav",
                "annotator_id": "a1",
                "annotation": "footsteps",
                "onset": 0.1,
                "offset": 0.9,
                "is_own_recording": True,
            },
            {
                "filename": "000001.wav",
                "annotator_id": "a2",
                "annotation": "keyboard_typing",
                "onset": 1.0,
                "offset": 1.5,
                "is_own_recording": False,
            },
        ]
    )
    annotations.to_csv(train_dir / "annotations.csv", index=False)

    dataset = MLPC2026SoundEventDataset(tmp_path, "train")

    assert len(dataset) == 2

    first_sample = dataset[0]
    assert first_sample["filename"] == "000001.wav"
    assert first_sample["audio_path"] == audio_dir / "000001.wav"
    assert first_sample["waveform"] is None
    assert first_sample["sample_rate"] is None

    pd.testing.assert_frame_equal(
        first_sample["metadata"].reset_index(drop=True),
        metadata.iloc[[0]].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        first_sample["annotations"].reset_index(drop=True),
        annotations.iloc[[0, 1]].reset_index(drop=True),
    )

    second_sample = dataset[1]
    assert second_sample["filename"] == "000002.wav"
    assert second_sample["annotations"].empty
    assert list(second_sample["annotations"].columns) == list(annotations.columns)


def test_test_split_works_without_csv_files(tmp_path: Path) -> None:
    test_dir = tmp_path / "test"
    audio_dir = test_dir / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "000010.wav").touch()

    dataset = MLPC2026SoundEventDataset(tmp_path, "test")
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["filename"] == "000010.wav"
    assert sample["metadata"].empty
    assert sample["annotations"].empty


def test_validation_alias_is_supported(tmp_path: Path) -> None:
    val_dir = tmp_path / "validation" / "audio"
    val_dir.mkdir(parents=True)
    (val_dir / "000020.wav").touch()

    dataset = MLPC2026SoundEventDataset(tmp_path, "val")
    assert dataset.split == "validation"


def test_duplicate_metadata_filenames_raise_error(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    audio_dir = train_dir / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "000001.wav").touch()

    metadata = pd.DataFrame(
        [
            {"filename": "000001.wav"},
            {"filename": "000001.wav"},
        ]
    )
    metadata.to_csv(train_dir / "metadata.csv", index=False)

    with pytest.raises(ValueError, match="duplicate filenames"):
        MLPC2026SoundEventDataset(tmp_path, "train")
