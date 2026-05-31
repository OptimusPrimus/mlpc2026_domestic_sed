from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpectrogramAugmentationConfig:
    filter_augment_p: float = 0.0
    filter_db_range: float = 6.0
    filter_n_band_min: int = 3
    filter_n_band_max: int = 6
    filter_min_bw: int = 6
    waveform_noise_max_level: float = 0.0


def add_waveform_noise(
    waveforms: torch.Tensor,
    *,
    waveform_lengths: torch.Tensor,
    max_noise_level: float,
) -> torch.Tensor:
    if max_noise_level <= 0.0:
        return waveforms

    noisy_waveforms = waveforms.clone()
    noise_levels = torch.rand(
        waveforms.shape[0],
        1,
        device=waveforms.device,
        dtype=waveforms.dtype,
    ) * max_noise_level
    noise = torch.randn_like(waveforms) * noise_levels
    for batch_index in range(noisy_waveforms.shape[0]):
        valid_samples = int(waveform_lengths[batch_index].item())
        if valid_samples <= 0:
            continue
        noisy_waveforms[batch_index, :valid_samples] = (
            noisy_waveforms[batch_index, :valid_samples] + noise[batch_index, :valid_samples]
        )
    return noisy_waveforms


def _sample_band_boundaries(
    *,
    n_freq_bin: int,
    n_band: int,
    min_bw: int,
    device: torch.device,
) -> torch.Tensor | None:
    adjusted_min_bw = min_bw
    while n_freq_bin - n_band * adjusted_min_bw + 1 < 0 and adjusted_min_bw > 1:
        adjusted_min_bw -= 1

    if n_freq_bin - n_band * adjusted_min_bw + 1 < 0:
        return None

    band_boundaries = torch.sort(
        torch.randint(
            0,
            n_freq_bin - n_band * adjusted_min_bw + 1,
            (n_band - 1,),
            device=device,
        )
    )[0] + torch.arange(1, n_band, device=device) * adjusted_min_bw
    return torch.cat(
        (
            torch.tensor([0], device=device),
            band_boundaries,
            torch.tensor([n_freq_bin], device=device),
        )
    )


def filter_augmentation(
    features: torch.Tensor,
    *,
    filter_db_range: float = 6.0,
    filter_n_band_min: int = 3,
    filter_n_band_max: int = 6,
    filter_min_bw: int = 6,
) -> torch.Tensor:
    squeeze_channel = False
    if features.ndim == 3:
        features = features.unsqueeze(1)
        squeeze_channel = True
    elif features.ndim != 4:
        raise ValueError(
            "Expected features of shape (batch, mel_bins, frames) or "
            "(batch, channels, mel_bins, frames)"
        )

    batch_size, _, n_freq_bin, _ = features.shape
    if filter_db_range < 0.0:
        return features.squeeze(1) if squeeze_channel else features
    if filter_n_band_min <= 0 or filter_n_band_max < filter_n_band_min:
        return features.squeeze(1) if squeeze_channel else features

    n_band = int(
        torch.randint(
            low=filter_n_band_min,
            high=filter_n_band_max + 1,
            size=(1,),
            device=features.device,
        ).item()
    )
    if n_band <= 1:
        return features.squeeze(1) if squeeze_channel else features

    band_boundaries = _sample_band_boundaries(
        n_freq_bin=n_freq_bin,
        n_band=n_band,
        min_bw=filter_min_bw,
        device=features.device,
    )
    if band_boundaries is None:
        return features.squeeze(1) if squeeze_channel else features

    band_factors = (
        torch.rand((batch_size, n_band + 1), device=features.device, dtype=features.dtype)
        * (2.0 * filter_db_range)
        - filter_db_range
    )
    freq_filt = torch.ones((batch_size, n_freq_bin, 1), device=features.device, dtype=features.dtype)
    for band_index in range(n_band):
        band_start = int(band_boundaries[band_index].item())
        band_end = int(band_boundaries[band_index + 1].item())
        if band_end <= band_start:
            continue
        for batch_index in range(batch_size):
            freq_filt[batch_index, band_start:band_end, :] = torch.linspace(
                band_factors[batch_index, band_index],
                band_factors[batch_index, band_index + 1],
                band_end - band_start,
                device=features.device,
                dtype=features.dtype,
            ).unsqueeze(-1)
    freq_filt = 10 ** (freq_filt / 20)
    augmented = features * freq_filt.unsqueeze(1)
    return augmented.squeeze(1) if squeeze_channel else augmented
