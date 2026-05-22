from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta


@dataclass(frozen=True)
class SpectrogramAugmentationConfig:
    frame_shift_range: float = 0.0
    mixup_p: float = 0.0
    mixup_alpha: float = 0.2
    mixup_beta: float = 0.2
    mixstyle_p: float = 0.0
    mixstyle_alpha: float = 0.4
    max_time_mask_size: float = 0.0
    filter_augment_p: float = 0.0
    filter_db_range_min: float = -6.0
    filter_db_range_max: float = 6.0
    filter_bands_min: int = 3
    filter_bands_max: int = 6
    filter_minimum_bandwidth: int = 6
    freq_warp_p: float = 0.0
    freq_warp_virtual_crop_scale: tuple[float, float] = (1.0, 1.5)
    freq_warp_freq_scale: tuple[float, float] = (1.0, 1.0)
    freq_warp_time_scale: tuple[float, float] = (1.0, 1.0)


def frame_shift(
    mels: torch.Tensor,
    labels: torch.Tensor,
    *,
    net_pooling: int = 1,
    shift_range: float = 0.125,
) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_mels = mels.clone()
    shifted_labels = labels.clone()
    _, _, _, frames = shifted_mels.shape
    abs_shift_mel = int(frames * shift_range)

    if abs_shift_mel <= 0:
        return shifted_mels, shifted_labels

    for bindx in range(shifted_mels.shape[0]):
        shift = int(random.gauss(0, abs_shift_mel))
        shifted_mels[bindx] = torch.roll(shifted_mels[bindx], shift, dims=-1)
        label_shift = -abs(shift) / net_pooling if shift < 0 else shift / net_pooling
        shifted_labels[bindx] = torch.roll(shifted_labels[bindx], round(label_shift), dims=-1)

    return shifted_mels, shifted_labels


def time_mask(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    net_pooling: int = 1,
    min_mask_ratio: float = 0.05,
    max_mask_ratio: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_features = features.clone()
    masked_labels = labels.clone()
    _, _, n_frame = masked_labels.shape

    if n_frame <= 1 or max_mask_ratio <= 0.0:
        return masked_features, masked_labels

    low = max(1, int(n_frame * min_mask_ratio))
    high = max(low + 1, int(n_frame * max_mask_ratio))
    high = min(high, n_frame)
    if low >= high:
        return masked_features, masked_labels

    t_width = torch.randint(low=low, high=high, size=(1,), device=masked_labels.device).item()
    max_start = n_frame - t_width
    if max_start <= 0:
        return masked_features, masked_labels

    t_low = torch.randint(low=0, high=max_start, size=(1,), device=masked_labels.device).item()
    masked_features[:, :, :, t_low * net_pooling : (t_low + t_width) * net_pooling] = 0
    masked_labels[:, :, t_low : t_low + t_width] = 0
    return masked_features, masked_labels


def mixup(
    data: torch.Tensor,
    *,
    targets: torch.Tensor,
    alpha: float = 0.2,
    beta: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        batch_size = data.size(0)
        c = np.random.beta(alpha, beta, size=batch_size)
        c = np.maximum(c, 1 - c)

        perm = torch.randperm(batch_size, device=data.device)
        cd = torch.tensor(c, dtype=data.dtype, device=data.device).view(batch_size, *([1] * (data.ndim - 1)))
        ct = torch.tensor(c, dtype=targets.dtype, device=targets.device).view(
            batch_size,
            *([1] * (targets.ndim - 1)),
        )
        mixed_data = cd * data + (1 - cd) * data[perm, :]
        mixed_target = torch.clamp(ct * targets + (1 - ct) * targets[perm, :], min=0, max=1)

    return mixed_data, mixed_target


def filt_aug_(
    features: torch.Tensor,
    *,
    db_range: tuple[float, float] = (-6, 6),
    n_band: tuple[int, int] = (3, 6),
    min_bw: int = 6,
) -> torch.Tensor:
    batch_size, _, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()
    if n_freq_band <= 1:
        return features

    adjusted_min_bw = min_bw
    while n_freq_bin - n_freq_band * adjusted_min_bw + 1 < 0 and adjusted_min_bw > 1:
        adjusted_min_bw -= 1

    if n_freq_bin - n_freq_band * adjusted_min_bw + 1 < 0:
        return features

    band_bndry_freqs = torch.sort(
        torch.randint(0, n_freq_bin - n_freq_band * adjusted_min_bw + 1, (n_freq_band - 1,))
    )[0] + torch.arange(1, n_freq_band) * adjusted_min_bw
    band_bndry_freqs = torch.cat(
        (
            torch.tensor([0]),
            band_bndry_freqs,
            torch.tensor([n_freq_bin]),
        )
    ).to(features.device)

    band_factors = (
        torch.rand((batch_size, n_freq_band + 1), device=features.device, dtype=features.dtype)
        * (db_range[1] - db_range[0])
        + db_range[0]
    )
    freq_filt = torch.ones((batch_size, n_freq_bin, 1), device=features.device, dtype=features.dtype)
    for i in range(n_freq_band):
        for j in range(batch_size):
            freq_filt[j, band_bndry_freqs[i] : band_bndry_freqs[i + 1], :] = torch.linspace(
                band_factors[j, i],
                band_factors[j, i + 1],
                band_bndry_freqs[i + 1] - band_bndry_freqs[i],
                device=features.device,
                dtype=features.dtype,
            ).unsqueeze(-1)
    freq_filt = 10 ** (freq_filt / 20)
    return features * freq_filt.unsqueeze(1)


def filter_augmentation(
    features: torch.Tensor,
    *,
    filter_db_range: tuple[float, float] = (-6, 6),
    filter_bands: tuple[int, int] = (3, 6),
    filter_minimum_bandwidth: int = 6,
) -> torch.Tensor:
    return filt_aug_(
        features,
        db_range=filter_db_range,
        n_band=filter_bands,
        min_bw=filter_minimum_bandwidth,
    )


def mixstyle(x: torch.Tensor, *, alpha: float = 0.4, eps: float = 1e-6) -> torch.Tensor:
    batch_size = x.size(0)
    f_mu = x.mean(dim=3, keepdim=True)
    f_var = x.var(dim=3, keepdim=True)

    f_sig = (f_var + eps).sqrt()
    f_mu, f_sig = f_mu.detach(), f_sig.detach()
    x_normed = (x - f_mu) / f_sig
    lmda = Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device, dtype=x.dtype)
    lmda = torch.max(lmda, 1 - lmda)
    perm = torch.randperm(batch_size, device=x.device)
    f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]
    mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)
    sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)
    return x_normed * sig_mix + mu_mix


class RandomResizeCrop(nn.Module):
    def __init__(
        self,
        virtual_crop_scale: tuple[float, float] = (1.0, 1.5),
        freq_scale: tuple[float, float] = (0.6, 1.0),
        time_scale: tuple[float, float] = (0.6, 1.5),
    ) -> None:
        super().__init__()
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = "bicubic"
        assert time_scale[1] >= 1.0 and freq_scale[1] >= 1.0

    @staticmethod
    def get_params(
        virtual_crop_size: tuple[int, int],
        in_size: tuple[int, int],
        time_scale: tuple[float, float],
        freq_scale: tuple[float, float],
    ) -> tuple[int, int, int, int]:
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w

    def forward(self, lms: torch.Tensor) -> torch.Tensor:
        virtual_crop_size = [int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)]
        virtual_crop_area = torch.zeros((lms.shape[0], virtual_crop_size[0], virtual_crop_size[1]), device=lms.device)
        _, lh, lw = virtual_crop_area.shape
        c, h, w = lms.shape
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, y : y + h, x : x + w] = lms
        i, j, h, w = self.get_params(
            tuple(virtual_crop_area.shape[-2:]),
            tuple(lms.shape[-2:]),
            self.time_scale,
            self.freq_scale,
        )
        crop = virtual_crop_area[:, i : i + h, j : j + w]
        lms = F.interpolate(
            crop.unsqueeze(1),
            size=lms.shape[-2:],
            mode=self.interpolation,
            align_corners=True,
        ).squeeze(1)
        return lms.float()

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(virtual_crop_size={self.virtual_crop_scale}"
        format_string += ", time_scale={0}".format(tuple(round(s, 4) for s in self.time_scale))
        format_string += ", freq_scale={0})".format(tuple(round(r, 4) for r in self.freq_scale))
        return format_string
