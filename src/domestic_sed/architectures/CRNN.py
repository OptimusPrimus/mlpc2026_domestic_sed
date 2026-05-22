from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Sequence

import torch
from torch import nn


def _to_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def _same_padding(kernel_size: int | tuple[int, int], dilation: int | tuple[int, int]) -> tuple[int, int]:
    kernel_h, kernel_w = _to_pair(kernel_size)
    dilation_h, dilation_w = _to_pair(dilation)
    pad_h = dilation_h * (kernel_h - 1) // 2
    pad_w = dilation_w * (kernel_w - 1) // 2
    return (pad_h, pad_w)


@dataclass(frozen=True)
class CRNNBlockConfig:
    out_channels: int
    conv1_kernel_size: int | tuple[int, int] = (3, 3)
    conv1_stride: int | tuple[int, int] = (1, 1)
    conv1_dilation: int | tuple[int, int] = (1, 1)
    conv1_padding: int | tuple[int, int] | str = "same"
    conv2_kernel_size: int | tuple[int, int] = (3, 3)
    conv2_stride: int | tuple[int, int] = (1, 1)
    conv2_dilation: int | tuple[int, int] = (1, 1)
    conv2_padding: int | tuple[int, int] | str = "same"
    pool_kernel_size: int | tuple[int, int] | None = None
    pool_stride: int | tuple[int, int] | None = None
    pool_padding: int | tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class CRNNSummaryEntry:
    name: str
    output_shape: tuple[int, int, int]
    receptive_field: tuple[int, int]
    effective_stride: tuple[int, int]


def build_default_crnn_blocks(p1: int, p2: int) -> list[CRNNBlockConfig]:
    if p1 < 0 or p2 < 0:
        raise ValueError("p1 and p2 must be non-negative")

    channels = [64, 64, 128, 128, 256, 256, 512]
    pooling_blocks = {1, 2, 4}
    blocks = [
        CRNNBlockConfig(
            out_channels=channels[0],
            conv1_kernel_size=(3, 3),
            conv2_kernel_size=(1, 1),
            pool_kernel_size=(2, 2),
            pool_stride=(2, 2),
        )
    ]

    for block_index, out_channels in enumerate(channels[1:], start=1):
        conv1_index = (2 * block_index) - 1
        conv2_index = 2 * block_index
        pool_kernel_size = (2, 2) if (block_index + 1) in pooling_blocks else None
        blocks.append(
            CRNNBlockConfig(
                out_channels=out_channels,
                conv1_kernel_size=(
                    3 if conv1_index <= p1 else 1,
                    3 if conv1_index <= p2 else 1,
                ),
                conv2_kernel_size=(
                    3 if conv2_index <= p1 else 1,
                    3 if conv2_index <= p2 else 1,
                ),
                pool_kernel_size=pool_kernel_size,
                pool_stride=(2, 2) if pool_kernel_size is not None else None,
            )
        )

    return blocks


class _PreNormResidualBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        config: CRNNBlockConfig,
        conv_bias: bool,
        use_batch_norm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        conv1_padding = CRNN._resolve_conv_padding(
            config.conv1_padding,
            config.conv1_kernel_size,
            config.conv1_dilation,
        )
        conv2_padding = CRNN._resolve_conv_padding(
            config.conv2_padding,
            config.conv2_kernel_size,
            config.conv2_dilation,
        )
        self.pre_norm1 = nn.Sequential(
            nn.BatchNorm2d(in_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.pre_norm2 = nn.Sequential(
            nn.BatchNorm2d(config.out_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.out_channels,
            kernel_size=config.conv1_kernel_size,
            stride=config.conv1_stride,
            padding=conv1_padding,
            dilation=config.conv1_dilation,
            bias=conv_bias and not use_batch_norm,
        )
        self.conv2 = nn.Conv2d(
            in_channels=config.out_channels,
            out_channels=config.out_channels,
            kernel_size=config.conv2_kernel_size,
            stride=config.conv2_stride,
            padding=conv2_padding,
            dilation=config.conv2_dilation,
            bias=conv_bias and not use_batch_norm,
        )
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

        conv1_stride = _to_pair(config.conv1_stride)
        conv2_stride = _to_pair(config.conv2_stride)
        residual_stride = (
            conv1_stride[0] * conv2_stride[0],
            conv1_stride[1] * conv2_stride[1],
        )
        if in_channels != config.out_channels or residual_stride != (1, 1):
            self.residual_projection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=config.out_channels,
                kernel_size=1,
                stride=residual_stride,
                bias=conv_bias and not use_batch_norm,
            )
        else:
            self.residual_projection = nn.Identity()

        if config.pool_kernel_size is None:
            self.pool = nn.Identity()
        else:
            self.pool = nn.MaxPool2d(
                kernel_size=config.pool_kernel_size,
                stride=config.pool_stride or config.pool_kernel_size,
                padding=config.pool_padding,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.residual_projection(inputs)
        outputs = self.conv1(self.pre_norm1(inputs))
        outputs = self.conv2(self.pre_norm2(outputs))
        outputs = self.dropout(outputs)
        outputs = outputs + residual
        return self.pool(outputs)


class CRNN(nn.Module):
    """Configurable CRNN for 128-bin log-mel spectrogram sequences."""

    def __init__(
        self,
        conv_blocks: Sequence[CRNNBlockConfig],
        *,
        input_bins: int = 128,
        input_channels: int = 1,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.0,
        bidirectional: bool = True,
        conv_bias: bool = True,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        output_size: int = 15,
    ) -> None:
        super().__init__()

        if input_bins <= 0:
            raise ValueError("input_bins must be positive")
        if not conv_blocks:
            raise ValueError("conv_blocks must contain at least one block")
        if lstm_num_layers <= 0:
            raise ValueError("lstm_num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= lstm_dropout < 1.0:
            raise ValueError("lstm_dropout must be in [0, 1)")

        self.input_bins = input_bins
        self.input_channels = input_channels
        self.use_batch_norm = use_batch_norm
        self.output_size = output_size
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        self.initial_conv_kernel_size = (5, 5)
        self.initial_conv_stride = (2, 2)
        self.initial_conv_dilation = (1, 1)
        self.initial_conv_padding = self._resolve_conv_padding("same", self.initial_conv_kernel_size, self.initial_conv_dilation)

        self.conv_blocks_config = list(conv_blocks)
        conv_layers: list[nn.Module] = []
        in_channels = input_channels
        self.initial_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=self.initial_conv_kernel_size,
            stride=self.initial_conv_stride,
            padding=self.initial_conv_padding,
            dilation=self.initial_conv_dilation,
            bias=conv_bias and not use_batch_norm,
        )
        self.initial_norm = nn.BatchNorm2d(input_channels) if use_batch_norm else nn.Identity()
        self.initial_activation = nn.ReLU()

        for block in self.conv_blocks_config:
            conv_layers.append(
                _PreNormResidualBlock(
                    in_channels=in_channels,
                    config=block,
                    conv_bias=conv_bias,
                    use_batch_norm=use_batch_norm,
                    dropout=dropout,
                )
            )
            in_channels = block.out_channels

        self.convolutional_stack = nn.Sequential(*conv_layers)

        conv_channels = self.conv_blocks_config[-1].out_channels
        conv_output_bins = self._reduced_frequency_bins()
        if conv_output_bins <= 0:
            raise ValueError("Convolutional stack collapses the frequency axis; adjust pooling or stride.")

        lstm_input_size = conv_channels * conv_output_bins
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        classifier_input_size = lstm_hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(classifier_input_size, output_size)

        self.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)
        elif inputs.ndim != 4:
            raise ValueError(
                "Expected input of shape (batch, mel_bins, frames) or "
                "(batch, channels, mel_bins, frames)"
            )

        if inputs.shape[-2] != self.input_bins:
            raise ValueError(f"Expected {self.input_bins} mel bins, got {inputs.shape[-2]}")

        features = self.initial_activation(self.initial_norm(self.initial_conv(inputs)))
        features = self.convolutional_stack(features)
        batch_size, channels, freq_bins, frames = features.shape
        features = features.permute(0, 3, 1, 2).contiguous().view(batch_size, frames, channels * freq_bins)
        features, _ = self.lstm(features)
        return self.classifier(features)

    def conv_output_shape(self, input_frames: int) -> tuple[int, int, int]:
        summary = self.summarize(input_frames=input_frames)
        for entry in reversed(summary):
            if entry.name.startswith("conv_block_") or entry.name == "initial_conv":
                return entry.output_shape
        raise RuntimeError("No convolutional summary entry found")

    def output_shape(self, input_frames: int) -> tuple[int, int]:
        summary = self.summarize(input_frames=input_frames)
        frames, classes, _ = summary[-1].output_shape
        return (frames, classes)

    def summarize(self, input_frames: int) -> list[CRNNSummaryEntry]:
        if input_frames <= 0:
            raise ValueError("input_frames must be positive")

        freq = self.input_bins
        time = input_frames
        channels = self.input_channels
        rf_freq = 1
        rf_time = 1
        jump_freq = 1
        jump_time = 1
        entries: list[CRNNSummaryEntry] = [
            CRNNSummaryEntry(
                name="input",
                output_shape=(channels, freq, time),
                receptive_field=(rf_freq, rf_time),
                effective_stride=(jump_freq, jump_time),
            )
        ]

        freq = self._conv_dim(
            freq,
            self.initial_conv_kernel_size[0],
            self.initial_conv_stride[0],
            self.initial_conv_padding[0],
            self.initial_conv_dilation[0],
        )
        time = self._conv_dim(
            time,
            self.initial_conv_kernel_size[1],
            self.initial_conv_stride[1],
            self.initial_conv_padding[1],
            self.initial_conv_dilation[1],
        )
        rf_freq = rf_freq + (self.initial_conv_kernel_size[0] - 1) * self.initial_conv_dilation[0] * jump_freq
        rf_time = rf_time + (self.initial_conv_kernel_size[1] - 1) * self.initial_conv_dilation[1] * jump_time
        jump_freq *= self.initial_conv_stride[0]
        jump_time *= self.initial_conv_stride[1]
        entries.append(
            CRNNSummaryEntry(
                name="initial_conv",
                output_shape=(channels, freq, time),
                receptive_field=(rf_freq, rf_time),
                effective_stride=(jump_freq, jump_time),
            )
        )

        for block_index, block in enumerate(self.conv_blocks_config, start=1):
            conv1_kernel = _to_pair(block.conv1_kernel_size)
            conv1_stride = _to_pair(block.conv1_stride)
            conv1_dilation = _to_pair(block.conv1_dilation)
            conv1_padding = self._resolve_conv_padding(block.conv1_padding, conv1_kernel, conv1_dilation)

            freq = self._conv_dim(freq, conv1_kernel[0], conv1_stride[0], conv1_padding[0], conv1_dilation[0])
            time = self._conv_dim(time, conv1_kernel[1], conv1_stride[1], conv1_padding[1], conv1_dilation[1])
            rf_freq = rf_freq + (conv1_kernel[0] - 1) * conv1_dilation[0] * jump_freq
            rf_time = rf_time + (conv1_kernel[1] - 1) * conv1_dilation[1] * jump_time
            jump_freq *= conv1_stride[0]
            jump_time *= conv1_stride[1]
            entries.append(
                CRNNSummaryEntry(
                    name=f"conv_block_{block_index}_conv_1",
                    output_shape=(block.out_channels, freq, time),
                    receptive_field=(rf_freq, rf_time),
                    effective_stride=(jump_freq, jump_time),
                )
            )

            conv2_kernel = _to_pair(block.conv2_kernel_size)
            conv2_stride = _to_pair(block.conv2_stride)
            conv2_dilation = _to_pair(block.conv2_dilation)
            conv2_padding = self._resolve_conv_padding(block.conv2_padding, conv2_kernel, conv2_dilation)

            freq = self._conv_dim(freq, conv2_kernel[0], conv2_stride[0], conv2_padding[0], conv2_dilation[0])
            time = self._conv_dim(time, conv2_kernel[1], conv2_stride[1], conv2_padding[1], conv2_dilation[1])
            rf_freq = rf_freq + (conv2_kernel[0] - 1) * conv2_dilation[0] * jump_freq
            rf_time = rf_time + (conv2_kernel[1] - 1) * conv2_dilation[1] * jump_time
            jump_freq *= conv2_stride[0]
            jump_time *= conv2_stride[1]
            entries.append(
                CRNNSummaryEntry(
                    name=f"conv_block_{block_index}_conv_2",
                    output_shape=(block.out_channels, freq, time),
                    receptive_field=(rf_freq, rf_time),
                    effective_stride=(jump_freq, jump_time),
                )
            )

            if block.pool_kernel_size is not None:
                pool_kernel = _to_pair(block.pool_kernel_size)
                pool_stride = _to_pair(block.pool_stride or block.pool_kernel_size)
                pool_padding = _to_pair(block.pool_padding)

                freq = self._pool_dim(freq, pool_kernel[0], pool_stride[0], pool_padding[0])
                time = self._pool_dim(time, pool_kernel[1], pool_stride[1], pool_padding[1])
                rf_freq = rf_freq + (pool_kernel[0] - 1) * jump_freq
                rf_time = rf_time + (pool_kernel[1] - 1) * jump_time
                jump_freq *= pool_stride[0]
                jump_time *= pool_stride[1]
                entries.append(
                    CRNNSummaryEntry(
                        name=f"conv_block_{block_index}_pool",
                        output_shape=(block.out_channels, freq, time),
                        receptive_field=(rf_freq, rf_time),
                        effective_stride=(jump_freq, jump_time),
                    )
                )

        if freq <= 0 or time <= 0:
            raise ValueError("Configuration produces a non-positive tensor dimension")

        lstm_feature_size = entries[-1].output_shape[0] * entries[-1].output_shape[1]
        lstm_output_size = self.lstm_hidden_size * (2 if self.bidirectional else 1)
        entries.append(
            CRNNSummaryEntry(
                name="lstm_input",
                output_shape=(time, lstm_feature_size, 1),
                receptive_field=(rf_freq, rf_time),
                effective_stride=(jump_freq, jump_time),
            )
        )
        entries.append(
            CRNNSummaryEntry(
                name="lstm_output",
                output_shape=(time, lstm_output_size, 1),
                receptive_field=(self.input_bins, input_frames),
                effective_stride=(jump_freq, jump_time),
            )
        )
        entries.append(
            CRNNSummaryEntry(
                name="output",
                output_shape=(time, self.output_size, 1),
                receptive_field=(self.input_bins, input_frames),
                effective_stride=(jump_freq, jump_time),
            )
        )
        return entries

    def print_summary(self, input_frames: int) -> None:
        print(self.format_summary(input_frames=input_frames))

    def format_summary(self, input_frames: int) -> str:
        lines = [
            "name | output_shape | receptive_field(freq,time) | stride(freq,time)",
            "-" * 76,
        ]
        for entry in self.summarize(input_frames=input_frames):
            lines.append(
                f"{entry.name} | {entry.output_shape} | {entry.receptive_field} | {entry.effective_stride}"
            )
        return "\n".join(lines)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, parameter in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)
                        hidden_size = parameter.shape[0] // 4
                        parameter.data[hidden_size : 2 * hidden_size].fill_(1.0)

        for module in self.modules():
            if isinstance(module, _PreNormResidualBlock):
                last_norm = module.pre_norm2[0]
                if isinstance(last_norm, nn.BatchNorm2d):
                    nn.init.zeros_(last_norm.weight)

    @staticmethod
    def _resolve_conv_padding(
        padding: int | tuple[int, int] | str,
        kernel_size: int | tuple[int, int],
        dilation: int | tuple[int, int],
    ) -> int | tuple[int, int]:
        if padding == "same":
            return _same_padding(kernel_size, dilation)
        if isinstance(padding, str):
            raise ValueError(f"Unsupported padding mode: {padding}")
        return padding

    @staticmethod
    def _conv_dim(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
        return floor((size + (2 * padding) - dilation * (kernel - 1) - 1) / stride + 1)

    @staticmethod
    def _pool_dim(size: int, kernel: int, stride: int, padding: int) -> int:
        return floor((size + (2 * padding) - kernel) / stride + 1)

    def _reduced_frequency_bins(self) -> int:
        freq = self._conv_dim(
            self.input_bins,
            self.initial_conv_kernel_size[0],
            self.initial_conv_stride[0],
            self.initial_conv_padding[0],
            self.initial_conv_dilation[0],
        )
        for block in self.conv_blocks_config:
            conv1_kernel = _to_pair(block.conv1_kernel_size)
            conv1_stride = _to_pair(block.conv1_stride)
            conv1_dilation = _to_pair(block.conv1_dilation)
            conv1_padding = self._resolve_conv_padding(block.conv1_padding, conv1_kernel, conv1_dilation)
            conv2_kernel = _to_pair(block.conv2_kernel_size)
            conv2_stride = _to_pair(block.conv2_stride)
            conv2_dilation = _to_pair(block.conv2_dilation)
            conv2_padding = self._resolve_conv_padding(block.conv2_padding, conv2_kernel, conv2_dilation)

            freq = self._conv_dim(freq, conv1_kernel[0], conv1_stride[0], conv1_padding[0], conv1_dilation[0])
            freq = self._conv_dim(freq, conv2_kernel[0], conv2_stride[0], conv2_padding[0], conv2_dilation[0])
            if block.pool_kernel_size is not None:
                pool_kernel = _to_pair(block.pool_kernel_size)
                pool_stride = _to_pair(block.pool_stride or block.pool_kernel_size)
                pool_padding = _to_pair(block.pool_padding)
                freq = self._pool_dim(freq, pool_kernel[0], pool_stride[0], pool_padding[0])
        return freq
