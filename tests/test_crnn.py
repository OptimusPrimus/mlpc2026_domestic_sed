from domestic_sed.architectures import CRNN, CRNNBlockConfig, build_default_crnn_blocks

import torch


def test_crnn_forward_and_summary() -> None:
    model = CRNN(
        conv_blocks=[
            CRNNBlockConfig(
                out_channels=32,
                conv1_kernel_size=(3, 3),
                conv2_kernel_size=(1, 1),
                pool_kernel_size=(2, 2),
                pool_stride=(2, 2),
            ),
            CRNNBlockConfig(
                out_channels=64,
                conv1_kernel_size=(3, 1),
                conv1_stride=(1, 1),
                conv2_kernel_size=(1, 3),
                conv2_dilation=(1, 2),
                pool_kernel_size=None,
            ),
        ],
        lstm_hidden_size=128,
        lstm_num_layers=1,
        lstm_dropout=0.1,
        use_batch_norm=True,
        dropout=0.2,
    )

    inputs = torch.randn(4, 128, 200)
    outputs = model(inputs)
    summary = model.summarize(input_frames=200)

    assert outputs.shape[0] == 4
    assert outputs.shape[2] == 15
    assert summary[0].name == "input"
    assert summary[1].name == "initial_conv"
    assert [entry.name for entry in summary[2:5]] == [
        "conv_block_1_conv_1",
        "conv_block_1_conv_2",
        "conv_block_1_pool",
    ]
    assert "conv_block_2_pool" not in {entry.name for entry in summary}
    assert summary[-2].name == "lstm_output"
    assert summary[-2].receptive_field == (128, 200)
    assert outputs.shape[1:] == model.output_shape(input_frames=200)
    assert model.conv_output_shape(input_frames=200) == summary[-4].output_shape


def test_crnn_lstm_respects_input_lengths() -> None:
    model = CRNN(
        conv_blocks=[
            CRNNBlockConfig(
                out_channels=16,
                conv1_kernel_size=(3, 3),
                conv2_kernel_size=(1, 1),
                pool_kernel_size=(2, 2),
                pool_stride=(2, 2),
            ),
        ],
        lstm_hidden_size=32,
        lstm_num_layers=1,
        lstm_dropout=0.0,
        use_batch_norm=False,
        dropout=0.0,
    ).eval()

    short_frames = 120
    padded_frames = 200
    short_input = torch.randn(1, 128, short_frames)
    padded_input = torch.zeros(2, 128, padded_frames)
    padded_input[0, :, :short_frames] = short_input[0]
    padded_input[1] = torch.randn(128, padded_frames)
    input_lengths = torch.tensor([short_frames, padded_frames], dtype=torch.long)

    standalone_output = model(short_input)
    padded_output = model(padded_input, input_lengths=input_lengths)
    valid_frames = model.output_shape(input_frames=short_frames)[0]

    torch.testing.assert_close(
        padded_output[0, :valid_frames],
        standalone_output[0],
    )
    assert torch.count_nonzero(padded_output[0, valid_frames:]) == 0


def test_build_default_crnn_blocks() -> None:
    blocks = build_default_crnn_blocks(p1=1, p2=1)

    assert len(blocks) == 12
    assert blocks[0].conv1_kernel_size == (3, 3)
    assert blocks[0].conv2_kernel_size == (1, 1)
    assert blocks[0].pool_kernel_size == (2, 2)
    assert blocks[0].out_channels == 32

    assert blocks[1].conv1_kernel_size == (3, 3)
    assert blocks[1].conv2_kernel_size == (1, 1)
    assert blocks[2].conv1_kernel_size == (1, 1)
    assert blocks[2].conv2_kernel_size == (1, 1)
    assert blocks[3].out_channels == 32
    assert blocks[4].out_channels == 64
    assert blocks[8].out_channels == 128

    mixed_blocks = build_default_crnn_blocks(p1=1, p2=2)
    assert mixed_blocks[1].conv1_kernel_size == (3, 3)
    assert mixed_blocks[1].conv2_kernel_size == (1, 3)

    scaled_blocks = build_default_crnn_blocks(p1=1, p2=1, channel_multiplier=2, depth=14)
    assert len(scaled_blocks) == 14
    assert scaled_blocks[0].out_channels == 64
    assert scaled_blocks[4].out_channels == 128
    assert scaled_blocks[8].out_channels == 256
    assert scaled_blocks[-1].out_channels == 256

    pooled_block_indices = [
        index for index, block in enumerate(blocks, start=1) if block.pool_kernel_size == (2, 2)
    ]
    assert pooled_block_indices == [1, 2, 4]
