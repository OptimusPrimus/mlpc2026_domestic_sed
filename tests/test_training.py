import torch

from domestic_sed.training import SoundEventLightningModule, build_arg_parser


def test_build_arg_parser_accepts_lr_linear_decay_epochs() -> None:
    args = build_arg_parser().parse_args(["--lr-linear-decay-epochs", "5"])

    assert args.lr_linear_decay_epochs == 5


def test_build_arg_parser_accepts_early_stopping_patience() -> None:
    args = build_arg_parser().parse_args(["--early-stopping-patience", "10"])

    assert args.early_stopping_patience == 10


def test_configure_optimizers_adds_linear_decay_scheduler() -> None:
    model = SoundEventLightningModule(
        class_to_index={f"class_{index}": index for index in range(15)},
        learning_rate=1e-3,
        weight_decay=0.0,
        lr_linear_decay_epochs=4,
        max_epochs=6,
    )

    configured = model.configure_optimizers()

    assert isinstance(configured, dict)
    scheduler = configured["lr_scheduler"]["scheduler"]
    optimizer = configured["optimizer"]
    lambda_fn = scheduler.lr_lambdas[0]

    assert isinstance(optimizer, torch.optim.AdamW)
    assert lambda_fn(0) == 1.0
    assert lambda_fn(1) == 1.0
    assert lambda_fn(2) == 1.0
    assert lambda_fn(3) == 0.75
    assert lambda_fn(4) == 0.5
    assert lambda_fn(5) == 0.25
    assert lambda_fn(6) == 0.0
