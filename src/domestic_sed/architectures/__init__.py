"""Model architectures for the domestic sound event detection task."""

from .CRNN import CRNN, CRNNBlockConfig, CRNNSummaryEntry, build_default_crnn_blocks

__all__ = ["CRNN", "CRNNBlockConfig", "CRNNSummaryEntry", "build_default_crnn_blocks"]
