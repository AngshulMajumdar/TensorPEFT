# Copyright 2026 Angshul Majumdar.
"""TensorPEFT: parameter-efficient fine-tuning with tensor gradient-pursuit adapters."""
__version__ = "0.1.0"

from .config import TensorPEFTConfig  # noqa: E402
from .model import TensorPEFTModel, get_tensorpeft_model  # noqa: E402


def __getattr__(name):  # lazy: importing the Trainer pulls in accelerate
    if name == "TensorPEFTTrainer":
        from .trainer import TensorPEFTTrainer
        return TensorPEFTTrainer
    raise AttributeError(name)


__all__ = ["TensorPEFTConfig", "TensorPEFTModel", "TensorPEFTTrainer", "get_tensorpeft_model", "__version__"]
