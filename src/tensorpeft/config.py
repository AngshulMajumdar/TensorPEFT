# Copyright 2026 Angshul Majumdar.
"""TensorPEFTConfig."""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union

CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapter_model.safetensors"


@dataclass
class TensorPEFTConfig:
    """Configuration of a TensorPEFT adapter.

    Args:
        atoms_per_group: rank-one CP atoms per tensor group (all chosen at initialisation).
            Trainable parameters = sum over groups of atoms_per_group * (1 + sum of the group's mode sizes).
        target_modules: ``"all-linear"`` (every Linear / Conv1D inside the transformer blocks) or a
            list of module-name suffixes, e.g. ``["q_proj", "v_proj"]`` (``"c_attn.q"`` etc. for GPT-2).
        exclude_modules: substrings of module names that are never adapted.
        split_fused_qkv: split GPT-2-style fused ``c_attn`` = [q|k|v] into q, k, v.
        hopm_iters: higher-order power iterations used to compute each atom.
        alpha_scale: coefficients are parametrised as ``alpha_scale * alpha_raw`` so that a single
            optimiser learning rate (about 3e-3 for Adam/AdamW) suits both coefficients and factors.
        init_batches: default number of batches used by ``init_atoms``.
    """
    atoms_per_group: int = 4
    target_modules: Union[str, List[str]] = "all-linear"
    exclude_modules: List[str] = field(default_factory=lambda: ["lm_head", "embed", "wte", "wpe"])
    split_fused_qkv: bool = True
    hopm_iters: int = 30
    alpha_scale: float = 100.0 / 3.0
    init_batches: int = 4
    base_model_name_or_path: Optional[str] = None
    peft_type: str = "TENSORPEFT"
    task_type: str = "CAUSAL_LM"
    tensorpeft_version: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    def save_pretrained(self, save_directory, extra=None):
        os.makedirs(save_directory, exist_ok=True)
        d = self.to_dict()
        if extra:
            d.update(extra)
        with open(os.path.join(save_directory, CONFIG_NAME), "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_dict(cls, d):
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in names})

    @classmethod
    def from_pretrained(cls, model_id, **hub_kwargs):
        path = _resolve(model_id, CONFIG_NAME, **hub_kwargs)
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d), d


def _resolve(model_id, filename, **hub_kwargs):
    local = os.path.join(model_id, filename)
    if os.path.isfile(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=model_id, filename=filename, **hub_kwargs)
