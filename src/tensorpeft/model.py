# Copyright 2026 Angshul Majumdar.
"""TensorPEFT: tensor gradient-pursuit adapters.

Matrices of equal shape across all transformer blocks are stacked into one update tensor
(layer x [role] x d_in x d_out).  Each group's update is a sum of rank-one CP atoms

    dW = sum_i  a_i  u_i(layer) o u_i(role) o u_i(in) o u_i(out).

``init_atoms`` runs one gradient probe, and the atoms are the leading rank-one CP components of
the negative gradient tensor (HOPM with deflation).  Coefficients start at zero, so the model is
unchanged until training starts; all coefficients and factors are then trained normally.
"""
import functools
import os

import torch
import torch.nn as nn

from . import __version__
from .config import CONFIG_NAME, WEIGHTS_NAME, TensorPEFTConfig, _resolve
from .cp import cp_value, outer, rank1_cp
from .layers import attach_hooks, find_slots


class TensorGroup(nn.Module):
    def __init__(self, slots, k, alpha_scale):
        super().__init__()
        self.slots = slots
        self.d_in, self.d_out = slots[0].shape
        self.layers = sorted({s.layer for s in slots})
        self.roles = sorted({s.role for s in slots})
        self.has_p = len(self.roles) > 1
        self.shape = (len(self.layers),) + ((len(self.roles),) if self.has_p else ()) + (self.d_in, self.d_out)
        self.index = {id(s): (self.layers.index(s.layer),) + ((self.roles.index(s.role),) if self.has_p else ())
                      for s in slots}
        self.k = k
        self.alpha_scale = alpha_scale
        dev = slots[0].module.weight.device
        self.alpha_raw = nn.Parameter(torch.zeros(k, device=dev))
        self.factors = nn.ParameterList([nn.Parameter(torch.zeros(k, n, device=dev)) for n in self.shape])
        self.register_buffer("initialized", torch.tensor(False, device=dev))
        self._cache = None

    @property
    def name(self):
        return f"{'+'.join(self.roles)} {list(self.shape)}"

    def n_params(self):
        return self.k * (1 + sum(self.shape))

    def meta(self):
        return {"roles": self.roles, "layers": self.layers, "shape": list(self.shape)}

    def prepare(self):
        if not bool(self.initialized):
            self._cache = None
            return
        a = self.alpha_raw * self.alpha_scale
        F = list(self.factors)
        C = torch.einsum("k,kl,kp->klp", a, F[0], F[1]) if self.has_p else a[:, None] * F[0]
        self._cache = (C, F[-2], F[-1])

    def delta(self, slot, x):
        if self._cache is None:
            return None
        C, U_in, U_out = self._cache
        c = C[(slice(None),) + self.index[id(slot)]]
        return ((x.to(U_in.dtype) @ U_in.T) * c) @ U_out

    @torch.no_grad()
    def dense_delta(self, slot):
        self.prepare()
        if self._cache is None:
            return None
        C, U_in, U_out = self._cache
        c = C[(slice(None),) + self.index[id(slot)]]
        return (U_in.T * c) @ U_out

    @torch.no_grad()
    def stacked_grad(self):
        w = self.slots[0].module.weight
        G = torch.zeros(self.shape, dtype=torch.float32, device=w.device)
        for s in self.slots:
            G[self.index[id(s)]] = s.grad().float()
        return G


class TensorPEFTModel(nn.Module):
    """Wraps a Hugging Face causal LM.  Use :func:`get_tensorpeft_model` to create one."""

    def __init__(self, base_model, config: TensorPEFTConfig):
        super().__init__()
        self.base_model = base_model
        self.peft_config = config
        config.tensorpeft_version = __version__
        if config.base_model_name_or_path is None:
            config.base_model_name_or_path = getattr(getattr(base_model, "config", None), "_name_or_path", None)
        for p in base_model.parameters():
            p.requires_grad_(False)
        self.slots = find_slots(base_model, config.target_modules, config.exclude_modules, config.split_fused_qkv)
        by_shape = {}
        for s in self.slots:
            by_shape.setdefault(s.shape, []).append(s)
        self.groups = nn.ModuleList([TensorGroup(ss, config.atoms_per_group, config.alpha_scale)
                                     for ss in by_shape.values()])
        self._group_of = {id(s): g for g in self.groups for s in g.slots}
        self._handles = attach_hooks(self.slots, lambda s, x: self._group_of[id(s)].delta(s, x))
        self._handles.append(base_model.register_forward_pre_hook(lambda m, a: self._prepare()))
        self._modules_w = list({id(s.module): s.module for s in self.slots}.values())

        # forward with the base model's signature (so HF Trainer keeps the right dataset columns)
        def forward(*args, **kwargs):
            return self.base_model(*args, **kwargs)
        functools.update_wrapper(forward, base_model.forward)
        self.forward = forward

    # ------------------------------------------------------------------ plumbing
    def _prepare(self):
        for g in self.groups:
            g.prepare()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base_model"), name)

    def get_base_model(self):
        return self.base_model

    @property
    def is_initialized(self):
        return all(bool(g.initialized) for g in self.groups)

    def get_nb_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.base_model.parameters()) + trainable
        return trainable, total

    def print_trainable_parameters(self):
        t, a = self.get_nb_trainable_parameters()
        print(f"trainable params: {t:,} || all params: {a:,} || trainable%: {100 * t / a:.4f}")

    def describe(self):
        return [f"{g.name}: {g.k} atoms, {g.n_params():,} params" for g in self.groups]

    # ------------------------------------------------------------------ initialisation
    def init_atoms(self, batches, n_batches=None, loss_fn=None, verbose=True):
        """Choose all atoms from one gradient probe.

        batches: iterable of dicts (``input_ids``, optional ``attention_mask`` / ``labels``) or of
            input-id tensors (causal-LM labels = inputs).  A DataLoader works.
        n_batches: how many batches to use (default ``config.init_batches``).
        loss_fn: optional ``loss_fn(base_model, batch) -> scalar``.
        Returns, per group, each atom's initial descent rate <-G, atom>  (= -dL/da at a = 0).
        """
        n_batches = n_batches or self.peft_config.init_batches
        dev = self._modules_w[0].weight.device
        was_training = self.base_model.training
        self.base_model.eval()
        for m in self._modules_w:
            m.weight.requires_grad_(True)
            m.weight.grad = None
        used = 0
        for b in batches:
            if used >= n_batches:
                break
            if loss_fn is not None:
                loss = loss_fn(self.base_model, b)
            else:
                if torch.is_tensor(b):
                    b = {"input_ids": b}
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                if "labels" not in b:
                    b["labels"] = b["input_ids"]
                loss = self.base_model(**b).loss
            loss.backward()
            used += 1
        if used == 0:
            raise ValueError("init_atoms: no batches given")
        report = []
        with torch.no_grad():
            for g in self.groups:
                negG = -g.stacked_grad() / used       # mean gradient over the batches used
                T = negG.clone()
                sig = []
                for i in range(g.k):
                    us, s = rank1_cp(T, iters=self.peft_config.hopm_iters)
                    for pl, u in zip(g.factors, us):
                        pl[i].copy_(u)
                    T = T - s * outer(us)                 # deflation
                    sig.append(float(cp_value(negG, us)))
                del negG, T
                g.alpha_raw.zero_()
                g.initialized.fill_(True)
                report.append(sig)
        for m in self._modules_w:
            m.weight.grad = None
            m.weight.requires_grad_(False)
        for p in self.parameters():
            p.grad = None
        self.base_model.train(was_training)
        if verbose:
            for g, s in zip(self.groups, report):
                print(f"TensorPEFT  {g.name}: {g.k} atoms, initial descent rate {s[0]:.4g} .. {s[-1]:.4g}")
        return report

    # ------------------------------------------------------------------ save / load
    def adapter_state_dict(self):
        sd = {}
        for gi, g in enumerate(self.groups):
            sd[f"group{gi}.alpha_raw"] = g.alpha_raw.detach().cpu().contiguous()
            for m, pl in enumerate(g.factors):
                sd[f"group{gi}.factor{m}"] = pl.detach().cpu().contiguous()
        return sd

    def save_pretrained(self, save_directory, **kwargs):
        if not self.is_initialized:
            raise RuntimeError("call init_atoms(...) before saving")
        from safetensors.torch import save_file
        os.makedirs(save_directory, exist_ok=True)
        self.peft_config.save_pretrained(save_directory, extra={"groups": [g.meta() for g in self.groups]})
        save_file(self.adapter_state_dict(), os.path.join(save_directory, WEIGHTS_NAME),
                  metadata={"format": "pt", "library": "tensorpeft"})
        readme = os.path.join(save_directory, "README.md")
        if not os.path.exists(readme):
            with open(readme, "w") as f:
                f.write(_model_card(self))

    @classmethod
    def from_pretrained(cls, base_model, model_id, is_trainable=False, **hub_kwargs):
        config, raw = TensorPEFTConfig.from_pretrained(model_id, **hub_kwargs)
        model = cls(base_model, config)
        saved = raw.get("groups")
        if saved is not None:
            mine = [g.meta() for g in model.groups]
            if [(m["roles"], m["shape"]) for m in mine] != [(m["roles"], m["shape"]) for m in saved]:
                raise ValueError("adapter does not match this base model's architecture")
        from safetensors.torch import load_file
        sd = load_file(_resolve(model_id, WEIGHTS_NAME, **hub_kwargs))
        with torch.no_grad():
            for gi, g in enumerate(model.groups):
                g.alpha_raw.copy_(sd[f"group{gi}.alpha_raw"])
                for m, pl in enumerate(g.factors):
                    pl.copy_(sd[f"group{gi}.factor{m}"])
                g.initialized.fill_(True)
        for p in model.parameters():
            if p.requires_grad and not is_trainable:
                p.requires_grad_(False)
        return model

    def push_to_hub(self, repo_id, private=None, token=None, commit_message="Upload TensorPEFT adapter"):
        import tempfile
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id, private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            self.save_pretrained(d)
            return api.upload_folder(repo_id=repo_id, folder_path=d, commit_message=commit_message)

    # ------------------------------------------------------------------ merge
    @torch.no_grad()
    def merge_and_unload(self):
        """Write dW into the base weights and return the plain base model (no inference overhead)."""
        for s in self.slots:
            d = self._group_of[id(s)].dense_delta(s)
            if d is not None:
                w = s.weight()
                w.add_(d.to(w.dtype))
        return self.unload()

    def unload(self):
        """Remove the adapter hooks without merging and return the base model."""
        for h in self._handles:
            h.remove()
        self._handles = []
        return self.base_model


def get_tensorpeft_model(model, config: TensorPEFTConfig = None):
    """Wrap a Hugging Face model with TensorPEFT.  Call ``.init_atoms(batches)`` before training."""
    return TensorPEFTModel(model, config or TensorPEFTConfig())


def _model_card(m):
    c = m.peft_config
    t, _ = m.get_nb_trainable_parameters()
    return f"""---
library_name: tensorpeft
base_model: {c.base_model_name_or_path or 'unknown'}
tags:
- tensorpeft
- peft
---

# TensorPEFT adapter

Base model: `{c.base_model_name_or_path}`; {c.atoms_per_group} atoms per tensor group; {t:,} adapter parameters.

```python
from transformers import AutoModelForCausalLM
from tensorpeft import TensorPEFTModel
base = AutoModelForCausalLM.from_pretrained("{c.base_model_name_or_path}")
model = TensorPEFTModel.from_pretrained(base, "<this repo id>")
model = model.merge_and_unload()   # optional
```
"""
