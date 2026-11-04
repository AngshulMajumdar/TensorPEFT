"""pytest tests for TensorPEFT (CPU, tiny random models)."""
import math

import pytest
import torch
import torch.nn.functional as F
import transformers as T

from tensorpeft import TensorPEFTConfig, TensorPEFTModel, get_tensorpeft_model
from tensorpeft.cp import cp_value, contract_except, outer, rank1_cp

V = 97


def tiny(arch):
    common = dict(vocab_size=V, max_position_embeddings=64)
    if arch == "gpt2":
        return T.GPT2LMHeadModel(T.GPT2Config(n_layer=3, n_embd=32, n_head=4, vocab_size=V, n_positions=64))
    if arch == "llama":
        return T.LlamaForCausalLM(T.LlamaConfig(num_hidden_layers=3, hidden_size=32, intermediate_size=48,
                                                num_attention_heads=4, num_key_value_heads=2, **common))
    if arch == "qwen2":
        return T.Qwen2ForCausalLM(T.Qwen2Config(num_hidden_layers=3, hidden_size=32, intermediate_size=48,
                                                num_attention_heads=4, num_key_value_heads=2, **common))
    if arch == "mistral":
        return T.MistralForCausalLM(T.MistralConfig(num_hidden_layers=3, hidden_size=32, intermediate_size=48,
                                                    num_attention_heads=4, num_key_value_heads=2, **common))
    if arch == "phi":
        return T.PhiForCausalLM(T.PhiConfig(num_hidden_layers=3, hidden_size=32, intermediate_size=48,
                                            num_attention_heads=4, **common))
    if arch == "gemma":
        return T.GemmaForCausalLM(T.GemmaConfig(num_hidden_layers=3, hidden_size=32, intermediate_size=48,
                                                num_attention_heads=4, num_key_value_heads=2, head_dim=8, **common))
    raise ValueError(arch)


ARCHS = ["gpt2", "llama", "qwen2", "mistral", "phi", "gemma"]


def batches(n=2, b=3, t=17, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, V, (b, t), generator=g) for _ in range(n)]


def lm_loss(model, x):
    logits = model(x[:, :-1]).logits
    return F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten())


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ---------------------------------------------------------------- CP / HOPM
@pytest.mark.parametrize("shape", [(5, 30, 40), (6, 4, 30, 20)])
def test_hopm_recovers_planted_rank1(shape):
    us = [F.normalize(torch.randn(n, dtype=torch.float64), dim=0) for n in shape]
    T_ = 5.0 * outer(us) + 0.05 * torch.randn(shape, dtype=torch.float64)
    est, sig = rank1_cp(T_, iters=50)
    assert min(abs(float(a @ b)) for a, b in zip(us, est)) > 0.99
    assert abs(sig - 5.0) < 0.3


def test_hopm_fixed_point():
    T_ = torch.randn(4, 3, 20, 25, dtype=torch.float64)
    est, sig = rank1_cp(T_, iters=500, tol=1e-14)
    res = max(float((contract_except(T_, est, n) / sig - est[n]).abs().max()) for n in range(4))
    assert res < 1e-6


# ---------------------------------------------------------------- model wrapping
@pytest.mark.parametrize("arch", ARCHS)
def test_unchanged_before_and_after_init(arch):
    base = tiny(arch).eval()
    x = batches(1)[0]
    ref = base(x).logits.detach().clone()
    model = get_tensorpeft_model(base, TensorPEFTConfig(atoms_per_group=2))
    assert torch.equal(model(x).logits, ref)
    model.init_atoms(batches(2), verbose=False)
    assert model.is_initialized
    assert float((model(x).logits - ref).abs().max()) < 1e-6        # a = 0 at init


@pytest.mark.parametrize("arch", ARCHS)
def test_initial_gradient_is_steepest_cp_direction(arch):
    """dL/d alpha_raw_i at init = -alpha_scale * sigma_i (atoms are descent directions)."""
    torch.set_default_dtype(torch.float64)
    try:
        base = tiny(arch).eval()
        model = get_tensorpeft_model(base, TensorPEFTConfig(atoms_per_group=3))
        xs = batches(2)
        sig = model.init_atoms(xs, verbose=False)
        loss = sum(lm_loss(model, x) for x in xs) / len(xs)
        grads = torch.autograd.grad(loss, [g.alpha_raw for g in model.groups])
        for g, gr, s in zip(model.groups, grads, sig):
            expect = -g.alpha_scale * torch.tensor(s, dtype=gr.dtype)
            assert torch.allclose(gr, expect, rtol=1e-6, atol=1e-10), (arch, gr, expect)
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize("arch", ARCHS)
def test_merge_equals_hooks_and_save_load_roundtrip(arch, tmp_path):
    torch.set_default_dtype(torch.float64)
    try:
        base = tiny(arch).eval()
        state = {k: v.clone() for k, v in base.state_dict().items()}
        x = batches(1, seed=5)[0]
        ref = base(x).logits.detach().clone()
        model = get_tensorpeft_model(base, TensorPEFTConfig(atoms_per_group=2))
        model.init_atoms(batches(2), verbose=False)
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(0.3 * torch.randn_like(p))
        hooked = model(x).logits.detach().clone()
        assert float((hooked - ref).abs().max()) > 1e-3          # the adapter does change the model
        model.save_pretrained(tmp_path)
        # reload onto a fresh copy of the same base
        base2 = tiny(arch).eval()
        base2.load_state_dict(state)
        re = TensorPEFTModel.from_pretrained(base2, str(tmp_path))
        assert float((re(x).logits - hooked).abs().max()) < 1e-10
        merged = model.merge_and_unload()
        assert float((merged(x).logits - hooked).abs().max()) < 1e-10
    finally:
        torch.set_default_dtype(torch.float32)


def test_param_count_and_grouping_gpt2():
    base = tiny("gpt2")
    m = get_tensorpeft_model(base, TensorPEFTConfig(atoms_per_group=4))
    names = [g.name for g in m.groups]
    assert names[0].startswith("attn.c_attn.k+attn.c_attn.q+attn.c_attn.v+attn.c_proj [3, 4, 32, 32]")
    expect = 4 * ((1 + 3 + 4 + 32 + 32) + (1 + 3 + 32 + 128) + (1 + 3 + 128 + 32))
    assert m.get_nb_trainable_parameters()[0] == expect
    assert all(not p.requires_grad for p in base.parameters())


def test_target_modules_subset():
    base = tiny("llama")
    m = get_tensorpeft_model(base, TensorPEFTConfig(target_modules=["q_proj", "v_proj"], atoms_per_group=1))
    roles = sorted({s.role for s in m.slots})
    assert roles == ["self_attn.q_proj", "self_attn.v_proj"]


def test_wrong_architecture_rejected(tmp_path):
    m = get_tensorpeft_model(tiny("llama"), TensorPEFTConfig(atoms_per_group=1))
    m.init_atoms(batches(1), verbose=False)
    m.save_pretrained(tmp_path)
    with pytest.raises(ValueError):
        TensorPEFTModel.from_pretrained(tiny("gpt2"), str(tmp_path))


def test_dict_batches_with_attention_mask_and_labels():
    base = tiny("llama").eval()
    m = get_tensorpeft_model(base, TensorPEFTConfig(atoms_per_group=1))
    x = batches(1)[0]
    m.init_atoms([{"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x}], verbose=False)
    assert m.is_initialized


def test_bf16_base():
    base = tiny("llama").to(torch.bfloat16).eval()
    m = get_tensorpeft_model(base, TensorPEFTConfig(atoms_per_group=2))
    m.init_atoms(batches(2), verbose=False)
    with torch.no_grad():
        for g in m.groups:
            g.alpha_raw.add_(0.01)
    out = m(batches(1)[0]).logits
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()
    assert all(p.dtype == torch.float32 for p in m.parameters() if p.requires_grad)


def test_one_adam_lr_trains():
    base = tiny("gpt2").eval()
    m = get_tensorpeft_model(base, TensorPEFTConfig(atoms_per_group=2))
    xs = batches(2, b=4, t=33)
    m.init_atoms(xs, verbose=False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-3, weight_decay=0.0)
    l0 = float(lm_loss(m, xs[0]))
    for _ in range(10):
        opt.zero_grad()
        lm_loss(m, xs[0]).backward()
        opt.step()
    assert float(lm_loss(m, xs[0])) < l0


def test_hf_trainer_integration(tmp_path):
    from tensorpeft import TensorPEFTTrainer

    base = tiny("llama")
    m = get_tensorpeft_model(base, TensorPEFTConfig(atoms_per_group=2))
    data = [{"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x.clone()}
            for x in torch.randint(0, V, (16, 24)).unbind(0)]
    m.init_atoms(torch.utils.data.DataLoader(data, batch_size=4), verbose=False)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(data)

        def __getitem__(self, i):
            return data[i]

    args = T.TrainingArguments(output_dir=str(tmp_path / "out"), per_device_train_batch_size=4, max_steps=6,
                               learning_rate=3e-3, warmup_steps=1, logging_steps=1, save_steps=3,
                               report_to=[], use_cpu=True)
    tr = TensorPEFTTrainer(model=m, args=args, train_dataset=DS())
    hist = tr.train()
    assert math.isfinite(hist.training_loss)
    tr.save_model(str(tmp_path / "final"))
    re = TensorPEFTModel.from_pretrained(tiny("llama"), str(tmp_path / "final"))
    assert re.is_initialized
    import os
    assert os.path.exists(tmp_path / "out" / "checkpoint-3" / "adapter_model.safetensors")
