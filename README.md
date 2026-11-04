# TensorPEFT

Parameter-efficient fine-tuning with tensor gradient-pursuit adapters, for Hugging Face causal LMs.

Equal-shaped weight matrices across all transformer blocks are stacked into one tensor per group
(`layers × roles × d_in × d_out`). The update to each tensor is a sum of rank-one CP atoms, chosen
from one gradient probe before training (HOPM with deflation). Coefficients start at zero, and all
coefficients and factors are then trained with any optimiser. A merged adapter has zero inference
overhead.

- Explanation with figures: [`docs/index.md`](docs/index.md)
- Live demo: https://huggingface.co/spaces/Angshul/TensorPEFT
- Hub repository: https://huggingface.co/Angshul/TensorPEFT

## Install

```
pip install .                      # from a clone of this repository
pip install git+https://huggingface.co/Angshul/TensorPEFT
```

Requires `torch>=2.1`, `transformers>=4.40`, `safetensors` and `huggingface_hub`.

## Usage

```python
from transformers import AutoModelForCausalLM
from tensorpeft import TensorPEFTConfig, get_tensorpeft_model, TensorPEFTModel

model = get_tensorpeft_model(AutoModelForCausalLM.from_pretrained("gpt2"),
                             TensorPEFTConfig(atoms_per_group=1))
model.init_atoms(train_dataloader, n_batches=4)   # gradient probe -> atoms
model.print_trainable_parameters()                # 9,259

# train: AdamW, lr 3e-3, ~10 warm-up steps, weight_decay 0 (or tensorpeft.TensorPEFTTrainer)

model.save_pretrained("adapter")                  # adapter_config.json + adapter_model.safetensors
model = TensorPEFTModel.from_pretrained(AutoModelForCausalLM.from_pretrained("gpt2"), "adapter")
merged = model.merge_and_unload()
```

| `TensorPEFTConfig` | default | |
|---|---|---|
| `atoms_per_group` | 4 | atoms per tensor group (sets the budget) |
| `target_modules` | `"all-linear"` | or name suffixes, e.g. `["q_proj", "v_proj"]` |
| `exclude_modules` | lm_head, embeddings | |
| `split_fused_qkv` | True | GPT-2 `c_attn` → q, k, v |
| `init_batches` | 4 | probe batches |
| `hopm_iters` | 30 | power-iteration sweeps per atom |
| `alpha_scale` | 33.3 | coefficient scaling so that one learning rate fits all parameters |

GPT-2 small budgets: 1 atom per group = 9,259 parameters, 4 = 37,036, 6 = 55,554.

## Results (GPT-2 124M → Bengali news, 300 steps, validation loss, lower is better)

| params | TensorPEFT | best baseline | Δ (3 seeds) | p |
|---|---:|---|---:|---:|
| ≈9k | 1.3245 | prompt tuning 1.3684 | −0.044 | 0.008 |
| ≈37k | 1.2558 | LayerNorm tuning 1.2504 | +0.005 | 0.37 |
| ≈55k | 1.2368 | (IA)³ 1.2880 | −0.051 | 0.026 |

Single-seed comparisons also cover LoRA, QLoRA, VeRA and BitFit, plus a ≈1.5k tier. At ≥115k
parameters LoRA was better. All numbers are in `benchmarks/results.json`.

## Repository

```
src/tensorpeft/      package (config, layer discovery, HOPM, model, Trainer)
tests/               pytest: GPT-2, Llama, Qwen2, Mistral, Phi, Gemma (CPU, tiny random models)
examples/            prepare_bengali_data.py, reproduce_gpt2_bengali.py, train_with_trainer.py
benchmarks/          results.json, atoms_gpt2_bengali.json, race_9k.json
scripts/             make_figures.py (renders docs/figures from benchmarks/)
docs/                GitHub Pages article (index.md + figures)
```

Reproduce a benchmark row:

```
python examples/prepare_bengali_data.py --out data
python examples/reproduce_gpt2_bengali.py --model gpt2 --data data --atoms-per-group 1
```

Tests: `pip install .[test] && pytest tests`

## Notes

- Probe memory is about 2× the size of the targeted weights, briefly. Restrict `target_modules` for large models.
- Quantised (bitsandbytes) bases are not supported in this version.
- Tested on CPU, and on CUDA with bf16 in Colab.
