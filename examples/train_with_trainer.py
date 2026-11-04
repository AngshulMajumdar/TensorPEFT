"""Fine-tune any Hugging Face causal LM on a text file with TensorPEFT and the HF Trainer.

  python train_with_trainer.py --model Qwen/Qwen2.5-0.5B --text corpus.txt --out tp_adapter \
      --atoms-per-group 6 --steps 300
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments

from tensorpeft import TensorPEFTConfig, TensorPEFTTrainer, get_tensorpeft_model

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--text", required=True, help="plain-text training file")
p.add_argument("--out", required=True)
p.add_argument("--atoms-per-group", type=int, default=4)
p.add_argument("--seq", type=int, default=256)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--bf16", action="store_true")
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if a.bf16 else torch.float32)

ids = tok(open(a.text, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
n = len(ids) // a.seq
blocks = ids[: n * a.seq].view(n, a.seq)
train = [{"input_ids": b} for b in blocks]

model = get_tensorpeft_model(model, TensorPEFTConfig(atoms_per_group=a.atoms_per_group))
collate = DataCollatorForLanguageModeling(tok, mlm=False)
loader = torch.utils.data.DataLoader(train, batch_size=a.batch, shuffle=True, collate_fn=collate)
model.init_atoms(loader, n_batches=4)          # the one extra step compared with LoRA
model.print_trainable_parameters()

args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=a.batch, max_steps=a.steps,
                         learning_rate=a.lr, warmup_steps=10, lr_scheduler_type="constant_with_warmup",
                         weight_decay=0.0, logging_steps=20, save_steps=100, bf16=a.bf16, report_to=[])
trainer = TensorPEFTTrainer(model=model, args=args, train_dataset=train, data_collator=collate)
trainer.train()
model.save_pretrained(a.out)          # adapter only (tens of KB for small budgets)
# model.push_to_hub("your-name/your-adapter")
