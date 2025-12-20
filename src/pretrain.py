import torch
from .config import ModelArgs
from .model import Transformer
from .dataset import get_batch
from .losses import compute_loss
from torch.nn.utils import clip_grad_norm_
import numpy as np
import os
from torch.optim import AdamW
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  # nano-moe/

args = ModelArgs()

torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

out_dir = "pretrain_" + args.out_dir
out_path = ROOT / out_dir
os.makedirs(out_path, exist_ok=True)
with (out_path / "config.json").open("w", encoding="utf-8") as f:
    json.dump(vars(args), f, indent=2)

rng = np.random.default_rng(args.seed)

model = Transformer(args).to(args.device)
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))

model.train()

for step in range(args.max_steps):
    x, y = get_batch(args, split='train', rng=rng)
    logits, router_logits = model(x)    
    loss, logs = compute_loss(logits, router_logits, y, args, debug_flag=False)
    optimizer.zero_grad(set_to_none=True)
    loss.backward() 
    grad = clip_grad_norm_(model.parameters(), args.grad_clip_norm)
    optimizer.step()

    if step%args.log_interval == 0:
        print(f"STEP: {step} | Loss: {loss.item():.3f} | CE_Loss: {logs['ce_loss']:.3f} | Load_Balancing Loss: {logs['Llb_final']:.3f} | Z_Loss: {logs['Lz_final']:.3f} | Grad_Norm: {float(grad):.3f}")
    
