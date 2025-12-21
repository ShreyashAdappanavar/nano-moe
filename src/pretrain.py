import torch
from .config import ModelArgs
from .model import Transformer
from .dataset import get_batch
from .losses import compute_loss, ce_loss_fn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import os
from torch.optim import AdamW
import json
from pathlib import Path
import matplotlib.pyplot as plt
import time
import shutil
from tokenizers import Tokenizer


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def atomic_replace(src: Path, dst: Path):
    try:
        os.replace(src, dst)  # atomic if same filesystem
    except PermissionError:
        # fallback: remove then replace
        if dst.exists():
            dst.unlink()
        os.replace(src, dst)

def save_ckpt(path: Path, step: int, model, optimizer, rng, val_rng, args):
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": rng.bit_generator.state,
        "numpy_val_rng_state": val_rng.bit_generator.state,
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    numbered = path / f"ckpt_step_{step:06d}.pt"
    torch.save(ckpt, numbered)

    tmp = path / "ckpt_latest_tmp.pt"
    torch.save(ckpt, tmp)
    atomic_replace(tmp, path / "ckpt_latest.pt")


ROOT = Path(__file__).resolve().parent.parent  # nano-moe/
META_PATH = ROOT / "data" / "shards" / "meta.json"
tok_path = str(ROOT / "tokenizer.json")

tok = Tokenizer.from_file(tok_path)

with META_PATH.open("r", encoding="utf-8") as f:
    meta = json.load(f)

args = ModelArgs()
assert args.vocab_size == meta["vocab_size"], "vocab_size in args and meta.json do not match"

torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

out_dir = "pretrain_" + args.out_dir
out_path = ROOT / out_dir
os.makedirs(out_path, exist_ok=True)
with (out_path / "config.json").open("w", encoding="utf-8") as f:
    json.dump(vars(args), f, indent=2)

train_log_path = out_path / "train_log.jsonl"
val_log_path   = out_path / "val_log.jsonl"
load_log_path  = out_path / "load_log.jsonl"


rng = np.random.default_rng(args.seed)
val_rng = np.random.default_rng(args.seed + 10)

model = Transformer(args).to(args.device)
optimizer = AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
    betas=(args.beta1, args.beta2),
)

latest_pt = out_path / "ckpt_latest.pt"
if latest_pt.exists():
    ckpt = torch.load(latest_pt, map_location="cpu")  # IMPORTANT: keep RNG state on CPU
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in ckpt: torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
    rng.bit_generator.state = ckpt["numpy_rng_state"]
    val_rng.bit_generator.state = ckpt["numpy_val_rng_state"]
    start_step = int(ckpt["step"]) + 1
    print(f"Resumed from step {int(ckpt['step'])}")
    model.to(args.device)  # move back to GPU after loading from CPU
    train_recs = load_jsonl(train_log_path)
    val_recs   = load_jsonl(val_log_path)
    load_recs  = load_jsonl(load_log_path)

else:
    start_step = 0



train_steps = []
train_ce = []
train_total = []
curr_lr = []

val_steps = []
val_ce = []
load_steps = []
load_max_by_layer = [] 
train_tps = []
train_mem_mb = []

best_val_ce = float("inf")
best_step = -1

model.train()

training_ran = False 

for step in range(start_step, args.max_steps):
# for step in range(start_step, start_step+30):

    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    training_ran = True
    x, y = get_batch(args, split="train", rng=rng)
    logits, router_logits = model(x)

    loss, logs = compute_loss(logits, router_logits, y, args, debug_flag=False)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = clip_grad_norm_(model.parameters(), args.grad_clip_norm)

    if step < args.warmup_steps:
        lr_t = args.lr * (step + 1) / (args.warmup_steps) 
    else: lr_t = args.lr

    for group in optimizer.param_groups:
        group['lr'] = lr_t

    optimizer.step()

    dt = time.perf_counter() - t0
    tokens_per_step = args.batch_size * args.max_seq_len
    tps = tokens_per_step / max(dt, 1e-9)

    if step % args.log_interval == 0:
        train_steps.append(step)
        train_ce.append(logs["ce_loss"])
        train_total.append(loss.item())
        curr_lr.append(lr_t)
        train_tps.append(tps)
        if torch.cuda.is_available():
            train_mem_mb.append(torch.cuda.max_memory_allocated() / (1024**2))
        else:
            train_mem_mb.append(0.0)
        
        print(
            f"STEP: {step} | Loss: {loss.item():.3f} | CE_Loss: {logs['ce_loss']:.3f} | "
            f"Load_Balancing Loss: {logs['Llb_final']:.3f} | Z_Loss: {logs['Lz_final']:.3f} | "
            f"Grad_Norm: {float(grad):.3f} | LR: {lr_t:.2e} | TPS: {tps:.0f} | MEM(MB): {train_mem_mb[-1]:.0f}"            
        )

        append_jsonl(train_log_path, {
        "step": step,
        "loss": float(loss.item()),
        "ce": float(logs["ce_loss"]),
        "llb": float(logs["Llb_final"]),
        "lz": float(logs["Lz_final"]),
        "grad": float(grad),
        "lr": float(lr_t),
        "tps": float(tps),
        "mem_mb": float(train_mem_mb[-1]) if torch.cuda.is_available() else 0.0,
    })

    if step % args.eval_interval == 0:
        print("========== EVAL STARTED ==========")
        val_rng = np.random.default_rng(args.seed + 10)
        with torch.no_grad():
            _, dbg = compute_loss(logits, router_logits, y, args, debug_flag=True)
        all_loads = dbg["all_loads"]
        max_load_per_layer = all_loads.max(dim=-1).values.detach().float().cpu().numpy()
        load_steps.append(step)
        load_max_by_layer.append(max_load_per_layer)
        print(f"STEP: {step} | max(load) per layer: {max_load_per_layer}")
        append_jsonl(load_log_path, {
        "step": step,
        "max_load_per_layer": max_load_per_layer.tolist()})

        model.eval()
        mean_ce = 0.0
        with torch.no_grad():
            for _ in range(args.eval_batches):
                xv, yv = get_batch(args, split="val", rng=val_rng)
                logits_v, _ = model(xv)
                mean_ce += ce_loss_fn(logits_v, yv, args).item()
        mean_ce /= args.eval_batches
        append_jsonl(val_log_path, {
        "step": step,
        "val_ce": float(mean_ce),
        })
        val_steps.append(step)
        val_ce.append(mean_ce)
        print(f"STEP: {step} | VAL CE: {mean_ce:.3f}")
        if mean_ce < best_val_ce:
            best_val_ce = mean_ce
            best_step = step
            save_ckpt(out_path, step, model, optimizer, rng, val_rng, args)
            shutil.copyfile(out_path / "ckpt_latest.pt", out_path / "ckpt_best.pt")
            print(f"STEP: {step} | NEW BEST VAL CE: {best_val_ce:.3f} (saved ckpt_best.pt)")
        model.train()

    if step % args.ckpt_interval == 0 and step > 0:
        save_ckpt(out_path, step, model, optimizer, rng, val_rng, args)

if training_ran: save_ckpt(out_path, step, model, optimizer, rng, val_rng, args)

print("\n" + "="*80)
print("FINAL MODEL SANITY CHECK: 5 GENERATIONS")
print("="*80)

for i in range(5):
    rng_local = np.random.default_rng(args.seed + 10 + i)

    with torch.no_grad():
        x, _ = get_batch(args, split="val", rng=rng_local)
        prompt = x[:1, :64]
        out = model.generate(prompt, max_new_tokens=64, temperature=0.8, use_kv_cache=True)

    prompt_ids = prompt[0].tolist()
    out_ids = out[0].tolist()

    prompt_text = tok.decode(prompt_ids)
    gen_text = tok.decode(out_ids[len(prompt_ids):])

    print("\n" + "-"*80)
    print(f"SAMPLE {i+1}/5 | prompt_tokens=64 | gen_tokens=64 | temp=0.8 | kv_cache=on")
    print("-"*80)
    print("PROMPT:")
    print(prompt_text)
    print("\nGENERATION:")
    print(gen_text)
print("\n" + "="*80)


train_recs = load_jsonl(train_log_path)
val_recs   = load_jsonl(val_log_path)
load_recs  = load_jsonl(load_log_path)

# de-dup by step (keep last)
train_map = {r["step"]: r for r in train_recs}
val_map   = {r["step"]: r for r in val_recs}

train_steps = sorted(train_map.keys())
train_ce    = [train_map[s]["ce"] for s in train_steps]
train_total = [train_map[s]["loss"] for s in train_steps]

val_steps = sorted(val_map.keys())
val_ce    = [val_map[s]["val_ce"] for s in val_steps]

curr_lr     = [train_map[s].get("lr", 0.0) for s in train_steps]
train_tps   = [train_map[s].get("tps", 0.0) for s in train_steps]
train_mem_mb = [train_map[s].get("mem_mb", 0.0) for s in train_steps]

load_map = {r["step"]: r for r in load_recs}  # de-dup by step (keep last)
load_steps = sorted(load_map.keys())

plt.figure()
plt.plot(train_steps, train_ce, label="train_ce")
plt.plot(train_steps, train_total, label="train_total")
plt.plot(val_steps, val_ce, label="val_ce")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.savefig(out_path / "loss_curves.png", dpi=150)
plt.close()


plt.figure()
plt.plot(train_steps, curr_lr)
plt.xlabel("step")
plt.ylabel("lr")
plt.tight_layout()
plt.savefig(out_path / "lr_curve.png", dpi=150)
plt.close()


plt.figure()
plt.plot(train_steps, train_tps)
plt.xlabel("step")
plt.ylabel("tokens/sec")
plt.tight_layout()
plt.savefig(out_path / "throughput_tokens_per_sec.png", dpi=150)
plt.close()

if torch.cuda.is_available():
    plt.figure()
    plt.plot(train_steps, train_mem_mb)
    plt.xlabel("step")
    plt.ylabel("max_memory_allocated (MB)")
    plt.tight_layout()
    plt.savefig(out_path / "vram_max_allocated_mb.png", dpi=150)
    plt.close()

if len(load_steps) > 0:
    arr = np.stack([load_map[s]["max_load_per_layer"] for s in load_steps], axis=0)  # (n_points, n_layers)
    L = arr.shape[1]

    plt.figure()
    for li in range(L):
        plt.plot(load_steps, arr[:, li], label=f"layer{li}")
    plt.xlabel("step")
    plt.ylabel("max(load)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "max_load_by_layer.png", dpi=150)
    plt.close()

plt.figure()
plt.plot(train_steps, [train_map[s]["ce"] for s in train_steps],  label="ce")
plt.plot(train_steps, [train_map[s]["llb"] for s in train_steps], label="llb")
plt.plot(train_steps, [train_map[s]["lz"] for s in train_steps],  label="lz")
plt.xlabel("step")
plt.ylabel("loss_component")
plt.legend()
plt.tight_layout()
plt.savefig(out_path / "loss_components.png", dpi=150)
plt.close()

plt.figure()
plt.plot(train_steps, [train_map[s].get("grad", 0.0) for s in train_steps])
plt.xlabel("step")
plt.ylabel("grad_norm")
plt.tight_layout()
plt.savefig(out_path / "grad_norm.png", dpi=150)
plt.close()

import bisect
g_steps = []
g_vals = []
for vs in val_steps:
    i = bisect.bisect_right(train_steps, vs) - 1
    if i < 0:
        continue
    g_steps.append(vs)
    g_vals.append(val_map[vs]["val_ce"] - train_map[train_steps[i]]["ce"])

if len(g_steps) > 0:
    plt.figure()
    plt.plot(g_steps, g_vals)
    plt.xlabel("step")
    plt.ylabel("val_ce - train_ce")
    plt.tight_layout()
    plt.savefig(out_path / "generalization_gap.png", dpi=150)
    plt.close()
