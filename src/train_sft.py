# train_sft.py
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tokenizers import Tokenizer

from src.config import ModelArgs
from src.model import Transformer
from src.losses import router_loss


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "synthetic_data_generation"
META_PATH = ROOT / "data" / "shards" / "meta.json"
TOK_PATH = ROOT / "tokenizer.json"
INSTRUCT_PATH = ROOT / "prompts" / "instruct_prompts.txt"

SFT_TRAIN_PATH = DATA_DIR / "sft_train.jsonl"
SFT_VAL_PATH   = DATA_DIR / "sft_val.jsonl"

PRETRAIN_CKPT = ROOT / "artifacts" / "pretraining" / "ckpt_best.pt"
OUT_DIR = ROOT / "artifacts" / "sft"

def load_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]
    
def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

@torch.no_grad()
def gen_5_from_val(model, tok: Tokenizer, val_rows: list[dict], bos_id: int, eos_id: int, device_toggle: str, rng: np.random.Generator):
    model.eval()
    idxs = rng.choice(len(val_rows), size=5, replace=False)
    for i, j in enumerate(idxs):
        p = str(val_rows[int(j)]["prompt"])
        ids = [bos_id] + tok.encode(p).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device_toggle)
        out = model.generate(
            idx,
            max_new_tokens=128,
            temperature=0.8,
            use_kv_cache=True,
            eos_id=eos_id,
            stop_on_eos=True,
        )[0].tolist()
        text = tok.decode(out[len(ids):])
        print(f"\n--- VAL SAMPLE {i+1}/5 ---")
        print("PROMPT:", p)
        print("GEN:", text)
    model.train()


def build_sequences(rows, tok: Tokenizer, bos_id: int, eos_id: int, max_len: int):
    seqs = []
    for r in rows:
        prompt = str(r["prompt"])
        resp   = str(r["response"])

        prompt_ids = tok.encode(prompt).ids
        full_ids   = tok.encode(prompt + resp).ids

        seq = [bos_id] + full_ids + [eos_id]

        if len(seq) > max_len:
            seq = seq[:max_len]
            seq[-1] = eos_id

        prompt_len = 1 + len(prompt_ids)
        prompt_len = min(prompt_len, len(seq))

        seqs.append((seq, prompt_len))
    return seqs


def make_batch(seqs, batch_ids, pad_id: int, device: str):
    batch = [seqs[i] for i in batch_ids]
    B = len(batch)
    maxL = max(len(s[0]) for s in batch)

    x = torch.full((B, maxL - 1), pad_id, dtype=torch.long)
    y = torch.full((B, maxL - 1), -100, dtype=torch.long)  # ignore by default

    for i, (seq, prompt_len) in enumerate(batch):
        seq_t = torch.tensor(seq, dtype=torch.long)
        L = seq_t.numel()

        xi = seq_t[:-1]
        yi = seq_t[1:]

        x[i, : L - 1] = xi

        # mask prompt targets (targets within prompt region are indices < prompt_len-1 in y)
        yi = yi.clone()
        cut = max(0, min(prompt_len - 1, yi.numel()))
        yi[:cut] = -100

        y[i, : L - 1] = yi

    return x.to(device), y.to(device)


@torch.no_grad()
def eval_epoch(model, seqs, args: ModelArgs, pad_id: int, rng: np.random.Generator):
    model.eval()
    order = np.arange(len(seqs))
    total_loss = 0.0
    total_tokens = 0

    for start in range(0, len(order), args.batch_size):
        batch_ids = order[start : start + args.batch_size]
        x, y = make_batch(seqs, batch_ids, pad_id=pad_id, device=args.device)

        logits, router_logits = model(x)
        vocab = logits.size(-1)

        ce = F.cross_entropy(
            logits.reshape(-1, vocab),
            y.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n = (y != -100).sum().item()

        total_loss += float(ce.item())
        total_tokens += int(n)

    model.train()
    return total_loss / max(1, total_tokens)


def save_ckpt(path: Path, model, optimizer, step: int, val_loss: float, args: ModelArgs):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "val_loss": float(val_loss),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def main():
    args = ModelArgs()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # user decisions
    args.max_seq_len = 512
    # args.batch_size = 64
    args.batch_size = 32

    # SFT-specific
    lr = 3e-5
    alpha_lb = 0.5 * args.alpha_lb
    beta_z = 0.5 * args.beta_z

    max_epochs = 30      
    patience = 4      
    min_delta = 0.0         

    best_val = float("inf")
    bad_epochs = 0

    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    bos_id = int(meta["special_token_ids"]["[BOS]"])
    eos_id = int(meta["special_token_ids"]["[EOS]"])
    pad_id = int(meta["special_token_ids"]["[PAD]"])

    tok = Tokenizer.from_file(str(TOK_PATH))

    train_rows = load_jsonl(SFT_TRAIN_PATH)
    val_rows = load_jsonl(SFT_VAL_PATH)

    train_seqs = build_sequences(train_rows, tok, bos_id, eos_id, args.max_seq_len)
    val_seqs = build_sequences(val_rows, tok, bos_id, eos_id, args.max_seq_len)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    model = Transformer(args).to(args.device)
    decay_params = []
    nodecay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "token_embeddings.weight":
            nodecay_params.append(p)
        elif p.dim() < 2:
            # biases + RMSNorm weights
            nodecay_params.append(p)
        elif "router" in name:
            # optional but recommended for MoE stability
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(args.beta1, args.beta2),
    )

    # load pretrained
    ckpt = torch.load(PRETRAIN_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)

    best_val = float("inf")
    step = 0

    print("Starting the training...")

    for epoch in range(max_epochs):
        order = np.arange(len(train_seqs))
        rng.shuffle(order)

        print(f"\nStarting epoch {epoch} ...\n===========================================================")
        for start in range(0, len(order), args.batch_size):
            batch_ids = order[start : start + args.batch_size]
            x, y = make_batch(train_seqs, batch_ids, pad_id=pad_id, device=args.device)

            logits, router_logits = model(x)
            vocab = logits.size(-1)

            ce = F.cross_entropy(
                logits.reshape(-1, vocab),
                y.reshape(-1),
                ignore_index=-100,
            )
            llb, lz = router_loss(router_logits, args, debug_flag=False)
            loss = ce + alpha_lb * llb + beta_z * lz

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            step += 1
            print(f"Step {step} done!")

        val_loss = eval_epoch(model, val_seqs, args, pad_id=pad_id, rng=rng)

        gen_5_from_val(model, tok, val_rows, bos_id, eos_id, args.device, rng)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        save_ckpt(OUT_DIR / "ckpt_latest.pt", model, optimizer, step, val_loss, args)

        with (OUT_DIR / "log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "step": step, "val_loss": val_loss}, ensure_ascii=False) + "\n")

        improved = (best_val - val_loss) > min_delta
        if improved:
            best_val = val_loss
            bad_epochs = 0
            save_ckpt(OUT_DIR / "ckpt_best.pt", model, optimizer, step, val_loss, args)
        else:
            bad_epochs += 1

        print(f"epoch={epoch} step={step} val_loss={val_loss:.6f} best={best_val:.6f} bad_epochs={bad_epochs}")

        if bad_epochs >= patience:
            print(f"Early stopping: no val improvement for {bad_epochs} epochs (patience={patience})")
            break


if __name__ == "__main__":
    main()