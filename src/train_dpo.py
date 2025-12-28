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

DPO_TRAIN_PATH = DATA_DIR / "dpo_train.jsonl"
DPO_VAL_PATH   = DATA_DIR / "dpo_val.jsonl"

SFT_CKPT = ROOT / "artifacts" / "sft" / "ckpt_best.pt"
OUT_DIR  = ROOT / "artifacts" / "dpo"


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dpo_seqs(rows, tok: Tokenizer, bos_id: int, eos_id: int, max_len: int):
    out = []
    skipped = 0

    limit = max_len - 10 

    for r in rows:
        prompt   = str(r["prompt"])
        chosen   = str(r["chosen"])
        rejected = str(r["rejected"])

        prompt_ids = tok.encode(prompt).ids
        if (1 + len(prompt_ids)) > limit:  
            skipped += 1
            print(f"[SKIP] prompt too long: tokens={1+len(prompt_ids)} limit={limit} id={r.get('id','?')}")
            continue

        prompt_len = 1 + len(prompt_ids)

        ch_full = tok.encode(prompt + chosen).ids
        rj_full = tok.encode(prompt + rejected).ids

        ch = [bos_id] + ch_full + [eos_id]
        rj = [bos_id] + rj_full + [eos_id]

        if len(ch) > max_len:
            ch = ch[:max_len]
            ch[-1] = eos_id
        if len(rj) > max_len:
            rj = rj[:max_len]
            rj[-1] = eos_id

        out.append((ch, prompt_len, rj, prompt_len))

    print(f"build_dpo_seqs: kept={len(out)} skipped={skipped} max_len={max_len} limit={limit}")
    return out



def _seq_to_xy(seq, prompt_len, pad_id):
    # x: seq[:-1]; y: seq[1:], with prompt targets masked to -100
    seq_t = torch.tensor(seq, dtype=torch.long)
    x = seq_t[:-1]
    y = seq_t[1:].clone()
    cut = max(0, min(prompt_len - 1, y.numel()))
    y[:cut] = -100
    return x, y


def make_batch(dpo_seqs, batch_ids, pad_id: int, device: str):
    # returns concatenated (2B, T) for chosen then rejected
    ch_xy = []
    rj_xy = []
    for i in batch_ids:
        ch, ch_plen, rj, rj_plen = dpo_seqs[i]
        ch_xy.append(_seq_to_xy(ch, ch_plen, pad_id))
        rj_xy.append(_seq_to_xy(rj, rj_plen, pad_id))

    all_xy = ch_xy + rj_xy
    B2 = len(all_xy)
    maxT = max(x.numel() for (x, _) in all_xy)

    x = torch.full((B2, maxT), pad_id, dtype=torch.long)
    y = torch.full((B2, maxT), -100, dtype=torch.long)

    for i, (xi, yi) in enumerate(all_xy):
        T = xi.numel()
        x[i, :T] = xi
        y[i, :T] = yi

    return x.to(device), y.to(device)


def seq_logp_from_logits(logits: torch.Tensor, targets: torch.Tensor):
    # logits: (B, T, V), targets: (B, T) with -100 ignored
    logp = F.log_softmax(logits, dim=-1)
    t = targets.clone()
    mask = (t != -100)
    t[~mask] = 0
    tok_logp = logp.gather(-1, t.unsqueeze(-1)).squeeze(-1)
    tok_logp = tok_logp * mask.to(tok_logp.dtype)
    return tok_logp.sum(dim=1)  # (B,)


@torch.no_grad()
def eval_epoch(policy, ref, dpo_seqs, args: ModelArgs, pad_id: int, beta: float):
    policy.eval()
    ref.eval()

    order = np.arange(len(dpo_seqs))
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for start in range(0, len(order), args.batch_size):
        batch_ids = order[start : start + args.batch_size]
        x, y = make_batch(dpo_seqs, batch_ids, pad_id=pad_id, device=args.device)

        logits_pi, _ = policy(x)
        logp_pi = seq_logp_from_logits(logits_pi, y)

        logits_ref, _ = ref(x)
        logp_ref = seq_logp_from_logits(logits_ref, y)

        B = len(batch_ids)
        pi_ch, pi_rj = logp_pi[:B], logp_pi[B:]
        rf_ch, rf_rj = logp_ref[:B], logp_ref[B:]

        d_pi = pi_ch - pi_rj
        d_rf = rf_ch - rf_rj
        d = d_pi - d_rf

        loss = (-F.logsigmoid(beta * d)).mean()
        acc = (d_pi > 0).float().mean()

        total_loss += float(loss.item())
        total_acc += float(acc.item())
        n_batches += 1

    policy.train()
    return total_loss / max(1, n_batches), total_acc / max(1, n_batches)


def save_ckpt(path: Path, policy, optimizer, step: int, val_loss: float, args: ModelArgs):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "val_loss": float(val_loss),
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def main():
    args = ModelArgs()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # keep close to train_sft.py decisions
    args.max_seq_len = 512
    # args.batch_size = 64
    args.batch_size = 32

    # DPO specifics
    lr = 3e-5
    beta = 0.1
    alpha_lb = 0.5 * args.alpha_lb
    beta_z   = 0.5 * args.beta_z

    max_epochs = 15
    patience = 3
    min_delta = 0.0

    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    bos_id = int(meta["special_token_ids"]["[BOS]"])
    eos_id = int(meta["special_token_ids"]["[EOS]"])
    pad_id = int(meta["special_token_ids"]["[PAD]"])

    tok = Tokenizer.from_file(str(TOK_PATH))

    train_rows = load_jsonl(DPO_TRAIN_PATH)
    val_rows   = load_jsonl(DPO_VAL_PATH)

    train_seqs = build_dpo_seqs(train_rows, tok, bos_id, eos_id, args.max_seq_len)
    val_seqs   = build_dpo_seqs(val_rows,   tok, bos_id, eos_id, args.max_seq_len)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    # policy + reference
    policy = Transformer(args).to(args.device)
    ref    = Transformer(args).to(args.device)

    ckpt = torch.load(SFT_CKPT, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["model"])
    ref.load_state_dict(ckpt["model"])
    policy.to(args.device)
    ref.to(args.device)
    for p in ref.parameters():
        p.requires_grad = False

    # optimizer (same param grouping as pretrain)
    decay_params = []
    nodecay_params = []
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        if name == "token_embeddings.weight":
            nodecay_params.append(p)
        elif p.dim() < 2:
            nodecay_params.append(p)
        elif "router" in name:
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

    best_val = float("inf")
    bad_epochs = 0
    step = 0

    print("Starting DPO training...")

    for epoch in range(max_epochs):
        order = np.arange(len(train_seqs))
        rng.shuffle(order)

        print(f"\nStarting epoch {epoch} ...\n===========================================================")
        for start in range(0, len(order), args.batch_size):
            batch_ids = order[start : start + args.batch_size]
            x, y = make_batch(train_seqs, batch_ids, pad_id=pad_id, device=args.device)

            # policy
            logits_pi, router_logits = policy(x)
            logp_pi = seq_logp_from_logits(logits_pi, y)

            # reference (no grad)
            with torch.no_grad():
                logits_ref, _ = ref(x)
                logp_ref = seq_logp_from_logits(logits_ref, y)

            B = len(batch_ids)
            pi_ch, pi_rj = logp_pi[:B], logp_pi[B:]
            rf_ch, rf_rj = logp_ref[:B], logp_ref[B:]

            d_pi = pi_ch - pi_rj
            d_rf = rf_ch - rf_rj
            d = d_pi - d_rf

            dpo_loss = (-F.logsigmoid(beta * d)).mean()

            llb, lz = router_loss(router_logits, args, debug_flag=False)
            loss = dpo_loss + alpha_lb * llb + beta_z * lz

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(policy.parameters(), args.grad_clip_norm)
            optimizer.step()

            step += 1
            print(f"Step {step} done!")

        val_loss, val_acc = eval_epoch(policy, ref, val_seqs, args, pad_id=pad_id, beta=beta)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        save_ckpt(OUT_DIR / "ckpt_latest.pt", policy, optimizer, step, val_loss, args)

        improved = (best_val - val_loss) > min_delta
        if improved:
            best_val = val_loss
            bad_epochs = 0
            save_ckpt(OUT_DIR / "ckpt_best.pt", policy, optimizer, step, val_loss, args)
        else:
            bad_epochs += 1

        with (OUT_DIR / "log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "step": step,
                "val_loss": val_loss,
                "val_pref_acc": val_acc,
                "bad_epochs": bad_epochs,
            }, ensure_ascii=False) + "\n")

        print(f"epoch={epoch} step={step} val_loss={val_loss:.6f} best={best_val:.6f} val_pref_acc={val_acc:.3f} bad_epochs={bad_epochs}")

        if bad_epochs >= patience:
            print(f"Early stopping: no val improvement for {bad_epochs} epochs (patience={patience})")
            break


if __name__ == "__main__":
    main()
