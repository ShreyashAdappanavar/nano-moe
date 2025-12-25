# eval.py
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tokenizers import Tokenizer

from src.config import ModelArgs
from src.model import Transformer
from src.dataset import get_batch
from src.losses import ce_loss_fn

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parent  # nano-moe/


def read_prompts(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [s.rstrip() for s in lines if s.strip()]
    return lines


def router_summaries(router_logits, args: ModelArgs):
    # router_logits: List[(B,T,E)] length = n_layers
    entropies = []
    max_loads = []
    loads_all = []
    importances_all = []
    usage_counts_all = []

    for layer_logits in router_logits:
        p = F.softmax(layer_logits, dim=-1)  # (B,T,E)
        ent = -(p * (p.clamp_min(1e-9).log())).sum(dim=-1)  # (B,T)
        entropies.append(ent.mean().item())

        topk = torch.topk(p, args.top_k, dim=-1).indices  # (B,T,K)
        B, T, K = topk.shape
        counts = torch.bincount(topk.reshape(B * T * K), minlength=args.num_experts).to(p.dtype)
        usage = counts / (B * T * K)
        usage_counts_all.append(counts.detach().cpu().numpy())
        loads_all.append(usage.detach().cpu().numpy())
        max_loads.append(float(usage.max().item()))

        importances = p.mean(dim=(0, 1))  # (E,)
        importances_all.append(importances.detach().cpu().numpy())

    usage_counts_all = np.stack(usage_counts_all, axis=0)  # (L,E)
    loads_all = np.stack(loads_all, axis=0)                # (L,E)
    importances_all = np.stack(importances_all, axis=0)    # (L,E)

    out = {
        "router_entropy_mean_per_layer": entropies,             # (L,)
        "max_load_per_layer": max_loads,                        # (L,)
        "load_frac_per_layer": loads_all.tolist(),              # (L,E)
        "importance_per_layer": importances_all.tolist(),       # (L,E)
        "usage_counts_per_layer": usage_counts_all.tolist(),    # (L,E)
    }
    return out


@torch.no_grad()
def measure_throughput(model: Transformer, tok: Tokenizer, args: ModelArgs, bos_id: int, eos_id: int,
                       prompt_text: str, prompt_len: int, max_new_tokens: int, use_kv_cache: bool, iters: int):
    ids = tok.encode(prompt_text).ids
    ids = [bos_id] + ids

    if len(ids) < prompt_len:
        pad = ids[-1] if len(ids) > 0 else bos_id
        ids = ids + [pad] * (prompt_len - len(ids))
    else:
        ids = ids[:prompt_len]

    idx = torch.tensor([ids], device=args.device, dtype=torch.long)

    # warmup
    _ = model.generate(
        idx.clone(),
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        use_kv_cache=use_kv_cache,
        eos_id=eos_id,
        stop_on_eos=False,
    )
    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.synchronize()

    times = []
    gen_tokens = []

    for _ in range(iters):
        if torch.cuda.is_available() and args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            idx.clone(),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            use_kv_cache=use_kv_cache,
            eos_id=eos_id,
            stop_on_eos=False,
        )
        if torch.cuda.is_available() and args.device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        gen_tokens.append(int(out.shape[1] - idx.shape[1]))

    t_mean = float(np.mean(times))
    tokens_mean = float(np.mean(gen_tokens))
    tps = tokens_mean / max(t_mean, 1e-9)

    return {
        "use_kv_cache": bool(use_kv_cache),
        "iters": int(iters),
        "prompt_len": int(prompt_len),
        "max_new_tokens": int(max_new_tokens),
        "gen_tokens_mean": tokens_mean,
        "time_sec_mean": t_mean,
        "tokens_per_sec": float(tps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str)
    ap.add_argument("--ckpt", type=str)
    ap.add_argument("--prompt_set", type=str, default="prompts/instruct_prompts.txt")
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--throughput_iters", type=int, default=5)
    args_cli = ap.parse_args()

    art_dir = ROOT / "artifacts" / args_cli.tag
    out_dir = art_dir / "eval_script_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = art_dir / "config.json"
    ckpt_path = art_dir / args_cli.ckpt
    tok_path = art_dir / "tokenizer.json"
    meta_path = art_dir / "meta.json"

    if not cfg_path.exists(): raise FileNotFoundError(cfg_path)
    if not ckpt_path.exists(): raise FileNotFoundError(ckpt_path)
    if not tok_path.exists(): raise FileNotFoundError(tok_path)
    if not meta_path.exists(): raise FileNotFoundError(meta_path)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    args = ModelArgs(**cfg)
    args.batch_size = 8

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bos_id = int(meta["special_token_ids"]["[BOS]"])
    eos_id = int(meta["special_token_ids"]["[EOS]"])

    tok = Tokenizer.from_file(str(tok_path))

    model = Transformer(args).to(args.device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rng = np.random.default_rng(args.seed + 999)
    val_rng = np.random.default_rng(args.seed + 10)

    print("Starting...")

    # val CE + ppl
    mean_ce = 0.0
    for _ in range(args.eval_batches):
        xv, yv = get_batch(args, split="val", rng=val_rng)
        logits_v, _ = model(xv)
        mean_ce += ce_loss_fn(logits_v, yv, args).item()
    mean_ce /= args.eval_batches
    ppl = float(np.exp(mean_ce))

    # routing summaries on one fresh val batch
    xv_dbg, _ = get_batch(args, split="val", rng=rng)
    _, router_logits_dbg = model(xv_dbg)
    routing = router_summaries(router_logits_dbg, args)

    # throughput on one fixed prompt
    prompt_path = ROOT / args_cli.prompt_set
    prompts = read_prompts(prompt_path)
    if len(prompts) == 0:
        raise ValueError(f"no prompts in {prompt_path}")
    prompt_text = prompts[0]

    thr_off = measure_throughput(
        model, tok, args, bos_id, eos_id,
        prompt_text=prompt_text,
        prompt_len=args_cli.prompt_len,
        max_new_tokens=args_cli.max_new_tokens,
        use_kv_cache=False,
        iters=args_cli.throughput_iters,
    )

    thr_on = measure_throughput(
        model, tok, args, bos_id, eos_id,
        prompt_text=prompt_text,
        prompt_len=args_cli.prompt_len,
        max_new_tokens=args_cli.max_new_tokens,
        use_kv_cache=True,
        iters=args_cli.throughput_iters,
    )

    out = {
        "tag": args_cli.tag,
        "ckpt": args_cli.ckpt,
        "device": args.device,
        "val": {
            "ce": float(mean_ce),
            "ppl": float(ppl),
            "eval_batches": int(args.eval_batches),
            "batch_size": int(args.batch_size),
            "max_seq_len": int(args.max_seq_len),
        },
        "throughput": {
            "prompt_set": args_cli.prompt_set,
            "prompt_text": prompt_text,
            "off": thr_off,
            "on": thr_on,
        },
        "routing": routing,
    }

    (out_dir / "eval.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'eval.json'}")

    # plots
    # 1) throughput
    plt.figure()
    plt.bar([0, 1], [thr_off["tokens_per_sec"], thr_on["tokens_per_sec"]])
    plt.xticks([0, 1], ["kv_cache=off", "kv_cache=on"])
    plt.ylabel("tokens/sec")
    plt.tight_layout()
    plt.savefig(out_dir / "throughput_kv_cache.png", dpi=150)
    plt.close()

    # 2) router entropy per layer
    ent = routing["router_entropy_mean_per_layer"]
    plt.figure()
    plt.plot(list(range(len(ent))), ent)
    plt.xlabel("layer")
    plt.ylabel("mean router entropy")
    plt.tight_layout()
    plt.savefig(out_dir / "router_entropy_per_layer.png", dpi=150)
    plt.close()

    # 3) max load per layer
    ml = routing["max_load_per_layer"]
    plt.figure()
    plt.plot(list(range(len(ml))), ml)
    plt.xlabel("layer")
    plt.ylabel("max(load)")
    plt.tight_layout()
    plt.savefig(out_dir / "max_load_per_layer.png", dpi=150)
    plt.close()

    # 4) overall expert usage (sum over layers)
    usage_counts = np.array(routing["usage_counts_per_layer"], dtype=np.float64)  # (L,E)
    overall = usage_counts.sum(axis=0)
    overall = overall / max(1.0, overall.sum())

    plt.figure()
    plt.bar(list(range(len(overall))), overall.tolist())
    plt.xlabel("expert")
    plt.ylabel("usage fraction (overall)")
    plt.tight_layout()
    plt.savefig(out_dir / "expert_usage_overall.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    main()
