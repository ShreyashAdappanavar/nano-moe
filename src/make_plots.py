import json
import time
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import torch
from tokenizers import Tokenizer
import matplotlib.pyplot as plt


# -------------------- repo root --------------------
def _find_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "src" / "model.py").exists():
            return p
    raise RuntimeError("Could not locate repo root (expected ./src/model.py)")


ROOT = _find_root()
sys.path.insert(0, str(ROOT))

from src.config import ModelArgs
from src.model import Transformer


# -------------------- paths --------------------
CHECKPOINTS_ALL = {
    "Pretrain": {
        "ckpt": ROOT / "artifacts" / "pretraining" / "ckpt_best.pt",
        "cfg":  ROOT / "artifacts" / "pretraining" / "ui_config.json",
        "train_log": ROOT / "artifacts" / "pretraining" / "train_log.jsonl",
        "val_log":   ROOT / "artifacts" / "pretraining" / "val_log.jsonl",
    },
    "SFT": {
        "ckpt": ROOT / "artifacts" / "sft" / "ckpt_best.pt",
        "cfg":  ROOT / "artifacts" / "sft" / "ui_config.json",
    },
    "DPO": {
        "ckpt": ROOT / "artifacts" / "dpo" / "ckpt_best.pt",
        "cfg":  ROOT / "artifacts" / "dpo" / "ui_config.json",
    },
}
CHECKPOINTS = {k: v for k, v in CHECKPOINTS_ALL.items() if v["ckpt"].exists() and v["cfg"].exists()}

TOKENIZER_PATH = ROOT / "tokenizer.json"
META_PATH = ROOT / "data" / "shards" / "meta.json"
PROMPTS_PATH = ROOT / "prompts" / "instruct_prompts.txt"  # optional
OUT_DIR = ROOT / "reports" / "figs"


# -------------------- utils --------------------
def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _exp_name(e: int) -> str:
    return f"Exp{e+1}"

def _load_prompts() -> list[str]:
    if PROMPTS_PATH.exists():
        lines = [l.strip() for l in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()]
        lines = [l for l in lines if l]
        if lines:
            return lines
    return [
        "Write a short children's story about a brave teapot.",
        "Write a story about a curious rabbit who finds a small key.",
        "Write a story about teamwork involving a kite and a red balloon.",
        "Write a story about kindness on a rainy day.",
        "Write a story about a robot who learns patience.",
    ]

def _encode_prompt(tok: Tokenizer, text: str, bos_id: int) -> list[int]:
    return [bos_id] + tok.encode(text or "").ids


# -------------------- model loading --------------------
def load_model(tag: str, device: str) -> tuple[Transformer, ModelArgs]:
    entry = CHECKPOINTS[tag]
    cfg = _load_json(entry["cfg"])

    args_fields = set(asdict(ModelArgs()).keys())
    filtered = {k: v for k, v in cfg.items() if k in args_fields}

    args = ModelArgs(**filtered)
    args.batch_size = 1
    args.device = device

    model = Transformer(args)
    ckpt = torch.load(str(entry["ckpt"]), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, args


@torch.no_grad()
def run_generate(
    model: Transformer,
    tok: Tokenizer,
    prompt: str,
    *,
    seed: int,
    temperature: float,
    max_new_tokens: int,
    use_kv_cache: bool,
    bos_id: int,
    eos_id: int,
    max_seq_len: int,
    device: str,
    trace: bool,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    prompt_ids = _encode_prompt(tok, prompt, bos_id)
    if len(prompt_ids) >= max_seq_len:
        prompt_ids = prompt_ids[: max_seq_len - 1]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    t0 = time.perf_counter()
    if trace:
        out, trace_out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_kv_cache=use_kv_cache,
            eos_id=eos_id,
            stop_on_eos=True,
            trace=True,
        )
    else:
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_kv_cache=use_kv_cache,
            eos_id=eos_id,
            stop_on_eos=True,
        )
        trace_out = None
    dt = time.perf_counter() - t0

    out_ids = out[0].tolist()
    gen_ids = out_ids[len(prompt_ids):]
    gen_tokens = len(gen_ids)
    tps = (gen_tokens / dt) if dt > 0 else 0.0

    return {
        "gen_tokens": gen_tokens,
        "latency_s": dt,
        "tps": tps,
        "trace": trace_out,
    }


# -------------------- plots --------------------
def plot_pretrain_training_summary():
    entry = CHECKPOINTS.get("Pretrain")
    if not entry:
        return
    train_log = entry.get("train_log")
    val_log = entry.get("val_log")
    if not train_log or not val_log:
        return

    train = _read_jsonl(train_log)
    val = _read_jsonl(val_log)
    if not train or not val:
        return

    train_map = {r["step"]: r for r in train}
    val_map = {r["step"]: r for r in val}

    ts = np.array(sorted(train_map.keys()), dtype=np.int64)
    vs = np.array(sorted(val_map.keys()), dtype=np.int64)

    train_ce = np.array([train_map[s].get("ce", np.nan) for s in ts], dtype=np.float64)
    train_total = np.array([train_map[s].get("loss", np.nan) for s in ts], dtype=np.float64)
    train_llb = np.array([train_map[s].get("llb", np.nan) for s in ts], dtype=np.float64)
    train_lz = np.array([train_map[s].get("lz", np.nan) for s in ts], dtype=np.float64)
    train_tps = np.array([train_map[s].get("tps", np.nan) for s in ts], dtype=np.float64)

    val_ce = np.array([val_map[s].get("val_ce", np.nan) for s in vs], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=False)

    ax = axes[0]
    ax.plot(ts, train_total, label="train_total")
    ax.plot(ts, train_ce, label="train_ce")
    ax.plot(vs, val_ce, label="val_ce")
    ax.set_title("Pretrain loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()

    ax = axes[1]
    ax.plot(ts, train_llb, label="router_llb")
    ax.plot(ts, train_lz, label="router_lz")
    ax.set_title("Pretrain router loss components")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()

    ax = axes[2]
    ax.plot(ts, train_tps, label="tokens/sec")
    ax.set_title("Pretrain throughput")
    ax.set_xlabel("step")
    ax.set_ylabel("tokens/sec")
    ax.legend()

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "pretrain_training_summary.png", dpi=160)
    plt.close(fig)


def plot_kv_throughput_mean(models, tok, meta, prompts):
    print("Starting KV Throughput")
    bos_id = int(meta["special_token_ids"]["[BOS]"])
    eos_id = int(meta["special_token_ids"]["[EOS]"])

    rows = []
    for tag, (model, args, device) in models.items():
        print(f"Starting {tag}")
        tps_off = []
        tps_on = []
        for i, p in enumerate(prompts):
            print(f"Starting Prompt {i+1}")
            seed = 1456 + i

            r0 = run_generate(
                model, tok, p,
                seed=seed, temperature=0.8, max_new_tokens=128,
                use_kv_cache=False, bos_id=bos_id, eos_id=eos_id,
                max_seq_len=int(args.max_seq_len), device=device,
                trace=False,
            )
            r1 = run_generate(
                model, tok, p,
                seed=seed, temperature=0.8, max_new_tokens=128,
                use_kv_cache=True, bos_id=bos_id, eos_id=eos_id,
                max_seq_len=int(args.max_seq_len), device=device,
                trace=False,
            )
            if r0["gen_tokens"] > 0:
                tps_off.append(r0["tps"])
            if r1["gen_tokens"] > 0:
                tps_on.append(r1["tps"])

        rows.append((tag, float(np.mean(tps_off)) if tps_off else 0.0, float(np.mean(tps_on)) if tps_on else 0.0))

    labels = [r[0] for r in rows]
    off = [r[1] for r in rows]
    on = [r[2] for r in rows]

    x = np.arange(len(labels))
    w = 0.35

    fig = plt.figure(figsize=(7.5, 3.4))
    ax = fig.add_subplot(111)
    ax.bar(x - w/2, off, width=w, label="KV OFF")
    ax.bar(x + w/2, on, width=w, label="KV ON")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("tokens/sec (mean over prompts)")
    ax.set_title("KV cache throughput")
    ax.legend()

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "kv_throughput_mean.png", dpi=160)
    plt.close(fig)


def plot_expert_usage_mean(models, tok, meta, prompts):
    print("Starting Expert Plotting")
    bos_id = int(meta["special_token_ids"]["[BOS]"])
    eos_id = int(meta["special_token_ids"]["[EOS]"])

    any_args = next(iter(models.values()))[1]
    E = int(any_args.num_experts)

    usage_by_tag = {}
    for tag, (model, args, device) in models.items():
        print(f"Starting {tag}")
        usage_sum = np.zeros((E,), dtype=np.float64)
        tok_count = 0

        for i, p in enumerate(prompts):
            print(f"Starting Prompt {i+1}")
            seed = 1456 + i
            r = run_generate(
                model, tok, p,
                seed=seed, temperature=0.8, max_new_tokens=128,
                use_kv_cache=True, bos_id=bos_id, eos_id=eos_id,
                max_seq_len=int(args.max_seq_len), device=device,
                trace=True,
            )
            tr = r["trace"]
            if not tr:
                continue

            topk_idx = tr["topk_indices"].cpu().numpy()   # (T,K)
            topk_w = tr["topk_weights"].cpu().numpy()     # (T,K)
            T = int(topk_idx.shape[0])
            K = int(topk_idx.shape[1])
            if T == 0:
                continue

            for t in range(T):
                for j in range(K):
                    e = int(topk_idx[t, j])
                    if 0 <= e < E:
                        usage_sum[e] += float(topk_w[t, j])
            tok_count += T

        usage = (usage_sum / tok_count) if tok_count > 0 else usage_sum
        usage_by_tag[tag] = usage

    n = len(usage_by_tag)
    fig_h = 1.6 * n + 0.6
    fig, axes = plt.subplots(n, 1, figsize=(7.5, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (tag, usage) in zip(axes, usage_by_tag.items()):
        y = 100.0 * usage  # percent
        bars = ax.bar(np.arange(E), y, width=0.6)
        ax.set_title(f"{tag} expert usage (mean over prompts)")
        ax.set_ylabel("% routed")
        ax.set_ylim(0, 100)
        ax.set_xticks(np.arange(E))
        ax.set_xticklabels([_exp_name(i) for i in range(E)])

        for b, v in zip(bars, y):
            ax.text(
                b.get_x() + b.get_width() / 2,
                min(99.0, v + 1.0),
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axes[-1].set_xlabel("expert")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "expert_usage_mean.png", dpi=160)
    plt.close(fig)


def plot_routing_confidence_hist(models, tok, meta, prompts):
    bos_id = int(meta["special_token_ids"]["[BOS]"])
    eos_id = int(meta["special_token_ids"]["[EOS]"])

    top1_all = {}
    for tag, (model, args, device) in models.items():
        vals = []
        for i, p in enumerate(prompts):
            seed = 1456 + i
            r = run_generate(
                model, tok, p,
                seed=seed, temperature=0.8, max_new_tokens=128,
                use_kv_cache=True, bos_id=bos_id, eos_id=eos_id,
                max_seq_len=int(args.max_seq_len), device=device,
                trace=True,
            )
            tr = r["trace"]
            if not tr:
                continue
            topk_w = tr["topk_weights"].cpu().numpy()  # (T,K)
            if topk_w.size == 0:
                continue
            vals.append(topk_w[:, 0])

        top1_all[tag] = (100.0 * np.concatenate(vals, axis=0)) if vals else np.array([], dtype=np.float64)

    fig = plt.figure(figsize=(7.5, 3.2))
    ax = fig.add_subplot(111)
    bins = np.linspace(0, 100, 41)

    for tag, arr in top1_all.items():
        if arr.size == 0:
            continue
        ax.hist(arr, bins=bins, histtype="step", linewidth=1.2, label=tag)

    ax.set_title("Routing confidence (top-1 weight %, aggregated over prompts)")
    ax.set_xlabel("top-1 routing weight (%)")
    ax.set_ylabel("count")
    ax.set_xlim(0, 100)
    ax.legend()

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "routing_confidence_hist.png", dpi=160)
    plt.close(fig)


def main():
    device = _pick_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    meta = _load_json(META_PATH)

    plot_pretrain_training_summary()

    prompts = _load_prompts()[:5]  # cap for speed

    models = {}
    for tag in CHECKPOINTS.keys():
        model, args = load_model(tag, device)
        models[tag] = (model, args, device)

    plot_kv_throughput_mean(models, tok, meta, prompts)
    plot_expert_usage_mean(models, tok, meta, prompts)
    plot_routing_confidence_hist(models, tok, meta, prompts)

    print(f"Wrote figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
