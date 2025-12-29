# app.py
import json
import time
import html
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import streamlit as st
import torch
from tokenizers import Tokenizer

# -------------------- repo root --------------------
def _find_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "src").exists() and (p / "src" / "model.py").exists():
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

# -------------------- helpers --------------------
def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner=False)
def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(tokenizer_path))

@st.cache_resource(show_spinner=False)
def load_meta(meta_path: Path) -> dict:
    return _load_json(meta_path)

@st.cache_resource(show_spinner=True)
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

def encode_prompt(tok: Tokenizer, text: str, bos_id: int) -> list[int]:
    ids = tok.encode(text or "").ids
    return [bos_id] + ids

def _token_str(tok: Tokenizer, token_id: int) -> str:
    s = tok.decode([int(token_id)])
    s = s.replace("\n", "\\n")
    return s if s != "" else "∅"

@torch.no_grad()
def run_generate(
    model: Transformer,
    tok: Tokenizer,
    prompt_text: str,
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
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    prompt_ids = encode_prompt(tok, prompt_text, bos_id)
    if len(prompt_ids) >= max_seq_len:
        prompt_ids = prompt_ids[: max_seq_len - 1]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, T)

    t0 = time.perf_counter()
    if trace:
        out, trace_out = model.generate(
            idx,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            use_kv_cache=bool(use_kv_cache),
            eos_id=int(eos_id),
            stop_on_eos=True,
            trace=True,
        )
    else:
        out = model.generate(
            idx,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            use_kv_cache=bool(use_kv_cache),
            eos_id=int(eos_id),
            stop_on_eos=True,
        )
        trace_out = None
    dt = time.perf_counter() - t0

    out_ids = out[0].tolist()
    gen_ids = out_ids[len(prompt_ids):]
    gen_text = tok.decode(gen_ids)

    gen_tokens = len(gen_ids)
    tps = (gen_tokens / dt) if dt > 0 else 0.0

    return {
        "gen_text": gen_text,
        "prompt_ids": prompt_ids,
        "gen_ids": gen_ids,
        "prompt_tokens": len(prompt_ids),
        "gen_tokens": gen_tokens,
        "latency_s": dt,
        "tokens_per_sec": tps,
        "trace": trace_out,
    }

def bench_kv(
    model: Transformer,
    tok: Tokenizer,
    prompt_text: str,
    *,
    seed: int,
    temperature: float,
    max_new_tokens: int,
    duration_s: float,
    bos_id: int,
    eos_id: int,
    max_seq_len: int,
    device: str,
) -> list[dict]:
    def loop(use_kv: bool) -> dict:
        _ = run_generate(
            model, tok, prompt_text,
            seed=seed,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_kv_cache=use_kv,
            bos_id=bos_id,
            eos_id=eos_id,
            max_seq_len=max_seq_len,
            device=device,
            trace=False,
        )

        t_end = time.perf_counter() + duration_s
        tps_list = []
        runs = 0
        while time.perf_counter() < t_end:
            r = run_generate(
                model, tok, prompt_text,
                seed=seed + runs + (1000 if use_kv else 0),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_kv_cache=use_kv,
                bos_id=bos_id,
                eos_id=eos_id,
                max_seq_len=max_seq_len,
                device=device,
                trace=False,
            )
            if r["gen_tokens"] > 0:
                tps_list.append(r["tokens_per_sec"])
            runs += 1

        if not tps_list:
            return {"kv_cache": use_kv, "runs": runs, "tps_mean": 0.0, "tps_median": 0.0}

        return {
            "kv_cache": use_kv,
            "runs": runs,
            "tps_mean": float(np.mean(tps_list)),
            "tps_median": float(np.median(tps_list)),
        }

    return [loop(False), loop(True)]

# -------------------- Screen 2 rendering --------------------
_EXPERT_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756",
    "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D",
]

def render_screen2(tok: Tokenizer, args: ModelArgs, last: dict):
    trace = last.get("trace")
    gen_ids = last.get("gen_ids", [])
    if not trace or not gen_ids:
        st.info("Run Generate to populate Screen 2.")
        return

    token_ids = trace["token_ids"]
    topk_idx = trace["topk_indices"].cpu().numpy()      # (T,K)
    topk_w   = trace["topk_weights"].cpu().numpy()      # (T,K)
    margin   = trace["margin"].cpu().numpy()            # (T,)

    T = len(token_ids)
    K = topk_idx.shape[1] if topk_idx.ndim == 2 else 0
    E = int(args.num_experts)

    # dominant expert per token = argmax over top-k weights, mapped to that expert id
    dom_k = np.argmax(topk_w, axis=1) if T > 0 else np.array([], dtype=np.int64)
    dom_e = topk_idx[np.arange(T), dom_k] if T > 0 else np.array([], dtype=np.int64)

    # weighted usage per expert (sum of top-k weights where expert appears)
    usage = np.zeros((E,), dtype=np.float64)
    for t in range(T):
        for j in range(K):
            e = int(topk_idx[t, j])
            if 0 <= e < E:
                usage[e] += float(topk_w[t, j])
    usage = usage / max(1, T)  # fraction per token (weights sum to 1 per token)

    st.subheader("Screen 2 — MoE Routing (last layer, generated tokens)")

    # Token strip (HTML)
    st.markdown(
        """
<style>
.tokenwrap { display:inline-block; margin:2px 2px 6px 0; vertical-align:top; }
.tokenbox  { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
             font-size: 12px; white-space: pre; padding: 2px 4px; border-radius: 4px;
             border: 1px solid rgba(0,0,0,0.15); }
.mbar      { height: 3px; margin-top: 2px; border-radius: 2px; background: rgba(0,0,0,0.08); }
.mfill     { height: 3px; border-radius: 2px; background: rgba(0,0,0,0.55); }
</style>
""",
        unsafe_allow_html=True,
    )

    parts = []
    for t in range(T):
        tid = int(token_ids[t])
        tok_s = html.escape(_token_str(tok, tid))
        e = int(dom_e[t]) if t < len(dom_e) else 0
        color = _EXPERT_COLORS[e % len(_EXPERT_COLORS)]
        m = float(margin[t]) if t < len(margin) else 0.0
        m = max(0.0, min(1.0, m))
        w = int(round(100.0 * m))

        # tooltip: show top-2 experts/weights + margin + token id
        top_pairs = []
        for j in range(min(2, K)):
            top_pairs.append(f"e{int(topk_idx[t,j])}:{float(topk_w[t,j]):.3f}")
        tip = html.escape(
            f"id={tid} | top={', '.join(top_pairs)} | margin={m:.3f}"
        )

        parts.append(
            f"""
<span class="tokenwrap" title="{tip}">
  <span class="tokenbox" style="background:{color}33; border-color:{color}66">{tok_s}</span>
  <div class="mbar"><div class="mfill" style="width:{w}%"></div></div>
</span>
"""
        )

    st.markdown("".join(parts), unsafe_allow_html=True)

    # Heatmap: last layer only (1 x E), intensity by usage
    max_u = float(np.max(usage)) if usage.size else 0.0
    if max_u <= 0:
        max_u = 1.0

    cells = []
    for e in range(E):
        u = float(usage[e])
        a = max(0.05, min(1.0, u / max_u))
        color = _EXPERT_COLORS[e % len(_EXPERT_COLORS)]
        tip = html.escape(f"expert={e} | usage={u:.4f}")
        cells.append(
            f'<div title="{tip}" style="width:42px;height:22px;border-radius:4px;'
            f'background:{color};opacity:{a:.3f};border:1px solid rgba(0,0,0,0.15);'
            f'display:inline-block;margin-right:4px"></div>'
        )

    st.markdown("**Last-layer expert usage (weighted, over generated tokens)**")
    st.markdown("".join(cells), unsafe_allow_html=True)

    # Optional: exact numbers in one line (still minimal)
    usage_str = "  ".join([f"e{e}:{usage[e]:.3f}" for e in range(E)])
    st.caption(f"usage: {usage_str}")

# -------------------- UI --------------------
st.set_page_config(page_title="Nano-MoE Demo", layout="wide")
st.title("Nano-MoE: Checkpoints + KV-cache + MoE Routing (Screen 1 + 2)")

if not CHECKPOINTS:
    st.error("No checkpoints found. Expected artifacts under ./artifacts/{pretraining,sft,dpo}/")
    st.stop()

device = _pick_device()
tok = load_tokenizer(TOKENIZER_PATH)
meta = load_meta(META_PATH)
bos_id = int(meta["special_token_ids"]["[BOS]"])
eos_id = int(meta["special_token_ids"]["[EOS]"])

col_l, col_r = st.columns([1, 2])

with col_l:
    tag = st.selectbox("Checkpoint", list(CHECKPOINTS.keys()), index=0)
    seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=1456, step=1)
    temperature = st.number_input("Temperature", min_value=0.0, max_value=5.0, value=0.8, step=0.1)
    max_new = st.number_input("Max new tokens", min_value=1, max_value=512, value=128, step=1)
    use_kv = st.toggle("KV cache", value=True)
    st.caption(f"Device: {device}")

with col_r:
    # val = "Write a very short children's story.\n - Protagonist: a silly cat\n Style:\n - Simple words and short sentences.\n"
    val = (    "Write a very short children's story.\n\n"
    "Constraints:\n"
    f"- Protagonist: a cheerful dog\n"
    "Style:\n"
    "- Simple words and short sentences.\n")
    prompt = st.text_area("Prompt", value=val, height=140)
    c1, c2 = st.columns([1, 1])
    gen_btn = c1.button("Generate", use_container_width=True)
    bench_btn = c2.button("KV-cache throughput benchmark", use_container_width=True)

model, args = load_model(tag, device)

# Screen 1: generate
if gen_btn:
    r = run_generate(
        model, tok, prompt,
        seed=int(seed),
        temperature=float(temperature),
        max_new_tokens=int(max_new),
        use_kv_cache=bool(use_kv),
        bos_id=bos_id,
        eos_id=eos_id,
        max_seq_len=int(args.max_seq_len),
        device=device,
        trace=True,  # Screen 2 needs this; simplest
    )
    st.session_state["last_generate"] = {
        "tag": tag,
        "prompt": prompt,
        "seed": int(seed),
        "temperature": float(temperature),
        "max_new_tokens": int(max_new),
        "use_kv_cache": bool(use_kv),
        "prompt_ids": r["prompt_ids"],
        "gen_ids": r["gen_ids"],
        "gen_text": r["gen_text"],
        "metrics": {
            "prompt_tokens": r["prompt_tokens"],
            "gen_tokens": r["gen_tokens"],
            "latency_s": r["latency_s"],
            "tokens_per_sec": r["tokens_per_sec"],
        },
        "trace": r["trace"],
        "args": {
            "num_experts": int(args.num_experts),
            "top_k": int(args.top_k),
            "max_seq_len": int(args.max_seq_len),
        },
    }

# Display Screen 1 output if available
last = st.session_state.get("last_generate")
if last:
    st.subheader("Screen 1 — Output")
    st.write(last["gen_text"])
    m = last["metrics"]
    st.caption(
        f"checkpoint={last['tag']} | prompt_tokens={m['prompt_tokens']} | gen_tokens={m['gen_tokens']} | "
        f"latency_s={m['latency_s']:.3f} | tokens/sec={m['tokens_per_sec']:.1f} | kv_cache={'ON' if last['use_kv_cache'] else 'OFF'}"
    )

# Screen 1: benchmark
if bench_btn:
    results = bench_kv(
        model, tok, prompt,
        seed=int(seed),
        temperature=float(temperature),
        max_new_tokens=int(max_new),
        duration_s=10.0,  # keep minimal; change to 15.0 for "30s total"
        bos_id=bos_id,
        eos_id=eos_id,
        max_seq_len=int(args.max_seq_len),
        device=device,
    )
    st.subheader("KV-cache throughput benchmark (same prompt/settings; OFF vs ON)")
    rows = []
    for r in results:
        rows.append(f"| {'ON' if r['kv_cache'] else 'OFF'} | {r['runs']} | {r['tps_mean']:.1f} | {r['tps_median']:.1f} |")
    st.markdown(
        "\n".join([
            "| KV cache | runs | tps_mean | tps_median |",
            "|---|---:|---:|---:|",
            *rows
        ])
    )

# Screen 2
st.divider()
render_screen2(tok, args, st.session_state.get("last_generate", {}))