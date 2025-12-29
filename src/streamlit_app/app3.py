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

def _exp_name(e: int) -> str:
    return f"Exp{e+1}"

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
        st.caption("MoE routing view: run Generate to populate.")
        return

    token_ids = trace["token_ids"]                       # list[int]
    topk_idx = trace["topk_indices"].cpu().numpy()       # (T,K)
    topk_w   = trace["topk_weights"].cpu().numpy()       # (T,K)
    margin   = trace["margin"].cpu().numpy()             # (T,)

    T = len(token_ids)
    K = int(topk_idx.shape[1]) if topk_idx.ndim == 2 else 0
    E = int(args.num_experts)

    if T == 0 or K == 0:
        st.caption("MoE routing view: empty trace.")
        return

    dom_k = np.argmax(topk_w, axis=1)
    dom_e = topk_idx[np.arange(T), dom_k]

    # weighted usage per expert: sum of weights where expert appears in top-k, normalized by tokens
    usage = np.zeros((E,), dtype=np.float64)
    for t in range(T):
        for j in range(K):
            e = int(topk_idx[t, j])
            if 0 <= e < E:
                usage[e] += float(topk_w[t, j])
    usage = usage / max(1, T)

    st.divider()
    st.caption("MoE routing (last layer, generated tokens).")
    st.caption("Color = top-1 routed expert for each generated token. Hover a token to see the top-1 and top-2 routing weights (percentages; they sum to 100%).")


    st.markdown(
        """
<style>
.tokenwrap { display:inline-block; margin:2px 2px 8px 0; vertical-align:top; }
.tokenbox  { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
             font-size: 12px; white-space: pre; padding: 2px 4px; border-radius: 4px;
             border: 1px solid rgba(0,0,0,0.12); }
.legendchip { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px;
              border: 1px solid rgba(0,0,0,0.12); }
</style>
""",
        unsafe_allow_html=True,
    )

    # Legend (static, compact)
    legend = []
    for e in range(E):
        c = _EXPERT_COLORS[e % len(_EXPERT_COLORS)]
        legend.append(
            f'<span class="legendchip" style="background:{c}"></span>'
            f'<span style="margin-right:12px;font-family:ui-monospace,monospace;font-size:12px;">{html.escape(_exp_name(e))}</span>'
        )
    st.markdown("".join(legend), unsafe_allow_html=True)

    # Token strip
    parts = []
    for t in range(T):
        tid = int(token_ids[t])
        tok_s = html.escape(_token_str(tok, tid))

        e1 = int(topk_idx[t, 0])
        w1 = float(topk_w[t, 0])
        if K >= 2:
            e2 = int(topk_idx[t, 1])
            w2 = float(topk_w[t, 1])
        else:
            e2, w2 = -1, 0.0

        de = int(dom_e[t])
        color = _EXPERT_COLORS[de % len(_EXPERT_COLORS)]

        # Nicer tooltip (line breaks via &#10;)
        w1_pct = 100.0 * w1
        w2_pct = 100.0 * w2 if K >= 2 else 0.0

        tip = (
            f"Token ID: {tid}&#10;"
            f"Top-1 expert: {_exp_name(e1)} — {w1_pct:.1f}%&#10;"
            f"Top-2 expert: {_exp_name(e2)} — {w2_pct:.1f}%"
            if K >= 2 else
            f"Token ID: {tid}&#10;"
            f"Top-1 expert: {_exp_name(e1)} — {w1_pct:.1f}%"
        )

        parts.append(
    f"""
<span class="tokenwrap" title="{tip}">
  <span class="tokenbox" style="background:{color}33; border-color:{color}66">{tok_s}</span>
</span>
"""
)

    st.markdown("".join(parts), unsafe_allow_html=True)

    # Expert usage: simple bar chart (readable)
    usage_labels = [_exp_name(e) for e in range(E)]
    usage_vals = {usage_labels[e]: float(usage[e]) for e in range(E)}
    st.caption("Last-layer expert usage (weighted fraction over generated tokens).")
    max_u = float(np.max(usage)) if usage.size else 0.0
    max_u = max(max_u, 1e-9)

    bars = []
    for e in range(E):
        u = float(usage[e])
        u_pct = 100.0 * u
        pct = max(0.0, min(100.0, u_pct))
        color = _EXPERT_COLORS[e % len(_EXPERT_COLORS)]
        bars.append(
            f"""
    <div style="display:flex;align-items:center;margin:4px 0;">
    <div style="width:60px;font-family:ui-monospace,monospace;font-size:12px;">{html.escape(_exp_name(e))}</div>
    <div style="flex:1;height:10px;background:rgba(0,0,0,0.08);border-radius:4px;overflow:hidden;">
        <div style="width:{pct:.1f}%;height:10px;background:{color};"></div>
    </div>
    <div style="width:56px;text-align:right;font-family:ui-monospace,monospace;font-size:12px;margin-left:8px;">
        {u_pct:.1f}
    </div>
    </div>
    """
        )
    st.markdown("".join(bars), unsafe_allow_html=True)

# -------------------- UI --------------------
st.set_page_config(page_title="Nano-MoE Demo", layout="wide")

if not CHECKPOINTS:
    st.error("No checkpoints found. Expected artifacts under ./artifacts/{pretraining,sft,dpo}/")
    st.stop()

st.title("Nano-MoE Demo")

device = _pick_device()
tok = load_tokenizer(TOKENIZER_PATH)
meta = load_meta(META_PATH)
bos_id = int(meta["special_token_ids"]["[BOS]"])
eos_id = int(meta["special_token_ids"]["[EOS]"])

# Settings in sidebar (compact)
with st.sidebar:
    st.caption(f"Device: {device}")
    tag = st.selectbox("Checkpoint", list(CHECKPOINTS.keys()), index=0)
    seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=1456, step=1)
    temperature = st.number_input("Temp", min_value=0.0, max_value=5.0, value=0.8, step=0.1)
    max_new = st.number_input("Max new", min_value=1, max_value=512, value=128, step=1)
    use_kv = st.toggle("KV cache", value=True)

model, args = load_model(tag, device)

# Main area: prompt + buttons + output directly below
prompt = st.text_area("Prompt", value="Write a short children's story about a brave teapot.", height=140)
c1, c2 = st.columns([1, 1])
gen_btn = c1.button("Generate", use_container_width=True)
bench_btn = c2.button("KV-cache throughput benchmark", use_container_width=True)

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
        trace=True,
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
    }

last = st.session_state.get("last_generate")
if last:
    st.write(last["gen_text"])
    m = last["metrics"]
    st.caption(
        f"checkpoint={last['tag']} | prompt_tokens={m['prompt_tokens']} | gen_tokens={m['gen_tokens']} | "
        f"latency_s={m['latency_s']:.3f} | tokens/sec={m['tokens_per_sec']:.1f} | kv_cache={'ON' if last['use_kv_cache'] else 'OFF'}"
    )

if bench_btn:
    results = bench_kv(
        model, tok, prompt,
        seed=int(seed),
        temperature=float(temperature),
        max_new_tokens=int(max_new),
        duration_s=15.0,  # minimal; change to 15.0 for "30s total"
        bos_id=bos_id,
        eos_id=eos_id,
        max_seq_len=int(args.max_seq_len),
        device=device,
    )
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

# Screen 2 (below)
render_screen2(tok, args, st.session_state.get("last_generate", {}))
