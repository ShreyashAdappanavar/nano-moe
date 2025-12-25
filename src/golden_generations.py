# scripts/golden_generations.py
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from src.config import ModelArgs
from src.model import Transformer


ROOT = Path(__file__).resolve().parent.parent  # nano-moe/

ART_DIR = ROOT / "artifacts" / "pretraining"
OUT_PATH = ART_DIR / "golden_generations2.json"

PROMPTS_DIR = ROOT / "prompts"
STORY_PROMPTS_PATH = PROMPTS_DIR / "story_prompts.txt"
INSTRUCT_PROMPTS_PATH = PROMPTS_DIR / "instruct_prompts.txt"

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
USE_KV_CACHE = False
STOP_ON_EOS = True

SEED = 1234


def read_prompts(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [s.rstrip() for s in lines if s.strip()]
    return lines


def main():
    ART_DIR.mkdir(parents=True, exist_ok=True)

    cfg_path = ART_DIR / "config.json"
    ckpt_path = ART_DIR / "ckpt_best.pt"
    tok_path = ART_DIR / "tokenizer.json"
    meta_path = ART_DIR / "meta.json"

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    args = ModelArgs(**cfg)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bos_id = int(meta["special_token_ids"]["[BOS]"])
    eos_id = int(meta["special_token_ids"]["[EOS]"])

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    tok = Tokenizer.from_file(str(tok_path))

    model = Transformer(args).to(args.device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    story_prompts = read_prompts(STORY_PROMPTS_PATH)
    instruct_prompts = read_prompts(INSTRUCT_PROMPTS_PATH)

    recs = []
    pid = 0

    def run_one(prompt_type: str, prompt_text: str):
        nonlocal pid

        prompt_ids = [bos_id] + tok.encode(prompt_text).ids
        idx = torch.tensor([prompt_ids], device=args.device, dtype=torch.long)

        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                use_kv_cache=USE_KV_CACHE,
                eos_id=eos_id,
                stop_on_eos=STOP_ON_EOS,
            )

        out_ids = out[0].tolist()
        gen_ids = out_ids[len(prompt_ids):]
        gen_text = tok.decode(gen_ids)

        recs.append({
            "prompt_id": pid,
            "prompt_type": prompt_type,
            "prompt_text": prompt_text,
            "prompt_ids": prompt_ids,
            "gen_ids": gen_ids,
            "gen_text": gen_text,
        })

        pid += 1

    print("Starting the story prompts...")
    for i, p in enumerate(story_prompts, start=1):
        run_one("story", p)
        print(f"Story Prompt {i} completed!")

    for i, p in enumerate(instruct_prompts, start=1):
        run_one("instruct", p)
        print(f"Instruction Prompt {i} completed!")

    out_obj = {
        "artifact_tag": "pretrain",
        "seed": SEED,
        "generation_settings": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "use_kv_cache": USE_KV_CACHE,
            "stop_on_eos": STOP_ON_EOS,
            "eos_id": eos_id,
        },
        "counts": {
            "story_prompts": len(story_prompts),
            "instruct_prompts": len(instruct_prompts),
            "total": len(recs),
        },
        "items": recs,
    }

    OUT_PATH.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
