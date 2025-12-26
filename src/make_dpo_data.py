# src/make_dpo.py
import os
import json
import time
from pathlib import Path

import numpy as np
from openai import OpenAI


ROOT = Path(__file__).resolve().parent.parent  # nano-moe/
BASE = ROOT / "data" / "synthetic_data_generation"

PROMPTS_PATH = BASE / "prompts_seed.jsonl"
SPEC_PATH    = BASE / "sft_generation_spec.json"

SFT_TRAIN = BASE / "sft_train.jsonl"
SFT_VAL   = BASE / "sft_val.jsonl"

OUT_TRAIN = BASE / "dpo_train.jsonl"
OUT_VAL   = BASE / "dpo_val.jsonl"

TRIES = 3
SEED = 1337
MAX_OUTPUT_TOKENS = 256

N_TRAIN = 10
N_VAL = 2

ADJECTIVES = [
    "brave", "shy", "curious", "kind", "sleepy", "cheerful", "gentle", "quiet",
    "tiny", "helpful", "careful", "honest", "silly", "proud", "patient",
]
NOUNS = [
    "teapot", "dragon", "rabbit", "robot", "pencil", "cloud", "turtle", "cat",
    "dog", "bear", "kite", "book", "lamp", "cookie", "train",
]


def has(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    with SPEC_PATH.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    model = str(spec["model"])
    system_prompt = str(spec["system_prompt"])

    client = OpenAI(api_key=api_key)
    rng = np.random.default_rng(SEED)

    # seeds map: id -> row
    seeds = {}
    with PROMPTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                seeds[str(r["id"])] = r

    def load_list(path: Path):
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    sft_train = load_list(SFT_TRAIN)
    sft_val   = load_list(SFT_VAL)

    def write_split(sft_rows, want_n: int, out_path: Path, split_name: str):
        out_path.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        t0 = time.perf_counter()
        with out_path.open("w", encoding="utf-8") as f_out:
            for sft in sft_rows:
                if written >= want_n:
                    break

                pid = str(sft["id"])
                seed = seeds[pid]  # crash if mismatch

                instruction = str(seed["instruction"])
                old_prot = str(seed["protagonist"])
                required = seed["required"]

                chosen = str(sft["response"]).strip()

                # alternate: even -> missing_required, odd -> wrong_protagonist
                if written % 2 == 0:
                    vt = "missing_required"
                    detail = str(required[0])
                    user_prompt = (
                        instruction
                        + "\n\nVIOLATION:\n"
                        + f'Do NOT include this exact phrase anywhere: "{detail}".\n'
                        + "Keep the story short.\n"
                    )
                else:
                    vt = "wrong_protagonist"
                    new_prot = old_prot
                    while new_prot.lower() == old_prot.lower():
                        new_prot = f"a {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)}"
                    detail = new_prot
                    user_prompt = (
                        instruction
                        + "\n\nVIOLATION:\n"
                        + f'The protagonist must be: "{detail}".\n'
                        + f'Do NOT include this exact string anywhere: "{old_prot}".\n'
                        + "Keep the story short.\n"
                    )

                rejected = None
                for attempt in range(TRIES):
                    try:
                        r = client.responses.create(
                            model=model,
                            instructions=system_prompt,
                            input=user_prompt,
                            temperature=0.4,
                            top_p=1.0,
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                        )
                        txt = (r.output_text or "").strip()
                    except Exception:
                        time.sleep(2 * (attempt + 1))
                        continue

                    if txt == "":
                        time.sleep(1 * (attempt + 1))
                        continue

                    if vt == "missing_required":
                        if has(txt, detail):
                            time.sleep(1 * (attempt + 1))
                            continue
                    else:
                        if (not has(txt, detail)) or has(txt, old_prot):
                            time.sleep(1 * (attempt + 1))
                            continue

                    rejected = txt
                    break

                if rejected is None:
                    continue

                f_out.write(json.dumps({
                    "id": pid,
                    "split": split_name,
                    "prompt": instruction,
                    "chosen": chosen,
                    "rejected": rejected,
                    "violation": vt,
                    "detail": detail,
                }, ensure_ascii=False) + "\n")

                written += 1
                if written % 5 == 0:
                    dt = time.perf_counter() - t0
                    avg = dt / max(1, written)
                    rem = want_n - written
                    eta_min = (rem * avg) / 60.0
                    print(f"{split_name}: {written}/{want_n} | avg_s={avg:.2f} | eta_min={eta_min:.1f}")
                
                print(f"Finished prompt {written}!")

        if written != want_n:
            raise RuntimeError(f"{split_name}: wrote {written}/{want_n}")

    write_split(sft_train, N_TRAIN, OUT_TRAIN, "train")
    write_split(sft_val,   N_VAL,   OUT_VAL,   "val")


if __name__ == "__main__":
    main()
