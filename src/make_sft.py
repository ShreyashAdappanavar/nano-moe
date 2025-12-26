import os
import json
import time
import re
from pathlib import Path
from datetime import datetime

from openai import OpenAI


ROOT = Path(__file__).resolve().parent.parent  # nano-moe/

PROMPTS_PATH = ROOT / "data" / "synthetic_data_generation" / "prompts_seed.jsonl"
SPEC_PATH = ROOT / "data" / "synthetic_data_generation" / "sft_generation_spec.json"

OUT_TRAIN = ROOT / "data" / "synthetic_data_generation" / "sft_train.jsonl"
OUT_VAL = ROOT / "data" / "synthetic_data_generation" / "sft_val.jsonl"
OUT_FAIL = ROOT / "data" / "synthetic_data_generation" / "rejects" / "sft_failed.jsonl"

MAX_CHARS = 2000
MAX_TRIES = 3


def count_sentences(text: str) -> int:
    parts = re.split(r"[.!?]+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)


def validate(text: str, required: list[str], banned: list[str], min_s: int, max_s: int) -> tuple[bool, str]:
    t = text.strip()
    if len(t) == 0:
        return False, "empty"
    if len(t) > MAX_CHARS:
        return False, "too_long"

    low = t.lower()
    for r in required:
        if r.lower() not in low:
            return False, "missing_required"

    for b in banned:
        if b.lower() in low:
            return False, "contains_banned"

    return True, "ok"


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    with SPEC_PATH.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    model = str(spec["model"])
    system_prompt = str(spec["system_prompt"])
    params = dict(spec.get("params", {}))

    temperature = float(params.get("temperature", 0.7))
    top_p = float(params.get("top_p", 1.0))
    max_output_tokens = int(params.get("max_output_tokens", 350))

    client = OpenAI(api_key=api_key)

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    OUT_VAL.parent.mkdir(parents=True, exist_ok=True)
    OUT_FAIL.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0
    n_train = 0
    n_val = 0

    t0_all = time.perf_counter()
    n_done = 0

    N_TOTAL = sum(1 for _ in PROMPTS_PATH.open("r", encoding="utf-8"))

    with PROMPTS_PATH.open("r", encoding="utf-8") as f_in, \
         OUT_TRAIN.open("w", encoding="utf-8") as f_tr, \
         OUT_VAL.open("w", encoding="utf-8") as f_va, \
         OUT_FAIL.open("w", encoding="utf-8") as f_fail:
        
        print("Starting the generation...")

        for i, line in enumerate(f_in):
            
            row = json.loads(line)

            pid = str(row["id"])
            split = str(row["split"])
            instruction = str(row["instruction"])

            required = row["required"]
            banned = row["banned"]
            min_s = row["min_sentences"]
            max_s = row["max_sentences"]

            out_text = None
            request_id = None
            fail_reason = None

            for attempt in range(MAX_TRIES):
                try:
                    resp = client.responses.create(
                        model=model,
                        instructions=system_prompt,
                        input=instruction,
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_output_tokens,
                    )
                    request_id = getattr(resp, "id", None)
                    out_text = resp.output_text or ""
                except Exception as e:
                    fail_reason = f"api_error:{type(e).__name__}"
                    time.sleep(2 * (attempt + 1))
                    continue

                ok, reason = validate(out_text, required, banned, min_s, max_s)
                if ok:
                    fail_reason = None
                    break

                fail_reason = reason
                time.sleep(1 * (attempt + 1))
            
            if out_text is None or fail_reason is not None:
                n_fail += 1
                f_fail.write(json.dumps({
                    "id": pid,
                    "split": split,
                    "reason": fail_reason if fail_reason is not None else "unknown",
                }, ensure_ascii=False) + "\n")

                n_done += 1

                if (i + 1) % 5 == 0:
                    now = time.perf_counter()
                    avg_s = (now - t0_all) / max(1, n_done)
                    rem = max(0, N_TOTAL - n_done)
                    eta_s = rem * avg_s
                    print(f"{i+1} | ok={n_ok} fail={n_fail} | train={n_train} val={n_val} | avg_s={avg_s:.2f} | eta_min={eta_s/60:.1f}")
                
                continue

            n_sent = count_sentences(out_text)

            item = {
                "id": pid,
                "prompt": instruction,
                "response": out_text.strip(),
                "constraints": {
                    "required": required,
                    "banned": banned,
                    "min_sentences": min_s,
                    "max_sentences": max_s,
                },
                "sentence_count": int(n_sent),
                "sentence_ok": bool(min_s <= n_sent <= max_s),
                "provenance": {
                    "provider": str(spec.get("provider", "openai")),
                    "model": model,
                    "system_prompt": system_prompt,
                    "params": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_output_tokens": max_output_tokens,
                    },
                    "request_id": request_id,
                    "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                },
            }

            if split == "train":
                f_tr.write(json.dumps(item, ensure_ascii=False) + "\n")
                n_train += 1
            else:
                f_va.write(json.dumps(item, ensure_ascii=False) + "\n")
                n_val += 1

            n_ok += 1
            n_done += 1

            if (i + 1) % 5 == 0:
                now = time.perf_counter()
                avg_s = (now - t0_all) / max(1, n_done)
                rem = max(0, N_TOTAL - n_done)
                eta_s = rem * avg_s
                print(f"{i+1} | ok={n_ok} fail={n_fail} | train={n_train} val={n_val} | avg_s={avg_s:.2f} | eta_min={eta_s/60:.1f}")

            print(f"Prompt {i} done!")
            

    print("Done")
    print(f"train={n_train} val={n_val} fail={n_fail}")


if __name__ == "__main__":
    main()
