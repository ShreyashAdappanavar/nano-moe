import json
from pathlib import Path
from datetime import datetime

import numpy as np


ROOT = Path(__file__).resolve().parent.parent  # nano-moe/
OUT_PATH = ROOT / "data" / "synthetic_data_generation" / "prompts_seed.jsonl"

N_TOTAL = 2500
N_TRAIN = 2000

SEED = 1337

MIN_SENTENCES = 4
MAX_SENTENCES = 12

THEME_SET = [
    "friendship",
    "kindness",
    "honesty",
    "courage",
    "curiosity",
    "sharing",
    "patience",
    "teamwork",
    "responsibility",
    "gratitude",
]

ADJECTIVES = [
    "brave", "shy", "curious", "kind", "sleepy", "cheerful", "gentle", "quiet",
    "tiny", "helpful", "careful", "honest", "silly", "proud", "patient",
]

NOUNS = [
    "teapot", "dragon", "rabbit", "robot", "pencil", "cloud", "turtle", "cat",
    "dog", "bear", "kite", "book", "lamp", "cookie", "train",
]

REQUIRED_PHRASES = [
    "red balloon", "blue umbrella", "gold coin", "small key", "paper boat",
    "wooden spoon", "green leaf", "yellow hat", "tiny bell", "silver button",
    "warm blanket", "old map", "soft pillow", "rainy day", "sunny hill",
    "quiet pond", "little song", "sticky jam", "clean cup", "bright star",
    "brown acorn", "purple scarf", "orange carrot", "striped sock",
]

BANNED_PHRASES = [
    "blood", "gun", "kill", "hate", "suicide", "sex"
]

P_BANNED = 0.15  # fraction of prompts with 1 banned phrase


def render_instruction(protagonist: str, theme: str, required: list[str], banned: list[str]):
    req_lines = "\n".join([f"  - {r}" for r in required])

    if len(banned) == 0:
        banned_block = "  NONE"
    else:
        banned_block = "\n".join([f"  - {b}" for b in banned])

    s = (
        "Write a children's story.\n\n"
        "Constraints:\n"
        f"- Protagonist: {protagonist}\n"
        f"- Theme: {theme}\n"
        f"- Length: {MIN_SENTENCES} to {MAX_SENTENCES} sentences.\n"
        "- Must include ALL of these exact phrases (case-insensitive match is acceptable):\n"
        f"{req_lines}\n"
        "- Must NOT include any of these phrases (case-insensitive match):\n"
        f"{banned_block}\n\n"
        "Style:\n"
        "- Simple words and short sentences.\n"
        "- Child-friendly tone.\n"
        "- No meta commentary about writing.\n"
        "- Do not use bullet points or numbered lists.\n\n"
        "Formatting:\n"
        "- Output plain text only.\n"
    )
    return s


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    items = []
    used_ids = set()

    for i in range(N_TOTAL):
        pid = f"p{i:05d}"
        assert pid not in used_ids
        used_ids.add(pid)

        split = "train" if i < N_TRAIN else "val"

        adj = rng.choice(ADJECTIVES)
        noun = rng.choice(NOUNS)
        protagonist = f"a {adj} {noun}"

        theme = str(rng.choice(THEME_SET))

        k_req = int(rng.integers(2, 5))  # 2..4
        required = rng.choice(REQUIRED_PHRASES, size=k_req, replace=False).tolist()

        banned = []
        if float(rng.random()) < P_BANNED:
            banned = [str(rng.choice(BANNED_PHRASES))]

        instruction = render_instruction(protagonist, theme, required, banned)

        obj = {
            "id": pid,
            "split": split,
            "protagonist": protagonist,
            "theme": theme,
            "required": required,
            "banned": banned,
            "min_sentences": MIN_SENTENCES,
            "max_sentences": MAX_SENTENCES,
            "instruction": instruction,
            "seed_meta": {
                "rng_seed": SEED,
                "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            },
        }

        items.append(obj)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT_PATH} lines={len(items)}")


if __name__ == "__main__":
    main()
