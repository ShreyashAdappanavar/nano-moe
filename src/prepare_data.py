# prepare_data_min.py
# Creates: data/shards/train.bin, data/shards/val.bin, data/shards/meta.json
# Format: flat stream of uint16 token ids: [BOS] + encode(text) + [EOS]

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer


def main():
    tokenizer_path = "tokenizer.json"
    out_dir = Path("data/shards")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = "roneneldan/TinyStories"
    split_name = "train"

    seed = 1337
    train_stories = 250_000
    val_stories = 10_000

    vocab_size = 10_000
    max_seq_len = 512  # used later by the loader; file is a flat stream

    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Your known ids (fail fast if mismatch)
    unk_id, pad_id, bos_id, eos_id = 0, 1, 2, 3
    vocab = tokenizer.get_vocab()
    assert vocab["[UNK]"] == unk_id
    assert vocab["[PAD]"] == pad_id
    assert vocab["[BOS]"] == bos_id
    assert vocab["[EOS]"] == eos_id

    ds = load_dataset(dataset_name)
    texts = ds[split_name]["text"]
    n = len(texts)

    total = train_stories + val_stories
    if total > n:
        raise ValueError(f"Requested {total} stories but dataset has {n}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_idx = perm[:train_stories]
    val_idx = perm[train_stories:total]

    def write_bin(indices: np.ndarray, path: Path):
        num_tokens = 0
        max_id = -1

        with open(path, "wb") as f:
            for num, i in enumerate(indices):
                ids = tokenizer.encode(texts[int(i)]).ids
                ids = [bos_id] + ids + [eos_id]

                if ids:
                    local_max = max(ids)
                    if local_max >= vocab_size:
                        raise ValueError(f"token id {local_max} >= vocab_size {vocab_size}")
                    if local_max > max_id:
                        max_id = local_max

                arr = np.asarray(ids, dtype=np.uint16)
                arr.tofile(f)
                num_tokens += arr.size
                print(f"Completed number {num}")

        if num_tokens == 0:
            raise RuntimeError(f"{path.name} is empty")
        return num_tokens, max_id

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    meta_path = out_dir / "meta.json"

    train_tokens, train_max = write_bin(train_idx, train_path)
    val_tokens, val_max = write_bin(val_idx, val_path)

    meta = {
        "dataset": dataset_name,
        "split": split_name,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "max_seq_len": max_seq_len,
        "special_token_ids": {
            "[UNK]": unk_id,
            "[PAD]": pad_id,
            "[BOS]": bos_id,
            "[EOS]": eos_id
        },
        "files": {
            "train": train_path.name,
            "val": val_path.name
        },
        "counts": {
            "train_stories": int(train_stories),
            "val_stories": int(val_stories),
            "train_tokens": int(train_tokens),
            "val_tokens": int(val_tokens)
        },
        "seed": seed
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:")
    print(f"  {train_path}  tokens={train_tokens}  max_id={train_max}")
    print(f"  {val_path}    tokens={val_tokens}    max_id={val_max}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
