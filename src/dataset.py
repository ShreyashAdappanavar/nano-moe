# src/data.py
import json
from pathlib import Path

import numpy as np
import torch
from .config import ModelArgs

ROOT = Path(__file__).resolve().parent.parent  # nano-moe/
META_PATH = ROOT / "data" / "shards" / "meta.json"

with META_PATH.open("r", encoding="utf-8") as f:
    meta = json.load(f)

train_path = ROOT / "data" / "shards" / meta["files"]["train"]
val_path   = ROOT / "data" / "shards" / meta["files"]["val"]

mm_train = np.memmap(train_path, dtype=np.uint16, mode="r")
mm_val   = np.memmap(val_path,   dtype=np.uint16, mode="r")

_rng = np.random.default_rng(1337)

def get_batch(args: ModelArgs, split: str):
    assert split in ("train", "val")
    mm = mm_train if split == "train" else mm_val

    B = args.batch_size
    T = args.max_seq_len
    L = T + 1

    N = int(mm.shape[0])
    if N < L + 1:
        raise ValueError(f"{split}.bin too small: N={N}, need at least {L+1}")

    starts = _rng.integers(0, N - L + 1, size=B)
    idx2d = starts[:, None] + np.arange(L, dtype=np.int64)[None, :]
    chunk = mm[idx2d].astype(np.int64, copy=False)  # (B, L)

    x = torch.from_numpy(chunk[:, :-1]).to(device=args.device, dtype=torch.long)
    y = torch.from_numpy(chunk[:, 1:]).to(device=args.device, dtype=torch.long)

    vocab_size = int(meta["vocab_size"])
    if int(x.max()) >= vocab_size or int(y.max()) >= vocab_size:
        raise ValueError("Token id >= vocab_size")

    return x, y