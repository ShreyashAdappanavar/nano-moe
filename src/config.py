from dataclasses import dataclass

@dataclass
class ModelArgs:
    d_model: int = 192
    hidden_dim: int = 512
    n_layers: int = 6
    n_heads: int = 8

    vocab_size: int = 10_000

    num_experts: int = 4
    num_shared_experts: int = 1
    top_k: int = 2

    n_kv_heads: int = 2
    max_seq_len: int = 512
    rope_theta: float = 10_000.0

    batch_size: int = 64

    rmsnorm_eps: float = 1e-6
    init_mean: float = 0.0
    init_std: float = 0.02

    lr: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0

    # 500M tokens target:
    # tokens/step = 64 * 512 = 32,768
    # steps ≈ 467,421,730 / 32,768 = 14_265
    max_steps: int = 14_265
    warmup_steps: int = 1_000
    log_interval: int = 10
    eval_interval: int = 500
    eval_batches: int = 20
    ckpt_interval: int = 1_000

    alpha_lb: float = 1e-2
    beta_z: float = 1e-3

    seed: int = 1456
    out_dir: str = "output"
    device: str = "cuda"


    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert (self.d_model // self.n_heads) % 2 == 0  # dk must be even for RoPE
        assert self.n_heads % self.n_kv_heads == 0
        assert self.top_k <= self.num_experts
        assert self.num_experts > 0 and self.num_shared_experts >= 0
        assert self.top_k >= 1