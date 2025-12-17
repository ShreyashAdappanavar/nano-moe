from dataclasses import dataclass

@dataclass
class ModelArgs:
    d_model: int = 256
    hidden_dim: int = 512
    n_layers: int = 4
    n_heads: int = 4
    vocab_size: int = 10_000
    
    num_experts: int = 8
    num_shared_experts: int = 1
    top_k: int = 2

    n_kv_heads: int = 2
    max_seq_len: int =  1024
    rope_theta: float = 10_000.0
    batch_size: int = 32

    rmsnorm_eps: float = 1e-6

    init_mean: float = 0.0
    init_std: float = 0.02

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert (self.d_model // self.n_heads) % 2 == 0  # dk must be even for RoPE
        assert self.n_heads % self.n_kv_heads == 0
        assert self.top_k <= self.num_experts
        assert self.num_experts > 0 and self.num_shared_experts >= 0
        assert self.top_k >= 1