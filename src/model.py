import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelArgs
import math

class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ip = args.d_model
        self.h_dim = args.hidden_dim
        self.op = args.d_model

        self.w1 = nn.Linear(self.ip, self.h_dim, bias=False)
        self.w2 = nn.Linear(self.ip, self.h_dim, bias=False)
        self.c_proj = nn.Linear(self.h_dim, self.op, bias=False)
    
    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, sequence_length, d_model)

        return self.c_proj(F.silu(self.w1(x)) * self.w2(x))
    
class MoELayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.d_model = args.d_model
        self.hidden_dim = args.hidden_dim
        self.num_shared_experts = args.num_shared_experts
        self.num_experts = args.num_experts
        self.top_k = args.top_k

        self.shared_experts = nn.ModuleList([MLP(args) for i in range(self.num_shared_experts)])
        self.experts = nn.ModuleList([MLP(args) for i in range(self.num_experts)])
        self.router = nn.Linear(self.d_model, self.num_experts, bias=False)
    
    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, sequence_length, d_model)
        batch_size, sequence_length, _ = x.size()

        shared_exps_op = []
        for i in self.shared_experts:
                shared_exps_op.append(i(x))

        logits = self.router(x) # (batch_size, sequence_length, num_experts)
        sf_max_logits = F.softmax(logits, dim=-1) # (batch_size, sequence_length, num_experts)
        topk = torch.topk(sf_max_logits, self.top_k) # .values and .indices : (batch_size, sequence_length, top_k)
        weights = topk.values # (batch_size, sequence_length, top_k)
        indices = topk.indices # (batch_size, sequence_length, top_k)

        weights = weights/torch.sum(weights, dim=-1, keepdim=True)

        x_flat = x.contiguous().view(batch_size * sequence_length, self.d_model) # (batch_size * sequence_length, d_model)
        weights = weights.contiguous().view(batch_size * sequence_length, self.top_k) # (batch_size * sequence_length, top_k)
        indices = indices.contiguous().view(batch_size * sequence_length, self.top_k) # (batch_size * sequence_length, top_k)

        routed_output = torch.zeros_like(x_flat) # (batch_size * sequence_length, d_model)

        for i, curr_expert in enumerate(self.experts):
            row, index = torch.where(indices==i)
            curr_expert_ip = x_flat[row] # (selected_rows, d_model)
            curr_expert_op = curr_expert(curr_expert_ip) # (selected_rows, d_model)
            curr_expert_op *= weights[row, index].unsqueeze(-1)
            routed_output[row] += curr_expert_op
            
        routed_output = routed_output.contiguous().view(batch_size, sequence_length, self.d_model)
              
        final_op = routed_output + sum(shared_exps_op) / max(1, self.num_shared_experts) 
        return final_op, logits
    
class CausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.hidden_dim = args.hidden_dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.max_seq_len = args.max_seq_len
        self.rope_theta = args.rope_theta
        self.batch_size = args.batch_size

        self.dk = self.d_model // self.n_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.kv_proj = nn.Linear(self.d_model, 2 * self.n_kv_heads * self.dk, bias=False)
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        causal_mask = torch.triu(torch.full((self.max_seq_len, self.max_seq_len), float("-inf")), 
                                 diagonal=1).contiguous().view(1, 1, self.max_seq_len, self.max_seq_len).contiguous()
        
        k_cache = torch.zeros((self.batch_size, self.max_seq_len, self.n_kv_heads, self.dk))
        v_cache = torch.zeros((self.batch_size, self.max_seq_len, self.n_kv_heads, self.dk))

        pairs = torch.arange(0, self.dk // 2) # (dk // 2,)
        frequencies = 1/self.rope_theta**(2 * pairs / self.dk) # (dk // 2,)

        positions = torch.arange(self.max_seq_len, dtype=torch.float32) # (max_seq_len,)

        angles = positions.contiguous().view(len(positions), 1).contiguous() @ frequencies.contiguous().view(1, len(frequencies)).contiguous() # (max_seq_len, dk // 2)

        sin_matrix = torch.sin(angles) # (max_seq_len, dk // 2)
        cos_matrix = torch.cos(angles) # (max_seq_len, dk // 2)

        self.register_buffer("cos_matrix", cos_matrix)
        self.register_buffer("sin_matrix", sin_matrix)
        self.register_buffer("causal_mask", causal_mask)
        self.register_buffer("k_cache", k_cache, persistent=False)
        self.register_buffer("v_cache", v_cache, persistent=False)

    def forward(self, x: torch.Tensor, start_posn: int = 0, use_kv_cache: bool = False):
        # x: (B, T, D)
        B, T, _ = x.size()
        assert B <= self.batch_size
        assert start_posn >= 0
        assert start_posn + T <= self.max_seq_len   
        dk = self.dk
        H = self.n_heads
        HKV = self.n_kv_heads
        rep = H // HKV

        q = self.q_proj(x)  # (B, T, D)
        k, v = self.kv_proj(x).split(HKV * dk, dim=-1)  # (B, T, HKV*dk), (B, T, HKV*dk)

        q = q.contiguous().view(B, T, H, dk).transpose(1, 2).contiguous()      # (B, H,   T, dk)
        k = k.contiguous().view(B, T, HKV, dk).transpose(1, 2).contiguous()    # (B, HKV, T, dk)
        v = v.contiguous().view(B, T, HKV, dk).transpose(1, 2).contiguous()    # (B, HKV, T, dk)

        # RoPE (keep dtype consistent)
        cos = self.cos_matrix[start_posn:start_posn + T].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)  # (1,1,T,dk/2)
        sin = self.sin_matrix[start_posn:start_posn + T].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)

        def apply_rope(t, cos, sin):
            t_even = t[..., 0::2]
            t_odd  = t[..., 1::2]
            out_even = t_even * cos - t_odd * sin
            out_odd  = t_even * sin + t_odd * cos
            out = torch.empty_like(t)
            out[..., 0::2] = out_even
            out[..., 1::2] = out_odd
            return out

        q = apply_rope(q, cos, sin)  # (B, H,   T, dk)
        k = apply_rope(k, cos, sin)  # (B, HKV, T, dk)

        if use_kv_cache:
            # write current block into cache (cache stores HKV heads)
            k_cache_block = k.transpose(1, 2).contiguous()  # (B, T, HKV, dk)
            v_cache_block = v.transpose(1, 2).contiguous()  # (B, T, HKV, dk)
            self.k_cache[:B, start_posn:start_posn + T] = k_cache_block
            self.v_cache[:B, start_posn:start_posn + T] = v_cache_block

            total = start_posn + T

            # read full history
            k_ctx = self.k_cache[:B, :total].transpose(1, 2).contiguous()  # (B, HKV, total, dk)
            v_ctx = self.v_cache[:B, :total].transpose(1, 2).contiguous()  # (B, HKV, total, dk)

            # expand to H heads
            k_ctx = k_ctx.repeat_interleave(rep, dim=1)  # (B, H, total, dk)
            v_ctx = v_ctx.repeat_interleave(rep, dim=1)  # (B, H, total, dk)

            logits = (q @ k_ctx.transpose(-1, -2)) / math.sqrt(dk)  # (B, H, T, total)

            # correct causal slice for absolute query positions
            logits = logits + self.causal_mask[:, :, start_posn:start_posn + T, :total]

            probs = logits.softmax(dim=-1)
            out = probs @ v_ctx  # (B, H, T, dk)

        else:
            # local attention (no cache)
            k_ctx = k.repeat_interleave(rep, dim=1)  # (B, H, T, dk)
            v_ctx = v.repeat_interleave(rep, dim=1)  # (B, H, T, dk)

            logits = (q @ k_ctx.transpose(-1, -2)) / math.sqrt(dk)  # (B, H, T, T)
            logits = logits + self.causal_mask[:, :, :T, :T]

            probs = logits.softmax(dim=-1)
            out = probs @ v_ctx  # (B, H, T, dk)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, D)
        return self.c_proj(out)
    

class RMSNorm(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.d_model = args.d_model
        self.eps = args.rmsnorm_eps

        self.weight = nn.Parameter(torch.ones(self.d_model))

    def forward(self, x: torch.Tensor):
        op = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        op = self.weight * op
        return op
    
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.attention = CausalSelfAttention(args)
        self.feed_forward = MoELayer(args)
        self.attention_norm = RMSNorm(args)
        self.ff_norm = RMSNorm(args)

    def forward(self, x: torch.Tensor, start_posn: int, use_kv_cache: bool):
        # x shape: (batch_size, sequence_length, d_model)

        h = x + self.attention(self.attention_norm(x), start_posn, use_kv_cache)
        moe_op, router_logits = self.feed_forward(self.ff_norm(h))
        out = h + moe_op
        
        return out, router_logits


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.vocab_size = args.vocab_size
        self.d_model = args.d_model
        self.n_layers = args.n_layers
        self.init_mean = args.init_mean
        self.init_std = args.init_std

        self.token_embeddings = nn.Embedding(self.vocab_size, self.d_model)
        
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(self.n_layers)])
        
        self.norm = RMSNorm(args)
        self.output = nn.Linear(self.d_model, self.vocab_size, bias=False)

        self.init_weights()

        self.output.weight = self.token_embeddings.weight

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(tensor=module.weight, mean=self.init_mean, std=self.init_std)
            elif isinstance(module, nn.Linear):
                if "router" in name:
                    nn.init.normal_(tensor=module.weight, mean=self.init_mean, std=self.init_std)
                else:
                    nn.init.kaiming_normal_(tensor=module.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor, start_posn: int = 0, use_kv_cache: bool = False):
        # x shape: (batch_size, sequence_length)

        h = self.token_embeddings(x) # (batch_size, sequence_length, d_model)

        all_router_logits = []
        for layer in self.layers:
            op, router_logits = layer(h, start_posn, use_kv_cache)
            all_router_logits.append(router_logits)
            h = op

        h = self.norm(h)
        
        return self.output(h), all_router_logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, use_kv_cache: bool = True):
        B, prompt_len = idx.shape
        assert prompt_len <= self.layers[0].attention.max_seq_len
        assert prompt_len + max_new_tokens <= self.layers[0].attention.max_seq_len
        
        if use_kv_cache:
            for layer in self.layers:
                layer.attention.k_cache.zero_()  # type: ignore
                layer.attention.v_cache.zero_()  # type: ignore

            # prefill cache with the full prompt
            logits, _ = self(idx, start_posn=0, use_kv_cache=True)
        else:
            logits = None

        for t in range(max_new_tokens):
            if use_kv_cache:
                if t == 0:
                    # use logits from prefill
                    next_logits = logits[:, -1, :] # type: ignore
                else:
                    # decode one token using cache
                    start_posn = idx.shape[1] - 1
                    logits_step, _ = self(idx[:, -1:], start_posn=start_posn, use_kv_cache=True)
                    next_logits = logits_step[:, -1, :]
            else:
                logits_full, _ = self(idx, start_posn=0, use_kv_cache=False)
                next_logits = logits_full[:, -1, :]

            if temperature == 0.0:
                idx_next = torch.argmax(next_logits, dim=-1, keepdim=True)  # (B, 1)
            else:
                next_logits = next_logits / temperature
                probs = F.softmax(next_logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)          # (B, 1)
 
 
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
