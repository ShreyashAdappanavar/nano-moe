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

        logits = F.softmax(self.router(x), dim=-1) # (batch_size, sequence_length, num_experts)
        topk = torch.topk(logits, self.top_k) # .values and .indices : (batch_size, sequence_length, top_k)
        weights = topk.values # (batch_size, sequence_length, top_k)
        indices = topk.indices # (batch_size, sequence_length, top_k)

        weights = weights/torch.sum(weights, dim=-1, keepdim=True)

        x_flat = x.view(batch_size * sequence_length, self.d_model) # (batch_size * sequence_length, d_model)
        weights = weights.view(batch_size * sequence_length, self.top_k) # (batch_size * sequence_length, top_k)
        indices = indices.view(batch_size * sequence_length, self.top_k) # (batch_size * sequence_length, top_k)

        routed_output = torch.zeros_like(x_flat) # (batch_size * sequence_length, d_model)

        for i, curr_expert in enumerate(self.experts):
            row, index = torch.where(indices==i)
            curr_expert_ip = x_flat[row] # (selected_rows, d_model)
            curr_expert_op = curr_expert(curr_expert_ip) # (selected_rows, d_model)
            curr_expert_op *= weights[row, index].unsqueeze(-1)
            routed_output[row] += curr_expert_op
            
        routed_output = routed_output.view(batch_size, sequence_length, self.d_model)

        # for i, curr_expert in enumerate(self.experts):
        #     mask = (indices==i) # (batch_size, sequence_length, top_k)

        #     if mask.any():
        #         cur_weights = weights * mask # Now the weight of the second chosen expert will be zero after multiplying it with mask
        #         routed_output += curr_expert(x) * torch.sum(cur_weights, dim=-1, keepdim=True)
                
        final_op = sum(shared_exps_op) + routed_output
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
                                 diagonal=1).view(1, 1, self.max_seq_len, self.max_seq_len).contiguous()
        
        k_cache = torch.zeros((self.batch_size, self.max_seq_len, self.n_kv_heads, self.dk))
        v_cache = torch.zeros((self.batch_size, self.max_seq_len, self.n_kv_heads, self.dk))

        pairs = torch.arange(0, self.dk // 2) # (dk // 2,)
        frequencies = 1/self.rope_theta**(2 * pairs / self.dk) # (dk // 2,)

        positions = torch.arange(self.max_seq_len, dtype=torch.float32) # (max_seq_len,)

        angles = positions.view(len(positions), 1).contiguous() @ frequencies.view(1, len(frequencies)).contiguous() # (max_seq_len, dk // 2)

        sin_matrix = torch.sin(angles) # (max_seq_len, dk // 2)
        cos_matrix = torch.cos(angles) # (max_seq_len, dk // 2)

        self.register_buffer("cos_matrix", cos_matrix)
        self.register_buffer("sin_matrix", sin_matrix)
        self.register_buffer("causal_mask", causal_mask)
        self.register_buffer("k_cache", k_cache, persistent=False)
        self.register_buffer("v_cache", v_cache, persistent=False)

    def forward(self, x: torch.Tensor, start_posn: int = 0, use_kv_cache: bool = False):
        # x shape: (batch_size, curr_seq_length, d_model)
        batch_size, curr_seq_len, _ = x.size()
        
        q = self.q_proj(x) # (batch_size, curr_seq_length, d_model)
        k, v = self.kv_proj(x).split(self.n_kv_heads * self.dk, dim=-1) # (batch_size, curr_seq_length, n_kv_heads * dk)

        q = q.view(batch_size, curr_seq_len, self.n_heads, self.dk).transpose(1,2) # (batch_size, n_heads, curr_seq_length, dk)
        k = k.view(batch_size, curr_seq_len, self.n_kv_heads, self.dk).transpose(1,2) # (batch_size, n_kv_heads, curr_seq_length, dk) 
        v = v.view(batch_size, curr_seq_len, self.n_kv_heads, self.dk).transpose(1,2) # (batch_size, n_kv_heads, curr_seq_length, dk)

        # k = k.repeat_interleave(self.n_heads//self.n_kv_heads, dim=1) # (batch_size, n_heads, curr_seq_length, dk) 
        # v = v.repeat_interleave(self.n_heads//self.n_kv_heads, dim=1) # (batch_size, n_heads, curr_seq_length, dk) 
        
        cos_angles = self.cos_matrix[start_posn : start_posn + curr_seq_len, :].unsqueeze(0).unsqueeze(0) # type: ignore[attr-defined] # (1, 1, curr_seq_len, dk // 2)
        sin_angles = self.sin_matrix[start_posn : start_posn + curr_seq_len, :].unsqueeze(0).unsqueeze(0) # type: ignore[attr-defined] # (1, 1, curr_seq_len, dk // 2)

        q_even = q[:, :, :, 0::2] # (batch_size, n_heads, curr_seq_length, dk//2)
        q_odd = q[:, :, :, 1::2] # (batch_size, n_heads, curr_seq_length, dk//2)

        k_even = k[:, :, :, 0::2] # (batch_size, n_kv_heads, curr_seq_length, dk//2)
        k_odd = k[:, :, :, 1::2] # (batch_size, n_kv_heads, curr_seq_length, dk//2)

        q_rotated_even = q_even*cos_angles - q_odd*sin_angles # (batch_size, n_heads, curr_seq_length, dk//2)
        q_rotated_odd = q_even*sin_angles + q_odd*cos_angles # (batch_size, n_heads, curr_seq_length, dk//2)
        
        k_rotated_even = k_even*cos_angles - k_odd*sin_angles # (batch_size, n_kv_heads, curr_seq_length, dk//2)
        k_rotated_odd = k_even*sin_angles + k_odd*cos_angles # (batch_size, n_kv_heads, curr_seq_length, dk//2)

        q_rotated = torch.zeros_like(q) # (batch_size, n_heads, curr_seq_length, dk)
        k_rotated = torch.zeros_like(k) # (batch_size, n_kv_heads, curr_seq_length, dk)

        q_rotated[:,:,:,0::2] = q_rotated_even # (batch_size, n_heads, curr_seq_length, dk)
        q_rotated[:,:,:,1::2] = q_rotated_odd # (batch_size, n_heads, curr_seq_length, dk)

        k_rotated[:,:,:,0::2] = k_rotated_even # (batch_size, n_kv_heads, curr_seq_length, dk)
        k_rotated[:,:,:,1::2] = k_rotated_odd # (batch_size, n_kv_heads, curr_seq_length, dk)


        if use_kv_cache:
            # 1. Write to Cache (Transpose to match cache shape)
            self.k_cache[:batch_size, start_posn : start_posn + curr_seq_len] = k_rotated.transpose(1, 2) # type:ignore (batch_size, curr_seq_length, n_kv_heads, dk)  
            self.v_cache[:batch_size, start_posn : start_posn + curr_seq_len] = v.transpose(1, 2) # type:ignore (batch_size, curr_seq_length, n_kv_heads, dk)

            # 2. Read History (Retrieve full valid sequence)
            k_final = self.k_cache[:batch_size, : start_posn + curr_seq_len] # type:ignore (batch_size, total_seq_len, n_kv_heads, dk)
            v_final = self.v_cache[:batch_size, : start_posn + curr_seq_len] # type:ignore (batch_size, total_seq_len, n_kv_heads, dk)

            # 3. Transpose back for Attention
            k_final = k_final.transpose(1, 2) # (batch_size, n_kv_heads, total_seq_len, dk)
            v_final = v_final.transpose(1, 2) # (batch_size, n_kv_heads, total_seq_len, dk)

            # 4. Expand History (GQA)
            k_final = k_final.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1) # (batch_size, n_heads, total_seq_len, dk)
            v_final = v_final.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1) # (batch_size, n_heads, total_seq_len, dk)

            # 5. Attention (No Mask needed)
            logits = (q_rotated @ k_final.transpose(-1,-2)) / math.sqrt(self.dk) # (batch_size, n_heads, curr_seq_length, total_seq_len)

            if curr_seq_len > 1:
                logits += self.causal_mask[:, :, :curr_seq_len, :curr_seq_len] # type:ignore

        else:
            # 1. Expand Local Tensors (GQA)
            k_final = k_rotated.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1) # (batch_size, n_heads, curr_seq_length, dk)
            v_final = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1) # (batch_size, n_heads, curr_seq_length, dk)
            
            # 2. Attention
            logits = (q_rotated @ k_final.transpose(-1,-2)) / math.sqrt(self.dk) # (batch_size, n_heads, curr_seq_length, curr_seq_length)
            
            # 3. Apply Mask
            logits += self.causal_mask[:, :, :curr_seq_len, :curr_seq_len] # type:ignore (batch_size, n_heads, curr_seq_length, curr_seq_length)

        # Final Aggregation (Shared)
        probs = logits.softmax(dim=-1) # (batch_size, n_heads, curr_seq_length, total_seq_len OR curr_seq_length)
        output = probs @ v_final # (batch_size, n_heads, curr_seq_length, dk)
        
        output = output.transpose(1,2).contiguous().view(batch_size, curr_seq_len, self.d_model) # (batch_size, curr_seq_length, d_model)
        
        return self.c_proj(output)
    

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
        # idx shape: (batch_size, sequence_length)
        
        prompt_len = idx.shape[1]

        for _ in range(max_new_tokens):
           
            if use_kv_cache and idx.shape[1] > prompt_len:
                # Fast Mode: Only feed the last generated token
                x_input = idx[:, -1:]
                start_posn = idx.shape[1] - 1
            else:
                # Full Mode: Feed everything (Prefill or No-Cache)
                x_input = idx
                start_posn = 0

            # Forward pass
            logits, _ = self(x_input, start_posn=start_posn, use_kv_cache=use_kv_cache)
            
            # Focus only on the last token's logits for prediction
            logits = logits[:, -1, :] / temperature
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx