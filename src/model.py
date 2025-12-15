import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelArgs

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
        self.router = nn.Linear(self.d_model, self.num_experts)
    
    def forward(self, x: torch.Tensor):
        # x shape: (batch_size, sequence_length, d_model)

        shared_exps_op = []
        for i in self.shared_experts:
                shared_exps_op.append(i(x))

        topk = torch.topk(F.softmax(self.router(x), dim=-1), self.top_k) # .values and .indices : (batch_size, sequence_length, top_k)
        weights = topk.values # (batch_size, sequence_length, top_k)
        indices = topk.indices # (batch_size, sequence_length, top_k)

        weights = weights/torch.sum(weights, dim=-1, keepdim=True)

        routed_output = torch.zeros_like(x)
        for i, curr_expert in enumerate(self.experts):
            mask = (indices==i) # (batch_size, sequence_length, top_k)

            if mask.any():
                cur_weights = weights * mask # Now the weight of the second chosen expert will be zero after multiplying it with mask
                routed_output += curr_expert(x) * torch.sum(cur_weights, dim=-1, keepdim=True)
                

        
        final_op = sum(shared_exps_op) + routed_output
        return final_op