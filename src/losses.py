import torch
import torch.nn.functional as F
from .config import ModelArgs
from typing import List

def layer_stats(layer_logits: torch.Tensor, args: ModelArgs):
    """
    Docstring for layer_stats
    
    :param layer_logits: router logits per layer
    :type layer_logits: torch.Tensor

    Returns load balancing loss and z loss for the router logics of the current layer
    """
    batch_size, sequence_len = layer_logits.shape[0], layer_logits.shape[1]
    # layer_logits: (batch_size, sequence_length, num_experts)
    p = F.softmax(layer_logits, dim=-1) # (batch_size, sequence_length, num_experts)
    topk = torch.topk(p, args.top_k, dim=-1).indices # (batch_size, sequence_length, top_k)

    importances = torch.mean(p, dim=(0,1))
    tmp = torch.bincount(topk.reshape(batch_size*sequence_len*args.top_k), minlength=args.num_experts)
    tmp = tmp.to(importances.dtype).to(layer_logits.device)
    load = tmp / (batch_size*sequence_len*args.top_k)
    
    Llb = args.num_experts * (torch.sum(importances * load))
    Lz = torch.mean(torch.square(torch.logsumexp(layer_logits, dim=-1)))

    return Llb, Lz, importances, load

def router_loss(router_logits: List[torch.Tensor], args: ModelArgs, debug_flag: bool = False):
    if len(router_logits) == 0:
        raise ValueError("router_logits is empty")

    ref = router_logits[0]
    Llb_total = ref.new_zeros(())
    Lz_total = ref.new_zeros(())

    if debug_flag:
        all_importances = []
        all_loads = []

    for i in range(len(router_logits)):
        Llb, Lz, importances, load = layer_stats(router_logits[i], args)
        Llb_total += Llb
        Lz_total += Lz

        if debug_flag:
            all_importances.append(importances)
            all_loads.append(load)
    
    Llb_final = Llb_total / len(router_logits)
    Lz_final = Lz_total / len(router_logits)

    if debug_flag:
        all_importances = torch.stack(all_importances, dim=0)
        all_loads = torch.stack(all_loads, dim=0)
        return Llb_final, Lz_final, all_importances, all_loads

    else: return Llb_final, Lz_final

def ce_loss_fn(logits: torch.Tensor, targets: torch.Tensor, args: ModelArgs):
    # logits: (batch_size, sequence_length, vocab_size)
    # target: (batch_size, sequence_length)

    assert targets.dtype == torch.long, "The dtype of targets is NOT torch.long"
    assert logits.shape[-1] == args.vocab_size, "logits.shape[-1] =/= args.vocab_size ! Why?"

    batch_size, sequence_len, curr_vocab_size = logits.shape[0], logits.shape[1], logits.shape[-1]

    logits = logits.reshape(batch_size * sequence_len, curr_vocab_size)
    targets = targets.reshape(batch_size * sequence_len)

    ce_loss = F.cross_entropy(logits, targets)

    return ce_loss

def compute_loss(logits: torch.Tensor, router_logits: List[torch.Tensor], targets: torch.Tensor, args: ModelArgs,
                 alpha: float = 1e-2, beta: float = 1e-3, debug_flag: bool = False):
       
    if not logits.is_contiguous(): logits = logits.contiguous()

    ce_loss_final = ce_loss_fn(logits, targets, args)

    if debug_flag:
        Llb_final, Lz_final, all_importances, all_loads = router_loss(router_logits, args, debug_flag=True)
    
    else: Llb_final, Lz_final = router_loss(router_logits, args)

    final_loss = ce_loss_final + alpha * Llb_final + beta * Lz_final

    if debug_flag: 
        return_dict = {"all_importances": all_importances,
                       "all_loads": all_loads, 
                       "ce_loss": ce_loss_final.item(),
                       "Llb_final": Llb_final.item(),
                       "Lz_final": Lz_final.item()}
            
    else: 
        return_dict = {"ce_loss": ce_loss_final.item(),
                       "Llb_final": Llb_final.item(),
                       "Lz_final": Lz_final.item()}
        
    return final_loss, return_dict