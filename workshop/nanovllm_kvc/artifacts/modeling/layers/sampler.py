import torch
from torch import nn

from sgl_kernel import (
    min_p_sampling_from_probs,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
)

from workshop.nanovllm_kvc.services.engine.io_struct import SamplingInfo
from workshop.nanovllm_kvc.services.utils.context import get_context

class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, sampling_infos: SamplingInfo):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            logits = logits[last_indices].contiguous()
        if sampling_infos.is_greedy_sampling:
            return logits.argmax(dim=-1)
        logits = logits.float().div_(sampling_infos.temperatures[:, None])
        probs = torch.softmax(logits, dim=-1)
        del logits
        
        if sampling_infos.need_min_p_sampling:
            probs = top_k_renorm_prob(probs, sampling_infos.top_ks)
            probs = top_p_renorm_prob(probs, sampling_infos.top_ps)
            sample_tokens = min_p_sampling_from_probs(probs, sampling_infos.min_ps)
        else:
            sample_tokens = top_k_top_p_sampling_from_probs(
                probs.contiguous(), 
                sampling_infos.top_ks,
                sampling_infos.top_ps, 
                filter_apply_order="joint", 
            )        
        return sample_tokens