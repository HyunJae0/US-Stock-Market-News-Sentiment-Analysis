import torch
import torch.nn as nn

"""
T5 LayerNorm: 'no additive bias' == RMSNorm
https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html
"""
class T5LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.d_model))
        self.eps = config.layer_norm_eps # 1e-6으로 설정할 것

    def forward(self, hidden_states):
        # variance
        variance = hidden_states.pow(2).to(torch.float32).mean(dim=-1, keepdim=True)
        # normalization # https://docs.pytorch.org/docs/stable/generated/torch.rsqrt.html
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps) # torch.rsqrt # x_i / RMS(x)
        return self.weight * hidden_states # scaling(gamma) # x_i / RMS(x) * gamma_i
