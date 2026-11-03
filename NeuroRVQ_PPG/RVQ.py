'''
Residual Vector Quantization Implementation.
Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from norm_ema_quantizer import NormEMAVectorQuantizer

class ResidualVectorQuantization(nn.Module):
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [NormEMAVectorQuantizer(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x):
        quantized_out = torch.zeros_like(x)
        residual = x

        all_losses = []
        all_indices = []
        n_q = len(self.layers)
        
        usage_ratios = []  # Track usage per quantizer
        total_codes = self.layers[0].num_tokens

        for layer in self.layers[:n_q]:
            quantized, loss, indices = layer(residual)
            # Auxilatory Loss
            loss = loss + 0.4 * F.mse_loss(quantized, residual.detach())
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)
           
            # --- Codebook usage tracking ---
            unique_codes = torch.unique(indices)
            usage_ratio = unique_codes.numel() / total_codes
            usage_ratios.append(float(usage_ratio))

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        out_losses = out_losses.mean()
        
        return quantized_out, out_indices, out_losses, usage_ratios
