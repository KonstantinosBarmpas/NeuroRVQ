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
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            # Auxilatory Loss
            loss = loss + 0.4 * F.mse_loss(quantized, residual.detach())

            all_indices.append(indices)
            all_losses.append(loss)
           
            # --- Codebook usage tracking ---
            unique_codes = torch.unique(indices)
            usage_ratio = unique_codes.numel() / total_codes
            usage_ratios.append(float(usage_ratio))

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        out_losses = out_losses.mean()
        
        return quantized_out, out_indices, out_losses, usage_ratios

    def encode(self, x):
        residual = x
        all_indices = []
        n_q = len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices):
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
