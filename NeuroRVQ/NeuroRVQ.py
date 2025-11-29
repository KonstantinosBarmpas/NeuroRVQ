import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import math
from functools import partial
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from NeuroRVQ_modules import Block, trunc_normal_
from RVQ import ResidualVectorQuantization

def inverse_fft_cos_sin(fft_amp, fft_sin_pha, fft_cos_pha):
    """
    Inverse FFT function using sin and cos
    :param fft_amp: amplitude
    :param fft_sin_pha: sine
    :param fft_cos_pha: cosine
    :return: inverse fft in time
    """
    imag = fft_amp * fft_sin_pha
    real = fft_amp * fft_cos_pha
    fft_y = torch.complex(real, imag)
    y = torch.fft.ifft(fft_y)
    return y

class PatchEmbed(nn.Module):
    """
    Project each codebook to the patch latent space
    :param in_chans: number of input channels
    :param embed_dim: dimension of embedding space
    """
    def __init__(self, in_chans=1, embed_dim=200):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
        
class MultiDimentionalTemporalConv(nn.Module):
    """
    EEG to Patch Embedding - Multidimentional Temporal Filtering
    :param in_chans: number of input channels
    :param out_chans: number of output channels
    """
    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        # Inception Style Seperate Branches - Group 1 #
        # Branch 1: >10 Hz assuming fs=200Hz
        self.conv1_1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 21), padding=(0, 10))
        self.norm1_1 = nn.GroupNorm(4, out_chans)
        self.pool1_1 = nn.AvgPool2d(kernel_size=(1, 2))

        # Branch 2: >13 Hz assuming fs=200Hz
        self.conv1_2 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), padding=(0, 7))
        self.norm1_2 = nn.GroupNorm(4, out_chans)
        self.pool1_2 = nn.AvgPool2d(kernel_size=(1, 2))

        # Branch 3: >20 Hz assuming fs=200Hz
        self.conv1_3 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 9), padding=(0, 4))
        self.norm1_3 = nn.GroupNorm(4, out_chans)
        self.pool1_3 = nn.AvgPool2d(kernel_size=(1, 2))

        # Branch 4: >40 Hz assuming fs=200Hz
        self.conv1_4 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 5), padding=(0, 2))
        self.norm1_4 = nn.GroupNorm(4, out_chans)
        self.pool1_4 = nn.AvgPool2d(kernel_size=(1, 2))
        self.gelu1 = nn.GELU()
        
        # Inception Style Seperate Branches - Group 2 #
        self.conv2_1 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 9), padding=(0, 4))
        self.norm2_1 = nn.GroupNorm(4, out_chans)
        self.pool2_1 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv2_2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 7), padding=(0, 3))
        self.norm2_2 = nn.GroupNorm(4, out_chans)
        self.pool2_2 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv2_3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 5), padding=(0, 2))
        self.norm2_3 = nn.GroupNorm(4, out_chans)
        self.pool2_3 = nn.AvgPool2d(kernel_size=(1, 4))
        
        self.conv2_4 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm2_4 = nn.GroupNorm(4, out_chans)
        self.pool2_4 = nn.AvgPool2d(kernel_size=(1, 4))
        self.gelu2 = nn.GELU()
                                
    def forward(self, x):
        x = rearrange(x, 'B N A T -> B (N A) T')
        x = x.unsqueeze(1)
        
        # First layer of filtering - Group 1
        x1 = self.pool1_1(self.gelu1(self.norm1_1(self.conv1_1(x))))
        x2 = self.pool1_2(self.gelu1(self.norm1_2(self.conv1_2(x))))
        x3 = self.pool1_3(self.gelu1(self.norm1_3(self.conv1_3(x))))
        x4 = self.pool1_4(self.gelu1(self.norm1_4(self.conv1_4(x))))
        
        # First layer of filtering - Group 2
        x1 = self.pool2_1(self.gelu2(self.norm2_1(self.conv2_1(x1))))
        x2 = self.pool2_2(self.gelu2(self.norm2_2(self.conv2_2(x2))))
        x3 = self.pool2_3(self.gelu2(self.norm2_3(self.conv2_3(x3))))
        x4 = self.pool2_4(self.gelu2(self.norm2_4(self.conv2_4(x4))))
        
        # Re-arrange
        x1 = rearrange(x1, 'B C NA T -> B NA (T C)')
        x2 = rearrange(x2, 'B C NA T -> B NA (T C)')
        x3 = rearrange(x3, 'B C NA T -> B NA (T C)')
        x4 = rearrange(x4, 'B C NA T -> B NA (T C)')
        return x1, x2, x3, x4

class NeuroRVQFM(nn.Module):
    """
    NeuroRVQ Foundation Model Class
    """
    def __init__(self, n_patches=256, patch_size=200, in_chans=1, out_chans=8, num_classes=5,
                 embed_dim=200, depth=12, num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., init_values=None, init_scale=0.001,
                 n_global_electrodes=127, vocab_size=8192, use_as_encoder=True, use_for_pretraining=False):
       
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.use_for_pretraining = use_for_pretraining
        self.use_as_encoder = use_as_encoder
        # Not necessary - legacy code
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # To identify whether patch_embed layer is used as tokenizer/encoder or as a decoder
        if use_as_encoder or use_for_pretraining:
            self.patch_embed = MultiDimentionalTemporalConv(out_chans=out_chans)
        else:
            self.patch_embed_1 = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
            self.patch_embed_2 = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
            self.patch_embed_3 = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
            self.patch_embed_4 = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(n_global_electrodes + 1, embed_dim), requires_grad=True)
        self.time_embed = nn.Parameter(torch.zeros(n_patches, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=nn.LayerNorm,
                init_values=init_values, window_size=None)
            for i in range(depth)])

        # If used for pre-training we need heads to predict 8 * 4 tokens per input sample
        # TODO: Modular
        if use_for_pretraining:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.norm_pre = nn.LayerNorm(embed_dim)
            self.head_pre_1 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_2 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_3 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_4 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_5 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_6 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_7 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_8 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_9 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_10 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_11 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_12 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_13 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_14 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_15 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_16 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_17 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_18 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_19 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_20 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_21 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_22 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_23 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_24 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_25 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_26 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_27 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_28 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_29 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_30 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_31 = nn.Linear(embed_dim, vocab_size)
            self.head_pre_32 = nn.Linear(embed_dim, vocab_size)
        else:
            self.norm = nn.Identity()
            self.fc_norm_1 = nn.LayerNorm(embed_dim)
            self.head_1 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.fc_norm_2 = nn.LayerNorm(embed_dim)
            self.head_2 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.fc_norm_3 = nn.LayerNorm(embed_dim)
            self.head_3 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.fc_norm_4 = nn.LayerNorm(embed_dim)
            self.head_4 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize the weights of the network
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # Initialization ONLY during pretraining
        if use_for_pretraining:
            trunc_normal_(self.mask_token, std=.02)
            trunc_normal_(self.head_pre_1.weight, std=.02)
            trunc_normal_(self.head_pre_2.weight, std=.02)
            trunc_normal_(self.head_pre_3.weight, std=.02)
            trunc_normal_(self.head_pre_4.weight, std=.02)
            trunc_normal_(self.head_pre_5.weight, std=.02)
            trunc_normal_(self.head_pre_6.weight, std=.02)
            trunc_normal_(self.head_pre_7.weight, std=.02)
            trunc_normal_(self.head_pre_8.weight, std=.02)
            trunc_normal_(self.head_pre_9.weight, std=.02)
            trunc_normal_(self.head_pre_10.weight, std=.02)
            trunc_normal_(self.head_pre_11.weight, std=.02)
            trunc_normal_(self.head_pre_12.weight, std=.02)
            trunc_normal_(self.head_pre_13.weight, std=.02)
            trunc_normal_(self.head_pre_14.weight, std=.02)
            trunc_normal_(self.head_pre_15.weight, std=.02)
            trunc_normal_(self.head_pre_16.weight, std=.02)
            trunc_normal_(self.head_pre_17.weight, std=.02)
            trunc_normal_(self.head_pre_18.weight, std=.02)
            trunc_normal_(self.head_pre_19.weight, std=.02)
            trunc_normal_(self.head_pre_20.weight, std=.02)
            trunc_normal_(self.head_pre_21.weight, std=.02)
            trunc_normal_(self.head_pre_22.weight, std=.02)
            trunc_normal_(self.head_pre_23.weight, std=.02)
            trunc_normal_(self.head_pre_24.weight, std=.02)
            trunc_normal_(self.head_pre_25.weight, std=.02)
            trunc_normal_(self.head_pre_26.weight, std=.02)
            trunc_normal_(self.head_pre_27.weight, std=.02)
            trunc_normal_(self.head_pre_28.weight, std=.02)
            trunc_normal_(self.head_pre_29.weight, std=.02)
            trunc_normal_(self.head_pre_30.weight, std=.02)
            trunc_normal_(self.head_pre_31.weight, std=.02)
            trunc_normal_(self.head_pre_32.weight, std=.02)
        else:
            if isinstance(self.head_1, nn.Linear):
                trunc_normal_(self.head_1.weight, std=.02)
            if isinstance(self.head_1, nn.Linear):
                self.head_1.weight.data.mul_(init_scale)
                self.head_1.bias.data.mul_(init_scale)
            if isinstance(self.head_2, nn.Linear):
                trunc_normal_(self.head_2.weight, std=.02)
            if isinstance(self.head_2, nn.Linear):
                self.head_2.weight.data.mul_(init_scale)
                self.head_2.bias.data.mul_(init_scale)
            if isinstance(self.head_3, nn.Linear):
                trunc_normal_(self.head_3.weight, std=.02)
            if isinstance(self.head_3, nn.Linear):
                self.head_3.weight.data.mul_(init_scale)
                self.head_3.bias.data.mul_(init_scale)
            if isinstance(self.head_4, nn.Linear):
                trunc_normal_(self.head_4.weight, std=.02)
            if isinstance(self.head_4, nn.Linear):
                self.head_4.weight.data.mul_(init_scale)
                self.head_4.bias.data.mul_(init_scale)

        self.apply(self._init_weights)
        self.fix_init_weight()

    # Function to initialize the weights of the network
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # Function to initialize the weights of the network
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Get number of layers from the transformer
    def get_num_layers(self):
        return len(self.blocks)

    # Get classification head
    def get_classifier(self):
        return self.head

    # Reset the classification head
    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.fc_norm = nn.LayerNorm(self.embed_dim * 4) # multiply dim by 4 for concat [x1,x2,x3,x4] 
        self.head = nn.Linear(self.embed_dim * 4, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def forward(self, x, temporal_embedding_ix, spatial_embedding_ix, return_patch_tokens=False, bool_masked_pos=None, use_for_pretraining=False, branch_idx=0):
        '''
        x: [batch size, number of electrodes, number of patches, patch size]
        For example, for an EEG sample of 4 seconds with 64 electrodes, x will be [batch size, 64, 4, 200]
        '''
        if (self.use_as_encoder):
            x1, x2, x3, x4 = self.patch_embed(x)
            x = x1
        else:
            if (branch_idx==0):
                x = self.patch_embed_1(x)
            elif (branch_idx==1):
                x = self.patch_embed_2(x)
            elif (branch_idx==2):
                x = self.patch_embed_3(x)
            elif (branch_idx==3):
                x = self.patch_embed_4(x)
            
        batch_size, seq_len, _ = x.size()

        # Concatenate the cls token - Legacy code
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # ONLY in Pre-Training - Masking technique used in LaBraM
        if (use_for_pretraining or bool_masked_pos is not None):
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            symmetric_bool_masked_pos = ~bool_masked_pos
            w_symmetric = symmetric_bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            
            x_symmetric_1 = x1 * (1 - w_symmetric) + mask_token * w_symmetric
            x1 = x1 * (1 - w) + mask_token * w
            x_symmetric_1 = torch.cat((cls_tokens, x_symmetric_1), dim=1)

            x_symmetric_2 = x2 * (1 - w_symmetric) + mask_token * w_symmetric
            x2 = x2 * (1 - w) + mask_token * w
            x_symmetric_2 = torch.cat((cls_tokens, x_symmetric_2), dim=1)

            x_symmetric_3 = x3 * (1 - w_symmetric) + mask_token * w_symmetric
            x3 = x3 * (1 - w) + mask_token * w
            x_symmetric_3 = torch.cat((cls_tokens, x_symmetric_3), dim=1)

            x_symmetric_4 = x4 * (1 - w_symmetric) + mask_token * w_symmetric
            x4 = x4 * (1 - w) + mask_token * w
            x_symmetric_4 = torch.cat((cls_tokens, x_symmetric_4), dim=1)

        if (self.use_as_encoder):
            x1 = torch.cat((cls_tokens, x1), dim=1)
            x2 = torch.cat((cls_tokens, x2), dim=1)
            x3 = torch.cat((cls_tokens, x3), dim=1)
            x4 = torch.cat((cls_tokens, x4), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        # Pad the spatial_embedding_ix - spatial_embedding_ix: (batch, n_patches), spatial_embedding: (n_electrodes + 1, embed_dim)
        spatial_embedding_ix = F.pad(input=spatial_embedding_ix, pad=(1, 0), mode='constant', value=0)  # for cls token (batch_size, n_patches + 1)
        # Gets the corresponding pos_embed
        spatial_embedding = self.pos_embed[spatial_embedding_ix.reshape(-1), :]  # (batch_size * (n_patches + 1), embed_dim)
        spatial_embedding = spatial_embedding.reshape(spatial_embedding_ix.shape[0], spatial_embedding_ix.shape[1], spatial_embedding.shape[-1])  # (batch_size, n_patches + 1, embed_dim)
        
        if (self.use_as_encoder):
            x1 = x1 + spatial_embedding
            x2 = x2 + spatial_embedding
            x3 = x3 + spatial_embedding
            x4 = x4 + spatial_embedding
        else:
            x = x + spatial_embedding

        #  temporal_embedding_ix: (batch, n_patches), temporal_embedding: (n_patches, embed_dim)
        temporal_embedding = self.time_embed[temporal_embedding_ix.reshape(-1), :]  # (batch_size * (n_patches), embed_dim)
        temporal_embedding = temporal_embedding.reshape(temporal_embedding_ix.shape[0], temporal_embedding_ix.shape[1], temporal_embedding.shape[-1])  # (batch_size, n_patches, embed_dim)
        
        if (self.use_as_encoder):
            x1[:, 1:, :] += temporal_embedding
            x1 = self.pos_drop(x1)
            x2[:, 1:, :] += temporal_embedding
            x2 = self.pos_drop(x2)
            x3[:, 1:, :] += temporal_embedding
            x3 = self.pos_drop(x3)
            x4[:, 1:, :] += temporal_embedding
            x4 = self.pos_drop(x4)
        else:
            x[:, 1:, :] += temporal_embedding
            x = self.pos_drop(x)
            
        if (self.use_as_encoder):
            # Pass the transformer blocks
            for i, x in enumerate([x1, x2, x3, x4]):
                for blk in self.blocks:
                    x = blk(x)
                    
                if (use_for_pretraining or bool_masked_pos is not None):
                    x = self.norm_pre(x)
                else:
                    x = self.norm(x)
                # All except cls token
                if i == 0:
                    x1 = x[:, 1:, :]
                elif i == 1:
                    x2 = x[:, 1:, :]
                elif i == 2:
                    x3 = x[:, 1:, :]
                else:
                    x4 = x[:, 1:, :]
        else:
            # Pass the transformer blocks
            for blk in self.blocks:
                x = blk(x)
            if (use_for_pretraining or bool_masked_pos is not None):
                x = self.norm_pre(x)
            else:
                x = self.norm(x)

            # All except cls token
            x = x[:, 1:, :]

        if (use_for_pretraining or bool_masked_pos is not None):
            for i, x_symmetric in enumerate([x_symmetric_1, x_symmetric_2, x_symmetric_3, x_symmetric_4]):
                x_symmetric += spatial_embedding
                x_symmetric[:, 1:, :] += temporal_embedding
                x_symmetric = self.pos_drop(x_symmetric)
                for blk in self.blocks:
                    x_symmetric = blk(x_symmetric)
                x_symmetric = self.norm_pre(x_symmetric)
                # All except cls token
                if i == 0:
                    x_symmetric_1 = x_symmetric[:, 1:, :]
                elif i == 1:
                    x_symmetric_2 = x_symmetric[:, 1:, :]
                elif i == 2:
                    x_symmetric_3 = x_symmetric[:, 1:, :]
                else:
                    x_symmetric_4 = x_symmetric[:, 1:, :]

        # ONLY in Pre-Training
        if (use_for_pretraining or bool_masked_pos is not None):
        
            logits = torch.stack([
                # x1 heads
                self.head_pre_1(x1[bool_masked_pos]),
                self.head_pre_2(x1[bool_masked_pos]),
                self.head_pre_3(x1[bool_masked_pos]),
                self.head_pre_4(x1[bool_masked_pos]),
                self.head_pre_5(x1[bool_masked_pos]),
                self.head_pre_6(x1[bool_masked_pos]),
                self.head_pre_7(x1[bool_masked_pos]),
                self.head_pre_8(x1[bool_masked_pos]),

                # x2 heads
                self.head_pre_9(x2[bool_masked_pos]),
                self.head_pre_10(x2[bool_masked_pos]),
                self.head_pre_11(x2[bool_masked_pos]),
                self.head_pre_12(x2[bool_masked_pos]),
                self.head_pre_13(x2[bool_masked_pos]),
                self.head_pre_14(x2[bool_masked_pos]),
                self.head_pre_15(x2[bool_masked_pos]),
                self.head_pre_16(x2[bool_masked_pos]),

                # x3 heads
                self.head_pre_17(x3[bool_masked_pos]),
                self.head_pre_18(x3[bool_masked_pos]),
                self.head_pre_19(x3[bool_masked_pos]),
                self.head_pre_20(x3[bool_masked_pos]),
                self.head_pre_21(x3[bool_masked_pos]),
                self.head_pre_22(x3[bool_masked_pos]),
                self.head_pre_23(x3[bool_masked_pos]),
                self.head_pre_24(x3[bool_masked_pos]),

                # x4 heads
                self.head_pre_25(x4[bool_masked_pos]),
                self.head_pre_26(x4[bool_masked_pos]),
                self.head_pre_27(x4[bool_masked_pos]),
                self.head_pre_28(x4[bool_masked_pos]),
                self.head_pre_29(x4[bool_masked_pos]),
                self.head_pre_30(x4[bool_masked_pos]),
                self.head_pre_31(x4[bool_masked_pos]),
                self.head_pre_32(x4[bool_masked_pos]),
            ], dim=0)
            
            logits_symmetric = torch.stack([
                # x_symmetric_1 heads
                self.head_pre_1(x_symmetric_1[~bool_masked_pos]),
                self.head_pre_2(x_symmetric_1[~bool_masked_pos]),
                self.head_pre_3(x_symmetric_1[~bool_masked_pos]),
                self.head_pre_4(x_symmetric_1[~bool_masked_pos]),
                self.head_pre_5(x_symmetric_1[~bool_masked_pos]),
                self.head_pre_6(x_symmetric_1[~bool_masked_pos]),
                self.head_pre_7(x_symmetric_1[~bool_masked_pos]),
                self.head_pre_8(x_symmetric_1[~bool_masked_pos]),

                # x_symmetric_2 heads
                self.head_pre_9(x_symmetric_2[~bool_masked_pos]),
                self.head_pre_10(x_symmetric_2[~bool_masked_pos]),
                self.head_pre_11(x_symmetric_2[~bool_masked_pos]),
                self.head_pre_12(x_symmetric_2[~bool_masked_pos]),
                self.head_pre_13(x_symmetric_2[~bool_masked_pos]),
                self.head_pre_14(x_symmetric_2[~bool_masked_pos]),
                self.head_pre_15(x_symmetric_2[~bool_masked_pos]),
                self.head_pre_16(x_symmetric_2[~bool_masked_pos]),

                # x_symmetric_3 heads
                self.head_pre_17(x_symmetric_3[~bool_masked_pos]),
                self.head_pre_18(x_symmetric_3[~bool_masked_pos]),
                self.head_pre_19(x_symmetric_3[~bool_masked_pos]),
                self.head_pre_20(x_symmetric_3[~bool_masked_pos]),
                self.head_pre_21(x_symmetric_3[~bool_masked_pos]),
                self.head_pre_22(x_symmetric_3[~bool_masked_pos]),
                self.head_pre_23(x_symmetric_3[~bool_masked_pos]),
                self.head_pre_24(x_symmetric_3[~bool_masked_pos]),

                # x_symmetric_4 heads
                self.head_pre_25(x_symmetric_4[~bool_masked_pos]),
                self.head_pre_26(x_symmetric_4[~bool_masked_pos]),
                self.head_pre_27(x_symmetric_4[~bool_masked_pos]),
                self.head_pre_28(x_symmetric_4[~bool_masked_pos]),
                self.head_pre_29(x_symmetric_4[~bool_masked_pos]),
                self.head_pre_30(x_symmetric_4[~bool_masked_pos]),
                self.head_pre_31(x_symmetric_4[~bool_masked_pos]),
                self.head_pre_32(x_symmetric_4[~bool_masked_pos]),
            ], dim=0)
        
            return  logits, logits_symmetric
        else:
            # ONLY in RVQ
            if return_patch_tokens:
                if (self.use_as_encoder):
                    return self.head_1(self.fc_norm_1(x1)), self.head_2(self.fc_norm_2(x2)), self.head_3(self.fc_norm_3(x3)), self.head_4(self.fc_norm_4(x4)), _
                else:
                    if (branch_idx==0):
                        return self.head_1(self.fc_norm_1(x)), _
                    elif (branch_idx==1):
                        return self.head_2(self.fc_norm_2(x)), _
                    elif (branch_idx==2):
                        return self.head_3(self.fc_norm_3(x)), _
                    elif (branch_idx==3):
                        return self.head_4(self.fc_norm_4(x)), _
            else:
                # ONLY in Fine-Tune
                x = torch.concat([x1,x2,x3,x4], dim=-1)
                return self.head(self.fc_norm(x.mean(1))), _



class NeuroRVQTokenizer(nn.Module):
    """
    NeuroRVQ Tokenizer
    """
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 n_code,
                 code_dim,
                 decoder_out_dim
                 ):

        super().__init__()
        self.patch_size = encoder_config['patch_size']
        self.code_dim = code_dim

        # Encoder layer of NeuroRVQFM
        self.encoder = NeuroRVQFM(n_patches=encoder_config['n_patches'], patch_size=encoder_config['patch_size'],
            in_chans=encoder_config['in_chans'], out_chans=encoder_config['out_chans_encoder'],
            num_classes = encoder_config['num_classes'], embed_dim=encoder_config['embed_dim'],
            depth=encoder_config['depth'], num_heads=encoder_config['num_heads'],
            mlp_ratio=encoder_config['mlp_ratio'], qkv_bias=encoder_config['qkv_bias'],
            qk_norm=partial(nn.LayerNorm, eps=1e-6), drop_rate=encoder_config['drop_rate'],
            attn_drop_rate=encoder_config['attn_drop_rate'], drop_path_rate=encoder_config['drop_path_rate'],
            init_values=encoder_config['init_values'], init_scale=encoder_config['init_scale'],
            n_global_electrodes=encoder_config['n_global_electrodes'], vocab_size=n_code,
            use_as_encoder=True, use_for_pretraining = False)

        # Decoder layer of NeuroRVQFM
        self.decoder = NeuroRVQFM(n_patches=decoder_config['n_patches'], patch_size=decoder_config['patch_size'],
            in_chans=decoder_config['in_chans'], out_chans=0,
            num_classes = decoder_config['num_classes'], embed_dim=decoder_config['embed_dim'],
            depth=decoder_config['depth'], num_heads=decoder_config['num_heads'],
            mlp_ratio=decoder_config['mlp_ratio'], qkv_bias=decoder_config['qkv_bias'],
            qk_norm=partial(nn.LayerNorm, eps=1e-6), drop_rate=decoder_config['drop_rate'],
            attn_drop_rate=decoder_config['attn_drop_rate'], drop_path_rate=decoder_config['drop_path_rate'],
            init_values=decoder_config['init_values'], init_scale=decoder_config['init_scale'],
            n_global_electrodes=decoder_config['n_global_electrodes'], vocab_size=n_code,
            use_as_encoder=False, use_for_pretraining = False)
        
        self.quantize_1 = ResidualVectorQuantization(num_quantizers = 8,
            n_embed=n_code, embedding_dim=code_dim, beta=1.0, kmeans_init=True, decay=0.99,
        )
        self.quantize_2 = ResidualVectorQuantization(num_quantizers = 8,
            n_embed=n_code, embedding_dim=code_dim, beta=1.0, kmeans_init=True, decay=0.99,
        )
        self.quantize_3 = ResidualVectorQuantization(num_quantizers = 8,
            n_embed=n_code, embedding_dim=code_dim, beta=1.0, kmeans_init=True, decay=0.99,
        )
        self.quantize_4 = ResidualVectorQuantization(num_quantizers = 8,
            n_embed=n_code, embedding_dim=code_dim, beta=1.0, kmeans_init=True, decay=0.99,
        )

        # Output dimension of the decoder layer
        self.decoder_out_dim = decoder_out_dim

        # Encoding head after the encoder transformer
        self.encode_task_layer_1 = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], code_dim)
        )
        self.encode_task_layer_2 = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], code_dim)
        )
        self.encode_task_layer_3 = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], code_dim)
        )
        self.encode_task_layer_4 = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], code_dim)
        )
        self.encode_task_layer_1.apply(self._init_weights)
        self.encode_task_layer_2.apply(self._init_weights)
        self.encode_task_layer_3.apply(self._init_weights)
        self.encode_task_layer_4.apply(self._init_weights)

        # Decoding heads after the decoder transformer
        self.decode_task_layer_amplitude = nn.Sequential(
            nn.Linear(4*decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.GELU(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        self.decode_task_layer_angle_sin = nn.Sequential(
            nn.Linear(4*decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
            nn.Tanh()
        )
        self.decode_task_layer_angle_cos = nn.Sequential(
            nn.Linear(4*decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
            nn.Tanh()
        )

        # Initialize model weights
        self.decode_task_layer_amplitude.apply(self._init_weights)
        self.decode_task_layer_angle_sin.apply(self._init_weights)
        self.decode_task_layer_angle_cos.apply(self._init_weights)

        # MSE loss function
        self.loss_fn = F.mse_loss

    # Function to initialize the weights of the network
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 'decoder.time_embed',
                'encoder.cls_token', 'encoder.pos_embed', 'encoder.time_embed'}

    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, temporal_embedding_ix, spatial_embedding_ix):
        quantize, code_ind, loss, usage_ratios = self.encode(data, temporal_embedding_ix, spatial_embedding_ix)
        # Convert [8, B*P] to [8, B, P]
        code_inds = [code_ind_ix.view(8, data.shape[0], -1) for code_ind_ix in code_ind]
        # Stack all codebooks [4, 8, B, P]
        stacked_code_inds = torch.stack(code_inds, dim=0)
        quantize_vecs = [rearrange(quantize_ix, 'b d a c -> b (a c) d').contiguous() for quantize_ix in quantize]
        output = {}
        output['token'] = stacked_code_inds
        output['input_img'] = data
        output['quantize'] = quantize_vecs
        return output

    def encode(self, x, temporal_embedding_ix, spatial_embedding_ix):
        batch_size, n, a, t = x.shape
        encoder_features_1, encoder_features_2, encoder_features_3, encoder_features_4, _ = self.encoder(x, temporal_embedding_ix=temporal_embedding_ix, spatial_embedding_ix=spatial_embedding_ix, return_patch_tokens=True)
            
        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features_1 = self.encode_task_layer_1(encoder_features_1.type_as(self.encode_task_layer_1[-1].weight))
            to_quantizer_features_2 = self.encode_task_layer_2(encoder_features_2.type_as(self.encode_task_layer_2[-1].weight))
            to_quantizer_features_3 = self.encode_task_layer_3(encoder_features_3.type_as(self.encode_task_layer_3[-1].weight))
            to_quantizer_features_4 = self.encode_task_layer_4(encoder_features_4.type_as(self.encode_task_layer_4[-1].weight))
            
        N = to_quantizer_features_1.shape[1]
        h, w = n, N // n
            
        # reshape tokens to feature maps for patch embed in decoder
        to_quantizer_features_1 = rearrange(to_quantizer_features_1, 'b (h w) c -> b c h w', h=h,
                                            w=w).contiguous()  # reshape for quantizer
        quantize_1, code_ind_1, loss_1, usage_ratios_1  = self.quantize_1(to_quantizer_features_1)
            
        to_quantizer_features_2 = rearrange(to_quantizer_features_2, 'b (h w) c -> b c h w', h=h,
                                            w=w).contiguous()  # reshape for quantizer
        quantize_2, code_ind_2, loss_2, usage_ratios_2  = self.quantize_2(to_quantizer_features_2)
            
        to_quantizer_features_3 = rearrange(to_quantizer_features_3, 'b (h w) c -> b c h w', h=h,
                                            w=w).contiguous()  # reshape for quantizer
        quantize_3, code_ind_3, loss_3, usage_ratios_3  = self.quantize_3(to_quantizer_features_3)
            
        to_quantizer_features_4 = rearrange(to_quantizer_features_4, 'b (h w) c -> b c h w', h=h,
                                            w=w).contiguous()  # reshape for quantizer
        quantize_4, code_ind_4, loss_4, usage_ratios_4  = self.quantize_4(to_quantizer_features_4)
            
        #  Combine loss
        loss = loss_1 + loss_2 + loss_3 + loss_4

        return [quantize_1, quantize_2, quantize_3, quantize_4], [code_ind_1, code_ind_2, code_ind_3, code_ind_4], loss, [usage_ratios_1, usage_ratios_2, usage_ratios_3, usage_ratios_4]

    def decode(self, quantize, temporal_embedding_ix, spatial_embedding_ix):
        
        for i, quantize_i in enumerate(quantize):
            if i == 0:
                decoder_features_1, _ = self.decoder(quantize_i, temporal_embedding_ix=temporal_embedding_ix,
                                spatial_embedding_ix=spatial_embedding_ix, return_patch_tokens=True, branch_idx = 0)
            elif i == 1:
                decoder_features_2, _ = self.decoder(quantize_i, temporal_embedding_ix=temporal_embedding_ix,
                                spatial_embedding_ix=spatial_embedding_ix, return_patch_tokens=True, branch_idx = 1)
            elif i == 2:
                decoder_features_3, _ = self.decoder(quantize_i, temporal_embedding_ix=temporal_embedding_ix,
                                spatial_embedding_ix=spatial_embedding_ix, return_patch_tokens=True, branch_idx = 2)
            else:
                decoder_features_4, _ = self.decoder(quantize_i, temporal_embedding_ix=temporal_embedding_ix,
                                spatial_embedding_ix=spatial_embedding_ix, return_patch_tokens=True, branch_idx = 3)
        decoder_features = torch.cat([decoder_features_1, decoder_features_2, decoder_features_3, decoder_features_4], dim=2)
        
        # Reconstruct Amplitude, Sine and Cosine
        rec_amplitude = self.decode_task_layer_amplitude(decoder_features)
        rec_angle_sin = self.decode_task_layer_angle_sin(decoder_features)
        rec_angle_cos = self.decode_task_layer_angle_cos(decoder_features)

        return rec_amplitude, rec_angle_sin, rec_angle_cos

    def get_codebook_indices(self, x, temporal_embedding_ix, spatial_embedding_ix):
        return self.get_tokens(x, temporal_embedding_ix, spatial_embedding_ix)['token']
        
    def calculate_phase_loss(self, rec_sin, target_sin, rec_cos, target_cos):
        target_sin = rearrange(target_sin, 'b n a c -> b (n a) c').contiguous()
        target_cos = rearrange(target_cos, 'b n a c -> b (n a) c').contiguous()
        rec = torch.stack((rec_cos, rec_sin), dim=-1)
        target = torch.stack((target_cos, target_sin), dim=-1)
        #  Cosine Similarity for direction and Enforcing Magnitude loss
        phase_loss = 1.0 - F.cosine_similarity(rec, target, dim=-1).mean() + 0.1 * ((rec_sin**2 + rec_cos**2 - 1) ** 2).mean()
        return phase_loss

    def calculate_rec_loss(self, rec, target):
        target = rearrange(target, 'b n a c -> b (n a) c').contiguous()
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def calculate_signal_rec_loss(self, rec, target):
        target = rearrange(target, 'b n a c -> b (n a) c').contiguous()
        rec = rearrange(rec, 'b n a c -> b (n a) c').contiguous()
        mse = self.loss_fn(rec, target)
        return mse

    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt(torch.var(x, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8))
        x = (x - mean) / std
        return x, mean, std

    def forward(self, x, temporal_embedding_ix, spatial_embedding_ix):
        """
        x: shape [B, N, T]
        """
        x = rearrange(x, 'B N (A T) -> B N A T', T=self.patch_size).contiguous()
        x_fft = torch.fft.fft(x, dim=-1)

        # Get the log ampltitude
        amplitude = torch.abs(x_fft)
        amplitude = torch.log1p(amplitude)
        amplitude, amp_mean, amp_std = self.std_norm(amplitude)
        
        # Get the sine / cosine of the phase
        angle = torch.angle(x_fft)
        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)

        # Encoding and Quantize
        quantize, code_ind, code_loss, usage_ratios = self.encode(x, temporal_embedding_ix, spatial_embedding_ix)

        # Decoding
        xrec_amp, xrec_angle_sin, xrec_angle_cos = self.decode(quantize, temporal_embedding_ix, spatial_embedding_ix)
        
        # Reconstruct raw signal from amplitude and sine / cosine
        ustd_xrec = (rearrange(xrec_amp, 'B N (A T) -> B N A T', T=self.patch_size).contiguous() * amp_std) + amp_mean  # unstandardize
        ustd_xrec = torch.expm1(ustd_xrec)
        ustd_xrec = rearrange(ustd_xrec, 'b n a c -> b (n a) c').contiguous()
        xrec_signal = torch.real(inverse_fft_cos_sin(ustd_xrec, xrec_angle_sin, xrec_angle_cos))

        # Standardize sample and Reconstructed signal for MSE
        std_x, _, _ = self.std_norm(x)
        std_xrec_signal, _, _ = self.std_norm(rearrange(xrec_signal, 'B N (A T) -> B N A T', T=self.patch_size).contiguous())
        signal_rec_loss = self.calculate_signal_rec_loss(std_xrec_signal, std_x)

        # Calculate losses from decoder
        rec_amplitude_loss = self.calculate_rec_loss(xrec_amp, amplitude)
        phase_loss = self.calculate_phase_loss(xrec_angle_sin, sin_angle, xrec_angle_cos, cos_angle)

        # Total loss
        loss = code_loss + rec_amplitude_loss + phase_loss + signal_rec_loss

        std_x = std_x.view(std_x.size(0), -1, 1,std_x.size(-1)).squeeze(2)
        std_xrec_signal = std_xrec_signal.squeeze(2)

        return std_x, std_xrec_signal

