"""DiT implementation for Decentralized Diffusion Models (Paper Section 3.2)"""

from __future__ import annotations
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers.attend import Attend
from x_transformers import (
    RMSNorm,
    FeedForward
)

from hyper_connections import (
    HyperConnections,
    Residual
)

from models.embeddings import TimestepEmbedder

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value):
    return (t / value).tanh() * value

def modulate(x, shift, scale):
    """Applies modulation to the input"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# rmsnorm

class MultiHeadRMSNorm(Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# attention

class JointAttention(Module):
    def __init__(
        self,
        *,
        dim_inputs: tuple[int, ...],
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash = False,
        softclamp = False,
        softclamp_value = 50.,
        attend_kwargs: dict = dict()
    ):
        super().__init__()
        """
        ein notation

        b - batch
        h - heads
        n - sequence
        d - feature dimension
        """

        dim_inner = dim_head * heads

        num_inputs = len(dim_inputs)
        self.num_inputs = num_inputs

        self.to_qkv = ModuleList([nn.Linear(dim_input, dim_inner * 3, bias = False) for dim_input in dim_inputs])

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)

        self.attend = Attend(
            flash = flash,
            softclamp_logits = softclamp,
            logit_softclamp_value = softclamp_value,
            **attend_kwargs
        )

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = ModuleList([nn.Linear(dim_inner, dim_input, bias = False) for dim_input in dim_inputs])

        self.qk_rmsnorm = qk_rmsnorm
        self.q_rmsnorms = ModuleList([])
        self.k_rmsnorms = ModuleList([])

        if qk_rmsnorm:
            self.q_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])
            self.k_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    def forward(
        self,
        inputs: tuple[Tensor],
        masks: tuple[Tensor | None] | None = None
    ):

        device = self.dummy.device

        assert len(inputs) == self.num_inputs

        masks = default(masks, (None,) * self.num_inputs)

        # project each modality separately for qkv
        # also handle masks, assume None means attend to all tokens

        all_qkvs = []
        all_masks = []

        for x, mask, to_qkv, q_rmsnorm, k_rmsnorm in zip(inputs, masks, self.to_qkv, self.q_rmsnorms, self.k_rmsnorms):

            qkv = to_qkv(x)
            qkv = self.split_heads(qkv)

            # optional qk rmsnorm per modality

            if self.qk_rmsnorm:
                q, k, v = qkv
                q = q_rmsnorm(q)
                k = k_rmsnorm(k)
                qkv = torch.stack((q, k, v))

            all_qkvs.append(qkv)

            # handle mask per modality

            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = device, dtype = torch.bool)

            all_masks.append(mask)

        # combine all qkv and masks

        all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
        all_masks, _ = pack(all_masks, 'b *')

        # attention

        q, k, v = all_qkvs

        outs, *_ = self.attend(q, k, v, mask = all_masks)

        # merge heads and then separate by modality for combine heads projection

        outs = self.merge_heads(outs)
        outs = unpack(outs, packed_shape, 'b * d')

        # separate combination of heads for each modality

        all_outs = []

        for out, to_out in zip(outs, self.to_out):
            out = to_out(out)
            all_outs.append(out)

        return tuple(all_outs)

class MMDiTBlock(nn.Module):
    def __init__(
        self,
        *,
        dim_text,
        dim_image,
        dim_cond = None,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash_attn = False,
        num_residual_streams = 1,
        ff_kwargs: dict = dict()
    ):
        super().__init__()

        residual_klass = Residual if num_residual_streams == 1 else HyperConnections

        self.text_attn_residual_fn = residual_klass(num_residual_streams, dim = dim_text)
        self.text_ff_residual_fn = residual_klass(num_residual_streams, dim = dim_text)

        self.image_attn_residual_fn = residual_klass(num_residual_streams, dim = dim_image)
        self.image_ff_residual_fn = residual_klass(num_residual_streams, dim = dim_image)

        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        if has_cond:
            dim_gammas = (
                *((dim_text,) * 4),
                *((dim_image,) * 4)
            )

            dim_betas = (
                *((dim_text,) * 2),
                *((dim_image,) * 2),
            )

            self.cond_dims = (*dim_gammas, *dim_betas)

            to_cond_linear = nn.Linear(dim_cond, sum(self.cond_dims))

            self.to_cond = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                to_cond_linear
            )

            nn.init.zeros_(to_cond_linear.weight)
            nn.init.zeros_(to_cond_linear.bias)
            nn.init.constant_(to_cond_linear.bias[:sum(dim_gammas)], 1.)

        self.text_attn_layernorm = nn.LayerNorm(dim_text, elementwise_affine = not has_cond)
        self.image_attn_layernorm = nn.LayerNorm(dim_image, elementwise_affine = not has_cond)

        self.text_ff_layernorm = nn.LayerNorm(dim_text, elementwise_affine = not has_cond)
        self.image_ff_layernorm = nn.LayerNorm(dim_image, elementwise_affine = not has_cond)

        self.joint_attn = JointAttention(
            dim_inputs = (dim_text, dim_image),
            dim_head = dim_head,
            heads = heads,
            flash = flash_attn,
            qk_rmsnorm = qk_rmsnorm
        )

        self.text_ff = FeedForward(dim_text, **ff_kwargs)
        self.image_ff = FeedForward(dim_image, **ff_kwargs)

    def forward(
        self,
        *,
        text_tokens,
        image_tokens,
        text_mask = None,
        time_cond = None,
        skip_feedforward_text_tokens = True
    ):
        assert not (exists(time_cond) ^ self.has_cond), 'time condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        if self.has_cond:
            (
                text_pre_attn_gamma,
                text_post_attn_gamma,
                text_pre_ff_gamma,
                text_post_ff_gamma,
                image_pre_attn_gamma,
                image_post_attn_gamma,
                image_pre_ff_gamma,
                image_post_ff_gamma,
                text_pre_attn_beta,
                text_pre_ff_beta,
                image_pre_attn_beta,
                image_pre_ff_beta,
            ) = self.to_cond(time_cond).split(self.cond_dims, dim = -1)

        # attention - text branch

        text_tokens, add_text_residual = self.text_attn_residual_fn(text_tokens)
        text_tokens = self.text_attn_layernorm(text_tokens)

        if self.has_cond:
            text_tokens = text_tokens * text_pre_attn_gamma + text_pre_attn_beta

        # attention - image branch

        image_tokens, add_image_residual = self.image_attn_residual_fn(image_tokens)
        image_tokens = self.image_attn_layernorm(image_tokens)

        if self.has_cond:
            image_tokens = image_tokens * image_pre_attn_gamma + image_pre_attn_beta

        # joint attention

        text_tokens, image_tokens = self.joint_attn(
            inputs = (text_tokens, image_tokens),
            masks = (text_mask, None)
        )

        if self.has_cond:
            text_tokens = text_tokens * text_post_attn_gamma
            image_tokens = image_tokens * image_post_attn_gamma

        text_tokens = add_text_residual(text_tokens)
        image_tokens = add_image_residual(image_tokens)

        if skip_feedforward_text_tokens:
            return text_tokens, image_tokens

        # feedforward - text branch

        text_tokens, add_text_residual = self.text_ff_residual_fn(text_tokens)
        text_tokens = self.text_ff_layernorm(text_tokens)

        if self.has_cond:
            text_tokens = text_tokens * text_pre_ff_gamma + text_pre_ff_beta

        text_tokens = self.text_ff(text_tokens)

        if self.has_cond:
            text_tokens = text_tokens * text_post_ff_gamma

        text_tokens = add_text_residual(text_tokens)

        # feedforward - image branch

        image_tokens, add_image_residual = self.image_ff_residual_fn(image_tokens)
        image_tokens = self.image_ff_layernorm(image_tokens)

        if self.has_cond:
            image_tokens = image_tokens * image_pre_ff_gamma + image_pre_ff_beta

        image_tokens = self.image_ff(image_tokens)

        if self.has_cond:
            image_tokens = image_tokens * image_post_ff_gamma

        image_tokens = add_image_residual(image_tokens)

        return text_tokens, image_tokens

class FinalLayer(nn.Module):
    """Final layer with adaLN modulation"""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # Ensure we always output exactly out_channels
        self.linear = nn.Linear(hidden_size, patch_size**2 * out_channels)
        self.patch_size = patch_size
        self.out_channels = out_channels
        
        # Initialize to zero
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        # Get AdaLN modulation parameters
        hidden_size = x.shape[-1]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        
        # Apply modulation
        x = modulate(self.norm_final(x), shift, scale)
        
        # Project to output and ensure output dimensions are consistent
        x = self.linear(x)  # [B, N, P*P*C]
        
        # Optional: Verify output shape consistency
        batch_size, n_tokens, features = x.shape
        expected_features = self.patch_size**2 * self.out_channels
        if features != expected_features:
            print(f"Warning: Features dimension {features} doesn't match expected {expected_features}")
            # Adjust by padding or truncating if needed
            if features < expected_features:
                pad_size = expected_features - features
                padding = torch.zeros((batch_size, n_tokens, pad_size), device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=2)
            else:
                x = x[:, :, :expected_features]
                
        return x

class ExpertMMDiT(nn.Module):
    """Implements expert model using MMDiT blocks"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim # Assuming hidden_dim still refers to image embedding dimension
        self.patch_size = config.patch_size
        self.in_channels = config.latent_channels  # VAE latent channels (16)
        self.out_channels = config.latent_channels  # Predict latent noise
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.ffn_dim = config.ffn_dim
        self.clip_embedding_dim = 768 # Assuming CLIP embedding dim is still 768
        self.router_hidden_size = config.router_hidden_size # Assuming router_hidden_size is still relevant for conditioning

        # Embeddings
        self.x_embedder = nn.Conv2d(
            self.in_channels,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        self.t_embedder = TimestepEmbedder(self.hidden_dim) # Use hidden_dim for time embedding dim

        # Text projection - assuming text projection is still needed to project CLIP embeddings
        self.text_projection = nn.Linear(self.clip_embedding_dim, self.router_hidden_size) # Project CLIP embeddings to router hidden dim for conditioning

        # MMDiT blocks - using MMDiTBlock (renamed DiTBlock)
        self.blocks = ModuleList([
            MMDiTBlock( # Changed from MMDiTBlockInternal to MMDiTBlock
                dim_image = self.hidden_dim, # Image embedding dimension
                dim_text = self.router_hidden_size, # Corrected: Parameter name to dim_text
                dim_cond = self.hidden_dim, # Time conditioning dimension (using hidden_dim)
                dim_head = self.config.num_heads, # Assuming num_heads is still relevant
                heads = self.config.num_heads, # Assuming heads is still relevant
                qk_rmsnorm = config.qk_rmsnorm, # Assuming qk_rmsnorm is in config
                ff_kwargs=dict(mult=self.config.ffn_dim/config.hidden_dim) # Assuming ffn_dim ratio is still relevant
            )
            for _ in range(self.num_layers)
        ])

        # Final layer - remains the same
        self.final_layer = FinalLayer(self.hidden_dim, self.patch_size, self.out_channels)

        # Initialize weights - remains mostly the same
        self.initialize_weights()

        # Enable gradient checkpointing - remains the same
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

    def initialize_weights(self):
        # Initialize patch embedding - remains the same
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.zeros_(self.x_embedder.bias)

        # Initialize timestep embedding MLP - using router_hidden_size now
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize text projection - remains the same
        nn.init.normal_(self.text_projection.weight, std=0.02)
        nn.init.zeros_(self.text_projection.bias)

        # Note: Block adaLN modulation layers are already initialized in MMDiTBlock

    def get_position_embeddings(self, h, w, device):
        """Generate position embeddings for arbitrary grid sizes"""
        grid_indices = torch.stack(torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        ), dim=-1).float()  # [H, W, 2]
        
        # Calculate embeddings for each dimension
        omega = torch.arange(self.hidden_dim // 4, device=device) / (self.hidden_dim // 4 - 1)
        omega = 1. / (10000 ** omega)
        
        # Calculate embeddings
        y_embed = grid_indices[..., 0:1] * omega
        x_embed = grid_indices[..., 1:2] * omega
        
        # Combine embeddings and flatten
        pos_embed = torch.cat([
            torch.sin(y_embed), torch.cos(y_embed),
            torch.sin(x_embed), torch.cos(x_embed)
        ], dim=-1).reshape(h * w, self.hidden_dim).unsqueeze(0)  # [1, H*W, D]
        
        return pos_embed
            
    def unpatchify(self, x, h, w):
        """Reshape patches back to image with explicit h/w dimensions"""
        batch_size = x.shape[0]
        
        # Reshape to [B, H, W, patch_size, patch_size, C]
        x = x.reshape(batch_size, h, w, self.patch_size, self.patch_size, self.out_channels)
        
        # Permute and reshape to image
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(batch_size, self.out_channels, 
                     h * self.patch_size, w * self.patch_size)
        return x
    
    def forward(self, x, t, text_embeds):
        cond_vector = self.t_embedder(t)

        # Patch embedding - remains the same
        x = self.x_embedder(x)  # [B, D, H', W']
        batch_size, hidden_size, h, w = x.shape
        x = x.reshape(batch_size, hidden_size, h * w).transpose(1, 2) # [B, N, D] - Reshape to [B, N, D]

        text_embeds = self.text_projection(text_embeds) # Project text embeddings - adjust if needed

        # MMDiT blocks - forward pass through MMDiT blocks
        for block in self.blocks:
            text_embeds, x = block( # MMDiTBlock now expects and returns text_embeds and image_tokens (x)
                image_tokens = x,
                text_tokens = text_embeds,
                time_cond = cond_vector, # Pass timestep conditioning
                text_mask = None # Assuming no text mask needed for now
            )

        x = self.final_layer(x, cond_vector) # Final layer still takes image tokens (x) and cond_vector
        x = self.unpatchify(x, h, w) # Unpatchify to get back to image shape
        return x

    def debug_tensor_shapes(self, prefix="", **tensors):
        """Debug tensor shapes during training"""
        if not self.training or torch.rand(1).item() > 0.01:  # Only log occasionally during training
            return
        
        rank = 0
        if torch.distributed.is_initialized():  # Check initialization first
            rank = torch.distributed.get_rank()
        
        lines = [f"[Rank {rank}] {prefix} Tensor Shapes:"]
        for name, tensor in tensors.items():
            if tensor is None:
                lines.append(f"  - {name}: None")
            else:
                lines.append(f"  - {name}: {tensor.shape}")
        
        message = "\n".join(lines)
        print(message)
        
        return message 