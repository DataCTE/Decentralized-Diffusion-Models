"""DiT implementation for Decentralized Diffusion Models (Paper Section 3.2)"""

import torch
import torch.nn as nn


from models.embeddings import TimestepEmbedder

def modulate(x, shift, scale):
    """Applies modulation to the input"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """Implements transformer block with adaLN-zero conditioning (Paper Eq. 4)"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        # More robust handling of conditioning vector dimensions
        if not isinstance(c, torch.Tensor):
            raise TypeError(f"Expected c to be a tensor, got {type(c)}")
        
        # Add debug information with rank and stage - FIXED ORDER OF CHECKS
        debug_enabled = (self.training and 
                        torch.distributed.is_initialized() and  # Check initialization first!
                        torch.distributed.get_rank() == 0 and 
                        torch.rand(1).item() < 0.01)
        
        if debug_enabled:
            rank = torch.distributed.get_rank()
            # print(f"[Rank {rank}] DiTBlock input tensor c shape: {c.shape}, dim: {c.dim()}, device: {c.device}")  # Disabled for production
        
        # Ensure c has at least 2 dimensions (batch_size, features)
        if c.dim() == 0:  # It's a scalar tensor
            c = c.unsqueeze(0).unsqueeze(0)  # Add batch and feature dimensions
            if debug_enabled:
                print(f"[Rank {rank}] DiTBlock expanded scalar tensor to shape: {c.shape}")
        elif c.dim() == 1:  # It's a 1D tensor (features only)
            c = c.unsqueeze(0)  # Add batch dimension
            if debug_enabled:
                print(f"[Rank {rank}] DiTBlock expanded 1D tensor to shape: {c.shape}")
        
        # Disabled for production runs
        #if self.training and torch.distributed.is_initialized() and torch.rand(1).item() < 0.001:
        #    if torch.distributed.get_rank() == 0:
                # print(f"Conditioning tensor shape: {c.shape}")  # Disabled per config
                
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Modulated attention
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            modulate(self.norm1(x), shift_msa, scale_msa),
            modulate(self.norm1(x), shift_msa, scale_msa)
        )[0]
        
        # Modulated MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x

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

class ExpertDiT(nn.Module):
    """Implements expert model from Paper Section 3.2"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_dim
        self.patch_size = config.patch_size
        self.in_channels = config.latent_channels  # VAE latent channels (16)
        self.out_channels = config.latent_channels  # Predict latent noise
        
        # Embeddings
        self.x_embedder = nn.Conv2d(
            self.in_channels, 
            self.hidden_size, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        
        # Position embedding
        self.pos_embed = None  # Will be generated dynamically
        
        # CLIP text conditioning components
        self.text_projection = nn.Linear(768, self.hidden_size)  # Project CLIP embeddings to DiT hidden dim
        self.text_cross_attention = nn.MultiheadAttention(
            self.hidden_size, 
            config.num_heads, 
            batch_first=True
        )
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, config.num_heads, config.ffn_dim/config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.out_channels)
        
        # Initialize weights
        self.initialize_weights()
        
        # Enable gradient checkpointing for memory efficiency from config
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        
    def initialize_weights(self):
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.zeros_(self.x_embedder.bias)
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Initialize text projection
        nn.init.normal_(self.text_projection.weight, std=0.02)
        nn.init.zeros_(self.text_projection.bias)
        
        # Note: Block adaLN modulation layers are already initialized in their constructor
            
    def get_position_embeddings(self, h, w, device):
        """Generate position embeddings for arbitrary grid sizes"""
        grid_indices = torch.stack(torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        ), dim=-1).float()  # [H, W, 2]
        
        # Calculate embeddings for each dimension
        omega = torch.arange(self.hidden_size // 4, device=device) / (self.hidden_size // 4 - 1)
        omega = 1. / (10000 ** omega)
        
        # Calculate embeddings
        y_embed = grid_indices[..., 0:1] * omega
        x_embed = grid_indices[..., 1:2] * omega
        
        # Combine embeddings and flatten
        pos_embed = torch.cat([
            torch.sin(y_embed), torch.cos(y_embed),
            torch.sin(x_embed), torch.cos(x_embed)
        ], dim=-1).reshape(h * w, self.hidden_size).unsqueeze(0)  # [1, H*W, D]
        
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
    
    def forward(self, x, t, text_embeds=None):
        # Remove text cross-attention
        cond_vector = self.t_embedder(t)
        
        # Original DiT processing
        for block in self.blocks:
            x = block(x, cond_vector)
            
        return self.final_layer(x, cond_vector)

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