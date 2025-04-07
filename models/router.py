import torch
import torch.nn as nn
import math

# Import necessary components from flux modules
from .flux.modules.layers import timestep_embedding, MLPEmbedder

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=32, patch_size=2, in_chans=4, embed_dim=768):
        super().__init__()
        self.img_size = img_size # Expected input spatial resolution
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # Ensure input dimensions match expected configuration
        if H != self.img_size or W != self.img_size:
            # Option 1: Raise error if size mismatch is critical
            raise ValueError(f"Input image size ({H}x{W}) doesn't match model's expected size ({self.img_size}x{self.img_size}).")
            # Option 2: Log a warning if flexible sizes are sometimes okay (requires careful pos embed handling)
            # print(f"Warning: Input image size ({H}x{W}) doesn't match model's expected size ({self.img_size}x{self.img_size}). Positional embeddings might be inaccurate.")
            # Option 3: Implement resizing (might affect performance/quality)
            # x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            
        if C != self.in_chans:
             raise ValueError(f"Input image channels ({C}) doesn't match model's expected channels ({self.in_chans}).")

        x = self.proj(x).flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        return x

class TimestepEmbedderCombined(nn.Module):
    """ Embeds scalar timesteps into vector representations using flux modules. """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = MLPEmbedder(frequency_embedding_size, hidden_size)
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        # Use the timestep_embedding function from flux.modules.layers
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class BasicTransformerBlock(nn.Module):
    """ Standard Transformer Block with AdaLN-Zero modulation """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # Ensure attention dropout is set if dropout_rate > 0
        self.attn = nn.MultiheadAttention(dim, num_heads, qkv_bias=qkv_bias, batch_first=True, dropout=dropout_rate) 
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # Using GELU with approximate="tanh" for potential compatibility if flux uses it.
        # Otherwise, a standard nn.GELU() is fine.
        self.mlp = nn.Sequential(
             nn.Linear(dim, mlp_hidden_dim, bias=True), # Bias=True is standard
             nn.GELU(approximate="tanh"),
             nn.Dropout(dropout_rate),
             nn.Linear(mlp_hidden_dim, dim, bias=True), # Bias=True is standard
             nn.Dropout(dropout_rate)
        )
        # adaLN-Zero modulation: predicts scale, shift, and gate values
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def _modulate(self, x, shift, scale):
        # Add eps for numerical stability? Usually not needed for LayerNorm outputs.
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, c):
        # c is the conditioning signal (e.g., time + optional condition embeddings)
        # Predict modulation parameters (scale, shift, gate for attn and mlp)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # Attention block with residual connection
        residual = x
        x_norm = self.norm1(x)
        x_mod = self._modulate(x_norm, shift_msa, scale_msa)
        # Ensure query, key, value are derived correctly for self-attention
        attn_output, _ = self.attn(query=x_mod, key=x_mod, value=x_mod, need_weights=False)
        x = residual + gate_msa.unsqueeze(1) * attn_output

        # MLP block with residual connection
        residual = x
        x_norm = self.norm2(x)
        x_mod = self._modulate(x_norm, shift_mlp, scale_mlp)
        mlp_output = self.mlp(x_mod)
        x = residual + gate_mlp.unsqueeze(1) * mlp_output

        return x

class RouterModel(nn.Module):
    """
    Predicts the probability that a noisy sample xt belongs to each data cluster.
    Uses a DiT-like architecture with a CLS token and a classification head,
    conditioned on time and optionally other embeddings (e.g., CLIP pooler).
    """
    def __init__(self,
                 num_clusters: int,
                 input_size: int = 32, # Spatial resolution of latent input (H and W)
                 patch_size: int = 2,
                 in_channels: int = 4, # Input channels (e.g., latent channels from VAE)
                 hidden_size: int = 768, # Dimension of the transformer
                 depth: int = 12,        # Number of transformer blocks
                 num_heads: int = 12,      # Number of attention heads
                 mlp_ratio: float = 4.0,   # Ratio for MLP hidden dimension
                 cond_dim: int = None, # Dimension of optional condition vector 'y' (e.g., CLIP pooler output)
                 dropout_rate: float = 0.0, # Dropout rate for transformer blocks
                 # learn_sigma is part of DiT but not used for router's classification goal
                 learn_sigma: bool = False): 
        super().__init__()
        if not isinstance(num_clusters, int) or num_clusters <= 0:
             raise ValueError("num_clusters must be a positive integer")
        self.num_clusters = num_clusters
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.has_cond = self.cond_dim is not None

        # Input embedding layers
        self.patch_embed = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedderCombined(hidden_size) # Uses flux timestep_embedding + MLPEmbedder

        self.num_patches = self.patch_embed.num_patches

        # Positional and classification tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_size), requires_grad=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)

        # Optional Conditioning Embedder (projects input condition 'y' to hidden_size)
        if self.has_cond:
            self.y_embedder = nn.Linear(self.cond_dim, hidden_size, bias=True)
        else:
            self.y_embedder = None # No conditioning embedder needed

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            BasicTransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, dropout_rate=dropout_rate)
            for _ in range(depth)
        ])

        # Output head
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.router_head = nn.Linear(hidden_size, num_clusters, bias=True)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.LayerNorm layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            # LayerNorm weights/biases are usually left at default (1/0) or initialized conditionally
            # elif isinstance(module, nn.LayerNorm):
            #     nn.init.constant_(module.bias, 0)
            #     nn.init.constant_(module.weight, 1.0)

        self.apply(_basic_init)

        # Initialize patch_embed projection like nn.Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # Ensure patch_embed bias is initialized if it exists (it's not standard in original DiT PatchEmbed)
        if hasattr(self.patch_embed.proj, 'bias') and self.patch_embed.proj.bias is not None:
             nn.init.constant_(self.patch_embed.proj.bias, 0)


        # Initialize positional embedding and CLS token from a normal distribution
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

        # Initialize timestep embedder MLP layers (MLPEmbedder has in_layer, out_layer)
        nn.init.normal_(self.t_embedder.mlp.in_layer.weight, std=0.02)
        nn.init.constant_(self.t_embedder.mlp.in_layer.bias, 0)
        nn.init.normal_(self.t_embedder.mlp.out_layer.weight, std=0.02)
        nn.init.constant_(self.t_embedder.mlp.out_layer.bias, 0)

        # Initialize condition embedder linear layer if it exists
        if self.has_cond and self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.weight, std=0.02)
            nn.init.constant_(self.y_embedder.bias, 0)

        # Zero-out the final linear layer of adaLN modulation in each block
        for block in self.blocks:
             nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
             nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out the final classification head layer
        nn.init.constant_(self.router_head.weight, 0)
        nn.init.constant_(self.router_head.bias, 0)


    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None):
        """
        Forward pass for the router model.

        Args:
            x (torch.Tensor): Input noisy tensor (e.g., latent image) [B, C, H, W].
                               Shape should match C=in_channels, H=W=input_size.
            t (torch.Tensor): Timestep tensor [B]. Values should be appropriate for
                               timestep_embedding (e.g., integers 0-999 or floats 0-1).
                               Scaling might be needed depending on training setup.
            y (torch.Tensor, optional): Conditioning tensor [B, cond_dim]. Defaults to None.

        Returns:
            torch.Tensor: Logits for each cluster [B, num_clusters].
        """
        B = x.shape[0]
        
        # 1. Patch and Position Embedding
        x_patches = self.patch_embed(x)  # (B, num_patches, hidden_size)
        
        if x_patches.shape[1] != self.num_patches:
             # This can happen if PatchEmbed handles dynamic sizes, but pos_embed is fixed
             raise ValueError(f"Number of patches ({x_patches.shape[1]}) does not match expected ({self.num_patches}) based on input_size and patch_size.")

        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_size)
        x = torch.cat((cls_token, x_patches), dim=1) # (B, 1 + num_patches, hidden_size)

        # Add positional embedding
        if x.shape[1] != self.pos_embed.shape[1]:
             # This should not happen if patch_embed checks size and num_patches is correct
             raise ValueError(f"Input sequence length ({x.shape[1]}) does not match positional embedding length ({self.pos_embed.shape[1]}). Check input_size, patch_size.")
        x = x + self.pos_embed # (B, 1 + num_patches, hidden_size)

        # 2. Time and Condition Embedding
        # Ensure 't' has the correct dtype (often float for timestep_embedding)
        t_embed = self.t_embedder(t.float())  # (B, hidden_size)
        cond_embed = t_embed

        # Add condition embedding if provided and expected
        if self.has_cond:
            if y is None:
                raise ValueError("Model configured for conditioning (cond_dim provided), but 'y' is None.")
            if self.y_embedder is None: # Should not happen if has_cond is True
                 raise RuntimeError("Internal error: has_cond is True but y_embedder is None.")

            if y.ndim != 2 or y.shape[0] != B or y.shape[1] != self.cond_dim:
                 raise ValueError(f"Condition tensor y has shape {y.shape}, expected [{B}, {self.cond_dim}]")

            y_embed = self.y_embedder(y) # (B, hidden_size)
            cond_embed = t_embed + y_embed # Combine embeddings by addition
        elif y is not None:
            # Warn if 'y' is given but not used
            print("Warning: RouterModel received condition 'y' but is not configured to use it (cond_dim is None).")

        # 3. Transformer Blocks
        for block in self.blocks:
             x = block(x, c=cond_embed)

        # 4. Final Layer Norm and Head
        # Apply final norm only to the CLS token embedding
        cls_output = self.norm_final(x[:, 0]) # (B, hidden_size)

        # Project CLS token output to cluster logits
        logits = self.router_head(cls_output) # (B, num_clusters)

        return logits
