"""Router model implementation (Paper Section 3.3)"""

import torch
import torch.nn as nn
from models.embeddings import TimestepEmbedder

class SelfAttentionBlock(nn.Module):
    """Efficient self-attention block for the router"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.ffn_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attn(self.norm(x), self.norm(x), self.norm(x))
        x = x + attn_out
        
        # FFN with residual connection
        x = x + self.ffn(self.ffn_norm(x))
        return x

class RouterModel(nn.Module):
    """Implements lightweight router from Paper Section 3.3"""
    def __init__(self, config):
        super().__init__()
        # Paper's temperature schedule
        self.initial_temp = 2.0  # Initial temperature
        self.min_temp = 0.5     # Minimum temperature
        self.temp_decay = 0.99995  # Decay rate
        self.current_step = 0
        
        # Embedding layer - Changed input channels to match latent dimension
        self.embedder = nn.Conv2d(
            config.latent_channels * 16,  # Changed: Account for the 16x channel dimension 
            config.router_hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Efficient spatial attention pooling
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(config.router_hidden_size, config.router_hidden_size // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.router_hidden_size // 4, config.router_hidden_size, kernel_size=1)
        )
        
        # Timestep embedding
        self.time_embedder = TimestepEmbedder(config.router_hidden_size)
        
        # Text embedding projection - add sequence dimension handling
        self.text_embed_proj = nn.Sequential(
            nn.Linear(config.clip_embedding_dim, config.router_hidden_size),
            nn.ReLU(),
            nn.Linear(config.router_hidden_size, config.router_hidden_size)
        )
        
        # Attention blocks (simplified compared to the DiT)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                config.router_hidden_size,
                config.router_num_heads  # Use direct head count instead of num_heads//2
            )
            for _ in range(2)  # Paper recommends 2 blocks for router
        ])
        
        # Class token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.router_hidden_size))
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.router_hidden_size),
            nn.Linear(config.router_hidden_size, config.num_experts)
        )
        
        # Initialize weights with smaller values for stability
        self._init_weights()
        
    def _init_weights(self):
        # Initialize embedder
        nn.init.xavier_uniform_(self.embedder.weight, gain=0.5)
        nn.init.zeros_(self.embedder.bias)
        
        # Initialize CLS token with small random values
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize final classifier with zeros
        if hasattr(self.classifier[-1], 'weight'):
            nn.init.normal_(self.classifier[-1].weight, std=0.02)
            nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, img, timesteps, txt):
        """
        Forward pass with temperature annealing (Section 3.3)
        
        Args:
            img: Input image tensor [B, C, H, W] or [B, C, D, H, W] for latents
            timesteps: Timestep values [B]
            txt: Text embeddings [B, L, D]
        """
        batch_size = img.shape[0]
        
        # Reshape 5D latents to 4D for conv2d
        if img.dim() == 5:
            B, C, D, H, W = img.shape
            print(f"[Router] Input img shape: {img.shape}")
            img = img.reshape(B, C * D, H, W)
            print(f"[Router] Reshaped img shape: {img.shape}")
        else:
            print(f"[Router] Input img shape: {img.shape}")
        
        # Patch embedding
        x = self.embedder(img)  # [B, D, H', W']
        print(f"[Router] After embedder shape: {x.shape}")
        
        # Spatial attention processing
        attn_features = self.spatial_attention(x)
        print(f"[Router] After spatial_attention shape: {attn_features.shape}")
        x = attn_features.mean(dim=(2, 3))  # [B, D]
        print(f"[Router] After spatial pooling shape: {x.shape}")
        
        # Timestep embedding
        t_emb = self.time_embedder(timesteps.float())  # [B, D]
        print(f"[Router] Timestep embedding shape: {t_emb.shape}")
        x = x + t_emb
        print(f"[Router] After adding timestep shape: {x.shape}")

        # Text embedding integration - Project text and reduce sequence dimension first
        print(f"[Router] Input text shape: {txt.shape}")
        text_emb = self.text_embed_proj(txt)  # Either [B, L, D] or [B, D]
        print(f"[Router] Projected text shape: {text_emb.shape}")

        # If the projected text embeddings are 3D, average over the token dimension
        # Handle both possible dimension arrangements
        if text_emb.dim() == 3:
            if text_emb.shape[1] == self.config.clip_embedding_dim:
                # Handle [B, D, L] format
                print(f"[Router] Text in [B, D, L] format")
                text_emb = text_emb.mean(dim=2)
            else:
                # Handle [B, L, D] format
                print(f"[Router] Text in [B, L, D] format")
                text_emb = text_emb.mean(dim=1)
            print(f"[Router] Text after pooling shape: {text_emb.shape}")

        # Verify dimensions match before addition
        assert x.shape == text_emb.shape, f"Shape mismatch: x={x.shape}, text_emb={text_emb.shape}"
        x = x + text_emb
        print(f"[Router] After adding text shape: {x.shape}")

        # Prepare for transformer - ensure correct dimensions
        x = x.unsqueeze(1)  # [B, 1, D]
        print(f"[Router] After unsqueeze shape: {x.shape}")
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        print(f"[Router] CLS token shape: {cls_tokens.shape}")
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 2, D]
        print(f"[Router] After adding CLS token shape: {x.shape}")

        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            print(f"[Router] After block {i} shape: {x.shape}")
            
        # Final processing
        cls_output = x[:, 0]
        print(f"[Router] CLS output shape: {cls_output.shape}")
        logits = self.classifier(cls_output)
        print(f"[Router] Logits shape: {logits.shape}")
        
        # Apply temperature scaling with decay
        temperature = max(
            self.min_temp,
            self.initial_temp * (self.temp_decay ** self.current_step)
        )
        print(f"[Router] Temperature value: {temperature}")
        self.current_step += 1
        
        return logits / temperature 