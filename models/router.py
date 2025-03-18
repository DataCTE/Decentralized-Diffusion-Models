"""Router model implementation (Paper Section 3.3)"""

import torch
import torch.nn as nn


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
    """Implements lightweight router from Paper Appendix A.3"""
    def __init__(self, config):
        super().__init__()
        # Embedding layer
        self.embedder = nn.Conv2d(
            config.latent_channels, 
            config.router_hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Efficient spatial attention pooling
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(config.router_hidden_size, config.router_hidden_size // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.router_hidden_size // 4, 1, kernel_size=1),
            nn.Softmax(dim=(2, 3))  # Spatial softmax
        )
        
        # Timestep embedding
        self.time_embedder = nn.Sequential(
            nn.Linear(1, config.router_hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.router_hidden_size // 2, config.router_hidden_size)
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
        
        # Learnable temperature for calibration
        # Initialize with a reasonable value
        self.register_buffer('temperature', torch.ones(1) * 1.0)
        
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

    def forward(self, x, t, text_embeddings=None):
        """
        Args:
            x: Input tensor [B, C, H, W]
            t: Timestep tensor [B,]
            text_embeddings: Optional text embeddings for conditional generation
        Returns:
            logits: Expert logits [B, num_experts]
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.embedder(x)  # [B, D, H', W']
        
        # Spatial attention pooling
        attn_weights = self.spatial_attention(x)  # [B, 1, H', W']
        x = (x * attn_weights).sum(dim=(2, 3))  # [B, D]
        
        # Timestep embedding
        t_emb = self.time_embedder(t.unsqueeze(-1))  # [B, D]
        
        # Add timestep information
        x = x + t_emb  # [B, D]
        
        # Expand to sequence for transformer blocks
        x = x.unsqueeze(1)  # [B, 1, D]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 2, D]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Get CLS token output only
        cls_output = x[:, 0]  # [B, D]
        
        # Apply classifier to get logits
        logits = self.classifier(cls_output)  # [B, num_experts]
        
        # Apply temperature scaling for better calibration
        # Lower temperature gives sharper distribution
        return logits / self.temperature
    
    def set_temperature(self, temp_value):
        """Set the temperature value for the model"""
        self.temperature.fill_(temp_value)
    
    def get_temperature(self):
        """Get the current temperature value"""
        return self.temperature.item() 