import torch
from torch import Tensor
from einops import rearrange

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    device = pos.device
    dtype = pos.dtype
    
    # Compute frequencies
    scale = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim
    omega = 1.0 / (theta ** scale)
    
    # Compute positions * omega
    pos_omega = pos.unsqueeze(-1) * omega  # (..., n, d/2)
    
    # Compute sin and cos components
    sin = torch.sin(pos_omega)
    cos = torch.cos(pos_omega)
    
    # Interleave sin and cos to form the rotary encoding
    rotary_enc = torch.stack([cos, sin], dim=-1)
    rotary_enc = rotary_enc.reshape(*pos.shape, dim)  # (..., n, d)
    
    return rotary_enc


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    # Reshape xq/xk to [..., seq_len, head_dim]
    xq = xq.transpose(1, 2)  # [B, seq_len, num_heads, head_dim]
    xk = xk.transpose(1, 2)
    
    # Remove dummy dimension and reshape freqs_cis
    freqs_cis = freqs_cis.squeeze(1)  # [B, seq_len, head_dim]
    
    # Apply RoPE to first half of dimensions
    head_dim = xq.shape[-1]
    xq_rot = xq * freqs_cis[..., :head_dim//2] + \
             torch.roll(xq, shifts=head_dim//2, dims=-1) * freqs_cis[..., head_dim//2:]
    xk_rot = xk * freqs_cis[..., :head_dim//2] + \
             torch.roll(xk, shifts=head_dim//2, dims=-1) * freqs_cis[..., head_dim//2:]
    
    return xq_rot.transpose(1, 2), xk_rot.transpose(1, 2)