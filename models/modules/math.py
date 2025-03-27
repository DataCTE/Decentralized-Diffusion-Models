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
    # Reshape xq/xk to [batch, num_heads, seq_len, head_dim]
    xq = xq.permute(0, 2, 1, 3)  # [B, nh, T, hs]
    xk = xk.permute(0, 2, 1, 3)
    
    # Reshape freqs_cis to match dimensions
    freqs_cis = freqs_cis.view(xq.size(0), xq.size(2), 1, -1)  # [B, T, 1, D]
    
    # Split dimensions for complex-valued computation
    xq_float = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_float = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # Apply RoPE using einsum for efficiency
    xq_out = torch.einsum('...td,...d->...td', xq_float, freqs_cis)
    xk_out = torch.einsum('...td,...d->...td', xk_float, freqs_cis)
    
    # Recombine and cast back to original dtype
    return (xq_out.flatten(3).type_as(xq),
            xk_out.flatten(3).type_as(xk))