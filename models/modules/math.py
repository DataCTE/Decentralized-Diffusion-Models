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
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    # Reshape xq/xk to [..., seq_len, head_dim]
    xq = xq.transpose(1, 2)  # [B, seq_len, num_heads, head_dim]
    xk = xk.transpose(1, 2)
    
    # Reshape freqs_cis to match xq dimensions
    freqs_cis = freqs_cis.view(freqs_cis.shape[0], xq.shape[1], -1)
    
    # Apply RoPE to first half of dimensions
    head_dim = xq.shape[-1]
    half_dim = head_dim // 2
    freq_shape = (1, 1, half_dim)
    
    sin, cos = torch.sin(freqs_cis), torch.cos(freqs_cis)
    xq_rot = xq[..., :half_dim] * cos + xq[..., half_dim:half_dim*2] * sin
    xk_rot = xk[..., :half_dim] * cos + xk[..., half_dim:half_dim*2] * sin
    
    # Concatenate rotated and unrotated parts
    xq_out = torch.cat([xq_rot, xq[..., half_dim*2:]], dim=-1)
    xk_out = torch.cat([xk_rot, xk[..., half_dim*2:]], dim=-1)
    
    return xq_out.transpose(1, 2), xk_out.transpose(1, 2)