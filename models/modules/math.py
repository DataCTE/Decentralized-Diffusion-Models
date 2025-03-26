import torch
from torch import Tensor
from einops import rearrange


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    """Attention with rotary position embeddings"""
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """Rotary Positional Embeddings implementation"""
    assert dim % 2 == 0
    # Ensure pos has at least one dimension (prevents 0-D errors)
    if pos.dim() == 0:
        pos = pos.unsqueeze(0)
    # If pos is 1-D (e.g. of shape (B,)), add a sequence dimension 
    if pos.dim() == 1:
        pos = pos.unsqueeze(1)
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to queries and keys"""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
