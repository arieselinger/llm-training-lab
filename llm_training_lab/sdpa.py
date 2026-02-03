"""
Implement scaled dot-product attention similiar to torch.nn.functional.scaled_dot_product_attention
without flash-attention or specific cuda kernels
"""

import math

import torch


def _scaled_dot_product_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  *,
  attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """

  Args:
    query: shape (B, H, Tq, Dh)
    key: shape (B, H, Tk, Dh)
    value: shape (B, H, Tk, Dh)
  """
  Dh = query.size(-1)

  scale = 1.0 / math.sqrt(Dh)
  scores = query @ key.transpose(-1, -2) * scale  # (B, H, Tq, Tk)

  if attn_mask is not None:
    keep_mask = attn_mask.to(dtype=torch.bool)
    scores = scores.masked_fill(~keep_mask, float("-inf"))

  probs = torch.softmax(scores.to(dtype=torch.float32), dim=-1).to(dtype=query.dtype)
  return probs @ value  # (B, H, Tq, Dh)


def scaled_dot_product_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  *,
  is_causal: bool = False,
  enable_gqa: bool = False,
  attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
  """

  Args:
    query: shape (B, Hq, Tq, Dh)
    key: shape (B, Hk, Tk, Dh)
    value: shape (B, Hk, Tk, Dh)
  """

  if enable_gqa:
    B, Hq, Tq, Dh = query.shape
    _, Hk, Tk, _ = key.shape
    assert Hq % Hk == 0
    G = Hq // Hk
    key = key.reshape(B, Hk, 1, Tk, Dh).expand(B, Hk, G, Tk, Dh).reshape(B, Hq, Tk, Dh)
    value = value.reshape(B, Hk, 1, Tk, Dh).expand(B, Hk, G, Tk, Dh).reshape(B, Hq, Tk, Dh)

  if is_causal:
    assert attn_mask is None
    Tq, Tk = query.size(-2), key.size(-2)
    attn_mask = torch.ones((Tq, Tk), dtype=torch.bool, device=query.device).tril()

  return _scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
