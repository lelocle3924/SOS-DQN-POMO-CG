"""POMO Attention Model for VRPTW (Kool et al. / Kwon et al. style).

Dimension glossary used in comments:
    B   = batch size  (may include POMO expansion: B_orig * P)
    N   = number of nodes  (depot + customers)
    D   = embedding_dim
    H   = num_heads
    d_k = D // H              (per-head dimension)
    F   = node_feature_dim    (raw input features per node)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Building blocks
# ======================================================================

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention (Q, K, V projections + output proj)."""

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_k = embedding_dim // num_heads
        assert self.d_k * num_heads == embedding_dim, (
            "embedding_dim must be divisible by num_heads"
        )

        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_o = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,   # (B, S_q, D)
        key: torch.Tensor,     # (B, S_k, D)
        value: torch.Tensor,   # (B, S_k, D)
        mask: torch.Tensor | None = None,  # (B, S_k) bool, True=masked
    ) -> torch.Tensor:         # (B, S_q, D)
        B = query.size(0)

        Q = self._reshape(self.W_q(query), B)   # (B, H, S_q, d_k)
        K = self._reshape(self.W_k(key), B)     # (B, H, S_k, d_k)
        V = self._reshape(self.W_v(value), B)   # (B, H, S_k, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0.0)

        out = torch.matmul(attn, V)                            # (B,H,S_q,d_k)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_o(out)

    def _reshape(self, x: torch.Tensor, B: int) -> torch.Tensor:
        return x.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)


class EncoderLayer(nn.Module):
    """Self-attention + FF with residual connections and batch-norm."""

    def __init__(self, embedding_dim: int, num_heads: int,
                 ff_dim: int) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(embedding_dim, num_heads)
        self.norm1 = nn.BatchNorm1d(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim),
        )
        self.norm2 = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        h = self.self_attn(x, x, x) + x
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)

        h = self.ff(h) + h
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        return h


# ======================================================================
# Encoder
# ======================================================================

class AttentionEncoder(nn.Module):
    def __init__(self, node_feature_dim: int, embedding_dim: int,
                 num_heads: int, num_layers: int, ff_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(node_feature_dim, embedding_dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(embedding_dim, num_heads, ff_dim)
             for _ in range(num_layers)]
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """node_features: (B, N, F)  →  embeddings: (B, N, D)"""
        x = self.input_proj(node_features)
        for layer in self.layers:
            x = layer(x)
        return x


# ======================================================================
# Decoder  (single-step; called once per decoding step)
# ======================================================================

class AttentionDecoder(nn.Module):
    LOGIT_CLIP = 10.0

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # Project [graph_emb || first_node_emb || last_node_emb]
        self.static_proj = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)
        # Project [remaining_capacity_norm, current_time_norm]
        self.dynamic_proj = nn.Linear(2, embedding_dim, bias=False)

        # Glimpse via multi-head attention
        self.glimpse_mha = MultiHeadAttention(embedding_dim, num_heads)

        # Final logit key projection (separate from encoder keys)
        self.logit_key_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(
        self,
        encoder_out: torch.Tensor,          # (B, N, D)
        graph_emb: torch.Tensor,             # (B, D)
        first_node_idx: torch.Tensor,        # (B,)   long
        last_node_idx: torch.Tensor,         # (B,)   long
        remaining_cap_norm: torch.Tensor,    # (B,)   [0,1]
        current_time_norm: torch.Tensor,     # (B,)   [0,1]
        mask: torch.Tensor,                  # (B, N) bool  True=masked
    ) -> torch.Tensor:                       # (B, N) logits
        B = encoder_out.size(0)
        batch_idx = torch.arange(B, device=encoder_out.device)

        first_emb = encoder_out[batch_idx, first_node_idx]     # (B, D)
        last_emb  = encoder_out[batch_idx, last_node_idx]      # (B, D)

        # Static context
        static_ctx = torch.cat([graph_emb, first_emb, last_emb], dim=-1)
        static_out = self.static_proj(static_ctx)              # (B, D)

        # Dynamic context
        dyn_feat = torch.stack(
            [remaining_cap_norm, current_time_norm], dim=-1
        )                                                       # (B, 2)
        dyn_out = self.dynamic_proj(dyn_feat)                  # (B, D)

        query = (static_out + dyn_out).unsqueeze(1)            # (B, 1, D)

        # Glimpse
        glimpse = self.glimpse_mha(
            query, encoder_out, encoder_out, mask
        ).squeeze(1)                                            # (B, D)

        # Final logits
        keys = self.logit_key_proj(encoder_out)                # (B, N, D)
        logits = torch.bmm(
            glimpse.unsqueeze(1), keys.transpose(1, 2)
        ).squeeze(1) / math.sqrt(self.embedding_dim)           # (B, N)

        logits = self.LOGIT_CLIP * torch.tanh(logits)
        logits = logits.masked_fill(mask, float("-inf"))

        return logits


# ======================================================================
# Full POMO Model
# ======================================================================

class POMOModel(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 7,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        ff_dim: int = 512,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = AttentionEncoder(
            node_feature_dim, embedding_dim, num_heads,
            num_encoder_layers, ff_dim,
        )
        self.decoder = AttentionDecoder(embedding_dim, num_heads)

        # Set later via set_decode_context(); avoids carrying problem data
        # inside nn.Module state_dict.
        self._vehicle_capacity: float = 1.0
        self._depot_tw_end: float = 24.0

    # ------------------------------------------------------------------

    def set_decode_context(self, vehicle_capacity: float,
                           depot_tw_end: float) -> None:
        """Store normalisation constants used by the decoder."""
        self._vehicle_capacity = vehicle_capacity
        self._depot_tw_end = depot_tw_end

    # ------------------------------------------------------------------

    def encode(
        self, node_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode node features.

        Returns (encoder_output, graph_embedding).
        """
        enc_out = self.encoder(node_features)           # (B, N, D)
        graph_emb = enc_out.mean(dim=1)                  # (B, D)
        return enc_out, graph_emb

    # ------------------------------------------------------------------

    def decode_step(
        self,
        encoder_output: torch.Tensor,
        graph_embedding: torch.Tensor,
        state,                              # VRPTWState
        mask: torch.Tensor,
        decode_method: str = "sampling",
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One auto-regressive step.  Returns (action, log_prob)."""

        first_idx = state.first_customer.clone()
        first_idx[~state.route_started] = 0

        cap_norm = (
            state.remaining_capacity / (self._vehicle_capacity + 1e-8)
        ).clamp(0.0, 1.0)
        time_norm = (
            state.current_time / (self._depot_tw_end + 1e-8)
        ).clamp(0.0, 1.0)

        logits = self.decoder(
            encoder_output, graph_embedding,
            first_idx, state.current_node,
            cap_norm, time_norm,
            mask,
        )                                                # (B, N)

        if decode_method == "greedy":
            action = logits.argmax(dim=-1)
            log_prob = torch.zeros(
                action.shape, dtype=torch.float32, device=action.device
            )
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)
            gathered = probs.gather(1, action.unsqueeze(1)).squeeze(1)
            log_prob = torch.log(gathered + 1e-8)

        return action, log_prob
