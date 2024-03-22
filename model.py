import torch
import torch.nn as nn

from attn import Attention, AttentionType


class MLP(nn.Sequential):

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__(
            nn.Linear(dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=bias),
        )


class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dim_per_head: int,
        seq_len: int,
        rank: int,
        n_ranks: int,
        dropout: float,
        causal: bool,
        attn_type: AttentionType,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, dim_per_head, seq_len, rank, n_ranks, dropout, causal, attn_type)
        self.mlp = MLP(dim, hidden_dim)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        dim: int,
        hidden_dim: int,
        dim_per_head: int,
        seq_len: int,
        rank: int,
        n_ranks: int,
        dropout: float,
        causal: bool,
        attn_type: AttentionType,
        vocab_size: int,
    ):
        super().__init__()

        self.pos_emb = nn.Parameter(torch.empty(1, seq_len, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.tok_emb = nn.Embedding(vocab_size, dim)

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                hidden_dim=hidden_dim,
                dim_per_head=dim_per_head,
                seq_len=seq_len,
                rank=rank,
                n_ranks=n_ranks,
                dropout=dropout,
                causal=causal,
                attn_type=attn_type,
            )
            for _ in range(n_blocks)
        ])
    
    def forward(self, tok_idxs):
        x = self.tok_emb(tok_idxs) + self.pos_emb
        x = self.blocks(x)
        return x
