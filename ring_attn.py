from enum import Enum

from einops import rearrange
from triton_attn import flash_attn_func, ring_attn_func, striped_attn_func
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class AttentionType(Enum):
    SDPA = 0
    FLASH = 1
    RING = 2
    STRIPED = 3


class RingAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        dim_per_head: int,
        seq_len: int,
        rank: int,
        n_ranks: int,
        dropout: float,
        causal: bool,
        attn_type: AttentionType = AttentionType.FLASH,
    ):
        super().__init__()
        assert isinstance(attn_type, AttentionType), attn_type
        assert dim % dim_per_head == 0, (dim, dim_per_head)
        assert seq_len % n_ranks == 0, (seq_len, n_ranks)

        if attn_type == AttentionType.STRIPED:
            assert causal, causal

        self.dim = dim
        self.dim_per_head = dim_per_head
        self.n_heads = dim // dim_per_head
        self.seq_len = seq_len
        self.seq_len_per_rank = seq_len // n_ranks
        self.dropout = dropout
        self.causal = causal
        self.attn_type = attn_type

        self.rank = rank
        self.n_ranks = n_ranks
        self.prev_rank = (self.rank - 1 + n_ranks) % n_ranks
        self.next_rank = (self.rank + 1) % n_ranks

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.o_drop = nn.Dropout(p=dropout)
    
    def shard(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape

        if self.attn_type == AttentionType.RING:
            x_shard = torch.split(x, seq_len // self.n_ranks, dim=1)[self.rank].contiguous()
        elif self.attn_type == AttentionType.STRIPED:
            x_shard = x.view(batch_size, seq_len // self.n_ranks, self.n_ranks, dim)[:, :, self.rank]
        elif self.attn_type in (AttentionType.SDPA, AttentionType.FLASH):
            x_shard = x
        else:
            raise ValueError(f"Invalid {self.attn_type=}")

        return x_shard

    def unshard(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn_type in (AttentionType.SDPA, AttentionType.FLASH):
            return x

        batch_size, shard_seq_len, dim = x.shape
        seq_len = self.n_ranks * shard_seq_len

        all_x = [torch.zeros_like(x) for _ in range(self.n_ranks)]
        dist.all_gather(all_x, x)

        if self.attn_type == AttentionType.RING:
            all_x = torch.cat(all_x, dim=1)
        elif self.attn_type == AttentionType.STRIPED:
            all_x = torch.stack(all_x, dim=2).view(batch_size, seq_len, dim)
        else:
            raise ValueError(f"Invalid {self.attn_type=}")

        return all_x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, "b n (three h d) -> three b n h d", three=3, h=self.n_heads, d=self.dim_per_head)
        q, k, v = torch.unbind(qkv, dim=0)
        sm_scale = k.shape[-1] ** -0.5

        if self.attn_type == AttentionType.SDPA:
            # NOTE: this is just for testing purposes
            o = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), is_causal=self.causal).permute(0, 2, 1, 3)
        elif self.attn_type == AttentionType.FLASH:
            o = flash_attn_func(q, k, v, None, self.causal, sm_scale)
        elif self.attn_type == AttentionType.RING:
            o = ring_attn_func(q, k, v, self.n_ranks, self.rank, self.prev_rank, self.next_rank, None, self.causal, sm_scale)
        elif self.attn_type == AttentionType.STRIPED:
            o = striped_attn_func(q, k, v, self.n_ranks, self.rank, self.prev_rank, self.next_rank, None, self.causal, sm_scale)
        else:
            raise ValueError(f"Invalid {self.attn_type=}")

        o = rearrange(o, "b n h d -> b n (h d)")
        o = self.o_proj(o)
        return o
