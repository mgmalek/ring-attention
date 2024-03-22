from copy import deepcopy
import logging

import pytest
import torch
import torch.distributed as dist

from ring_attn import RingAttention, AttentionType
from utils import get_device, run_distributed_fn


COMMON_TEST_PARAMS = {
    "batch_size": [4],
    "dim": [1024],
    "dim_per_head": [64],
}


def _test_attn(
    batch_size: int,
    dim: int,
    dim_per_head: int,
    seq_len: int,
    causal: bool,
    base_attn_type: AttentionType,
    cand_attn_type: AttentionType,
):
    torch.manual_seed(42)
    dtype = torch.bfloat16

    device = get_device()
    torch.cuda.set_device(device)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Initialize models
    standard_attn = RingAttention(
        dim, dim_per_head, seq_len, rank=rank, n_ranks=world_size,
        dropout=0.0, causal=causal, attn_type=base_attn_type
    )
    standard_attn.to(device=device, dtype=dtype)

    dist_attn = deepcopy(standard_attn)
    dist_attn.attn_type = cand_attn_type

    # Initialize dummy inputs
    x = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
    x_shard = dist_attn.shard(x)

    # Forward model
    logging.info("Waiting at initial barrier")
    dist.barrier()

    logging.info(f"Forwarding local model")
    x_local = x.clone().requires_grad_(True)
    local_out = standard_attn(x_local)
    local_out.sum().backward()

    logging.info(f"Forwarding distributed model")
    x_shard = x_shard.clone().requires_grad_(True)
    dist_out = dist_attn(x_shard)
    dist_out.sum().backward()

    # Gather outputs across devices
    all_dist_out = dist_attn.unshard(dist_out)
    all_x_shard_grad = dist_attn.unshard(x_shard.grad)

    # Compare outputs
    if dist.get_rank() == 0:
        logging.info("Comparing outputs")
        logging.info(f"{torch.linalg.norm(local_out - all_dist_out).item()         = :12.8f}")
        logging.info(f"{torch.mean(torch.abs(local_out - all_dist_out)).item()      = :12.8f}")
        logging.info(f"{torch.max(torch.abs(local_out - all_dist_out)).item()      = :12.8f}")

        logging.info("Comparing input gradients")
        logging.info(f"{torch.linalg.norm(x_local.grad - all_x_shard_grad).item()         = :12.8f}")
        logging.info(f"{torch.mean(torch.abs(x_local.grad - all_x_shard_grad)).item()      = :12.8f}")
        logging.info(f"{torch.max(torch.abs(x_local.grad - all_x_shard_grad)).item()      = :12.8f}")
 
    logging.info(f"Waiting at final barrier")
    dist.barrier()


def common_test_params(fn):
    for param_name, param_vals in COMMON_TEST_PARAMS.items():
        fn = pytest.mark.parametrize(param_name, param_vals)(fn)
    return fn


@common_test_params
@pytest.mark.parametrize("seq_len", [1536])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn(batch_size, dim, dim_per_head, seq_len, causal):
    logging.getLogger().setLevel(logging.DEBUG)
    world_size = 1
    base_attn_type = AttentionType.SDPA
    cand_attn_type = AttentionType.FLASH
    kwargs = dict(
        batch_size=batch_size, dim=dim, dim_per_head=dim_per_head, seq_len=seq_len,
        causal=causal, base_attn_type=base_attn_type, cand_attn_type=cand_attn_type
    )
    run_distributed_fn(_test_attn, world_size=world_size, kwargs=kwargs)


@common_test_params
@pytest.mark.parametrize("seq_len", [1536])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("world_size", list(range(torch.cuda.device_count())))
def test_ring_attn(batch_size, dim, dim_per_head, seq_len, causal, world_size):
    logging.getLogger().setLevel(logging.DEBUG)
    base_attn_type = AttentionType.FLASH
    cand_attn_type = AttentionType.RING
    kwargs = dict(
        batch_size=batch_size, dim=dim, dim_per_head=dim_per_head, seq_len=seq_len,
        causal=causal, base_attn_type=base_attn_type, cand_attn_type=cand_attn_type
    )
    run_distributed_fn(_test_attn, world_size=world_size, kwargs=kwargs)


@common_test_params
@pytest.mark.parametrize("seq_len", [1536])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("world_size", list(range(torch.cuda.device_count())))
def test_striped_attn(batch_size, dim, dim_per_head, seq_len, causal, world_size):
    logging.getLogger().setLevel(logging.DEBUG)
    base_attn_type = AttentionType.FLASH
    cand_attn_type = AttentionType.STRIPED
    kwargs = dict(
        batch_size=batch_size, dim=dim, dim_per_head=dim_per_head, seq_len=seq_len,
        causal=causal, base_attn_type=base_attn_type, cand_attn_type=cand_attn_type
    )
    run_distributed_fn(_test_attn, world_size=world_size, kwargs=kwargs)
