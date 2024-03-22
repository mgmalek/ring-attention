from datetime import datetime
from enum import Enum
import logging
from pathlib import Path
from statistics import mean

import torch
import torch.distributed as dist

from ring_attn import RingAttention, AttentionType
from utils import combine_traces, get_device, run_distributed_fn


class PerfTestType(Enum):
    PROFILE = 0
    TIMING = 1


def _test_attn_perf(
    batch_size: int,
    dim: int,
    dim_per_head: int,
    seq_len: int,
    dtype: torch.dtype,
    causal: bool,
    num_warmup_iters: int,
    num_active_iters: int,
    attn_type: AttentionType,
    test_type: PerfTestType,
    log_dir: Path,
):
    torch.manual_seed(42)
    dtype = torch.bfloat16

    device = get_device()
    torch.cuda.set_device(device)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Initialize Models
    attn = RingAttention(
        dim, dim_per_head, seq_len, rank=rank, n_ranks=world_size,
        dropout=0.0, causal=causal, attn_type=attn_type,
    )
    attn.to(device=device, dtype=dtype)

    x = torch.randn(batch_size, seq_len, dim, dtype=dtype, device=device)
    x_shard = attn.shard(x)

    # Run profiling
    logging.info("Waiting at initial barrier")
    dist.barrier()

    num_iters = num_warmup_iters + num_active_iters

    if test_type == PerfTestType.PROFILE:
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=num_warmup_iters, active=num_active_iters, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for iter_num in range(num_iters):
                if rank == 0:
                    logging.info(f"Starting {iter_num=}")
                prof.step()
                dist_out = attn(x_shard)
                dist_out.sum().backward()

    elif test_type == PerfTestType.TIMING:
        events = []
        for iter_num in range(num_iters):
            if rank == 0:
                logging.info(f"Starting {iter_num=}")
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            dist_out = attn(x_shard)
            dist_out.sum().backward()
            end_event.record()

            if iter_num >= num_warmup_iters:
                events.append((start_event, end_event))
        
        torch.cuda.synchronize()

        durations = []
        for start_event, end_event in events:
            durations.append(start_event.elapsed_time(end_event))
        
        mean_duration = mean(durations)
        logging.info(f"{mean_duration = :8.4f}")

    else:
        raise ValueError(f"Invalid {test_type=}")

    logging.info(f"Waiting at final barrier")
    dist.barrier()


if __name__ == "__main__":
    world_size = 4

    common_kwargs = dict(
        batch_size=2,
        dim=4096,
        dim_per_head=128,
        dtype=torch.bfloat16,
        num_warmup_iters=4,
        num_active_iters=4,
        test_type=PerfTestType.PROFILE,
    )

    for seq_len, attn_type in [
        (4_096, AttentionType.RING),
        (8_192, AttentionType.RING),
        (16_384, AttentionType.RING),
        (32_768, AttentionType.RING),
        (65_536, AttentionType.RING),
        (4_096, AttentionType.STRIPED),
        (8_192, AttentionType.STRIPED),
        (16_384, AttentionType.STRIPED),
        (32_768, AttentionType.STRIPED),
        (65_536, AttentionType.STRIPED),
    ]:
        causal = {
            AttentionType.RING: False,
            AttentionType.STRIPED: True,
        }[attn_type]

        logging.getLogger().setLevel(logging.DEBUG)
        log_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir_name = f"{log_dir_name}_seqlen_{seq_len}_type_{attn_type.name}_causal_{causal}"
        log_dir = Path("./log") / log_dir_name

        kwargs = dict(**common_kwargs, seq_len=seq_len, attn_type=attn_type, causal=causal, log_dir=log_dir)
        run_distributed_fn(_test_attn_perf, world_size=world_size, kwargs=kwargs)

        if log_dir.exists():
            combine_traces(log_dir)
