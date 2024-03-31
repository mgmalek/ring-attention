from datetime import datetime
from enum import Enum
from itertools import product
import logging
from multiprocessing import Queue
from pathlib import Path
from statistics import mean

import pandas as pd
import torch
import torch.distributed as dist

from ring_attn import RingAttention, AttentionType
from utils import combine_traces, get_device, run_distributed_fn


class SweepList(list):
    pass


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
    queue: Queue,
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
        peak_mem_gbs = []
        for iter_num in range(num_iters):
            if rank == 0:
                logging.info(f"Starting {iter_num=}")
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.reset_peak_memory_stats()
            initial_peak_mem_bytes = torch.cuda.max_memory_allocated(device=device)

            start_event.record()
            dist_out = attn(x_shard)
            dist_out.sum().backward()
            end_event.record()

            final_peak_mem_bytes = torch.cuda.max_memory_allocated(device=device)
            peak_mem_consumed_bytes = final_peak_mem_bytes - initial_peak_mem_bytes
            peak_mem_consumed_gb = peak_mem_consumed_bytes / 1e9

            if iter_num >= num_warmup_iters:
                events.append((start_event, end_event))
                peak_mem_gbs.append(peak_mem_consumed_gb)
        
        torch.cuda.synchronize()

        durations = []
        for start_event, end_event in events:
            durations.append(start_event.elapsed_time(end_event))
        
        mean_duration = mean(durations)
        logging.info(f"{mean_duration = :8.4f}")

        mean_peak_mem_gb = mean(peak_mem_gbs)
        logging.info(f"{mean_peak_mem_gb = :8.4f}")

        queue.put(dict(rank=rank, mean_duration=mean_duration, mean_peak_mem_gb=mean_peak_mem_gb))

    else:
        raise ValueError(f"Invalid {test_type=}")

    logging.info(f"Waiting at final barrier")
    dist.barrier()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    world_size = 4

    test_config = dict(
        batch_size=2,
        dim_per_head=SweepList((64, 128)),
        dim=SweepList((1024, 2048, 4096, 8192)),
        seq_len=SweepList((16384, 32768, 65536)),
        attn_type=SweepList((AttentionType.STRIPED, AttentionType.STAIRCASE)),
        dtype=torch.bfloat16,
        num_warmup_iters=4,
        num_active_iters=4,
        test_type=PerfTestType.TIMING,
    )

    queue = Queue()

    timing_results = []

    common_kwargs = {k: v for k, v in test_config.items() if not isinstance(v, SweepList)}
    all_sweep_kwargs = {k: v for k, v in test_config.items() if isinstance(v, SweepList)}
    sweep_keys, all_sweep_vals = zip(*all_sweep_kwargs.items())

    for sweep_vals in product(*all_sweep_vals):
        sweep_kwargs = dict(zip(sweep_keys, sweep_vals))
        attn_kwargs = dict(**common_kwargs, **sweep_kwargs)

        attn_type = attn_kwargs["attn_type"]
        seq_len = attn_kwargs["seq_len"]
        
        causal = {
            AttentionType.RING: False,
            AttentionType.STRIPED: True,
            AttentionType.STAIRCASE: True,
        }[attn_type]

        attn_kwargs["causal"] = causal

        print(f"Starting test with the following configuration:\n{attn_kwargs}")

        log_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir_name = f"{log_dir_name}_seqlen_{seq_len}_type_{attn_type.name}_causal_{causal}"
        log_dir = Path("./log") / log_dir_name

        kwargs = dict(**attn_kwargs, log_dir=log_dir, queue=queue)
        run_distributed_fn(_test_attn_perf, world_size=world_size, kwargs=kwargs)

        while not queue.empty():
            timing_results.append(dict(**queue.get(), **attn_kwargs))

        if log_dir.exists():
            combine_traces(log_dir)

    if len(timing_results):
        log_dir = Path("./log")
        log_dir.mkdir(exist_ok=True, parents=True)

        timing_output_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_timings.csv"

        timing_df = pd.DataFrame(timing_results)
        timing_df.to_csv(timing_output_path, index=False)
        print(f"Wrote timing data to {str(timing_output_path)}")
