import gzip
import json
import logging
from multiprocessing import Queue
import os
from pathlib import Path
import traceback
from typing import Callable, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def get_device() -> torch.device:
    # TODO: generalize this for multinode
    return torch.device("cuda", index=dist.get_rank())


def setup_distributed_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    bf = logging.Formatter(
        fmt='[{rank}/{world_size}] {levelname:8s} {message}',
        style='{',
        defaults=dict(rank=dist.get_rank(), world_size=dist.get_world_size()),
    )
    handler.setFormatter(bf)
    root_logger.addHandler(handler)


def init_process(q: Queue, rank: int, world_size: int, fn: Callable, kwargs, backend: str = "nccl"):
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29504'

        # NOTE: the default size of NCCL_BUFFSIZE is 4MB. Increasing the size of this
        # buffer to 64MB avoids hangs when running collective ops on large tensors.
        os.environ['NCCL_BUFFSIZE'] = str(64 * 1024 * 1024)

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        setup_distributed_logging()
        logging.info(f"Rank {dist.get_rank()} has joined the process group")
        fn(**kwargs)
    except Exception as e:
        q.put(traceback.format_exc())


def run_distributed_fn(fn: Callable, world_size: int, kwargs=None):
    processes: List[mp.Process] = []
    
    assert kwargs is None or isinstance(kwargs, dict), type(kwargs)
    kwargs = kwargs or {}

    q = Queue()

    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(q, rank, world_size, fn, kwargs))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    exceptions = []
    while not q.empty():
        exceptions.append(q.get())

    if len(exceptions):
        exceptions_str = "\n".join(exceptions)
        raise Exception(f"Found the following exceptions in child processes:\n{exceptions_str}")


def combine_traces(log_dir: Path) -> None:
    logging.info("Combining all traces into a single trace file")

    all_trace_data = []
    for trace_path in log_dir.glob("*.pt.trace.json"):
        with open(trace_path, "r") as f:
            trace_data = json.load(f)
        all_trace_data.append(trace_data)
    
    all_trace_data.sort(key=lambda v: v["distributedInfo"]["rank"])
        
    combined_trace = None
    for trace_data in all_trace_data:
        trace_events = trace_data.pop("traceEvents")
        distributed_info = trace_data.pop("distributedInfo")
        rank = distributed_info["rank"]

        if combined_trace is None:
            combined_trace = trace_data
            combined_trace["traceEvents"] = []

        for e in trace_events:
            if (pid := e.get("pid")):
                if isinstance(pid, str):
                    e["pid"] = f"[rank {rank}] {pid}"
                else:
                    e["pid"] = rank * 1_000_000 + pid
            if (args := e.get("args")) and (name := args.get("name")):
                args["name"] = f"[rank {rank}] {name}"
        
        combined_trace["traceEvents"].extend(trace_events)

    with open(log_dir / "combined_trace.pt.trace.json", "w") as f:
        json.dump(combined_trace, f, indent=2)

    with gzip.open(log_dir / "combined_trace.pt.trace.json.gz", "wt") as f:
        json.dump(combined_trace, f, indent=2)
