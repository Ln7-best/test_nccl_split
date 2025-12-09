#!/usr/bin/env python3
"""
Simplified NCCL initialization performance test.

This is a minimal version for quick testing.
Usage: torchrun --nproc_per_node=8 test_nccl_init_simple.py [--with-split|--without-split]
"""

import os
import sys
import time
import torch
import torch.distributed as dist
from datetime import timedelta


def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Parse command line argument
    use_split = '--without-split' not in sys.argv
    
    if rank == 0:
        mode = "WITH device_id (ncclCommSplit enabled)" if use_split else "WITHOUT device_id (full reinit)"
        print(f"\n{'='*60}")
        print(f"Testing: {mode}")
        print(f"{'='*60}\n")
    
    torch.cuda.set_device(rank)
    
    # Initialize default process group
    start = time.time()
    
    if use_split:
        # WITH ncclCommSplit optimization
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=300),
            device_id=torch.device('cuda', rank)  # Key: enables splitting
        )
    else:
        # WITHOUT ncclCommSplit optimization
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=300)
            # Key: no device_id, prevents splitting
        )
    
    default_init_time = time.time() - start
    
    # Create 7-GPU subgroup
    subgroup_start = time.time()
    subgroup_ranks = list(range(7))
    
    if rank < 7:
        subgroup = dist.new_group(ranks=subgroup_ranks, backend='nccl')
        subgroup_init_time = time.time() - subgroup_start
        
        # Quick verification
        tensor = torch.ones(100, device=f'cuda:{rank}') if rank == 0 else torch.zeros(100, device=f'cuda:{rank}')
        dist.broadcast(tensor, src=0, group=subgroup)
        assert torch.allclose(tensor, torch.ones(100, device=f'cuda:{rank}'))
        
        print(f"[Rank {rank}] Default PG: {default_init_time:.4f}s | Subgroup: {subgroup_init_time:.4f}s | Total: {default_init_time + subgroup_init_time:.4f}s")
    else:
        dist.new_group(ranks=subgroup_ranks, backend='nccl')
        print(f"[Rank {rank}] Default PG: {default_init_time:.4f}s | Not in subgroup")
    
    dist.barrier()
    
    if rank == 0:
        print(f"\n{'='*60}\n")
    
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
