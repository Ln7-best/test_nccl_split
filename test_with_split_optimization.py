#!/usr/bin/env python3
"""
Test script for NCCL initialization WITH ncclCommSplit optimization.

This script tests creating a 7-GPU subgroup from an 8-GPU setup using ncclCommSplit optimization.
The optimization is enabled by specifying device_id in init_process_group.

Usage:
    torchrun --nproc_per_node=8 test_with_split_optimization.py
"""

import os
import time
import torch
import torch.distributed as dist
from datetime import timedelta

def cleanup():
    """Clean up the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def test_with_split_optimization():
    """
    Test initialization WITH ncclCommSplit optimization.
    
    Key: Initialize default PG with device_id to enable eager initialization
    and splitting support.
    """
    print(f"[Rank {os.environ.get('RANK', 0)}] Testing WITH ncclCommSplit optimization...")
    
    # Clean up any existing process group
    cleanup()
    
    # Step 1: Initialize default process group WITH device_id
    # This enables eager NCCL communicator initialization and splitting
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    start_time = time.time()
    
    # CRITICAL: Specify device_id to enable ncclCommSplit optimization
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=300),
        device_id=torch.device('cuda', rank)  # This enables splitting!
    )
    
    default_pg_init_time = time.time() - start_time
    
    # Step 2: Create a 7-GPU subgroup (exclude rank 7)
    # This should use ncclCommSplit for faster initialization
    subgroup_ranks = list(range(7))  # [0, 1, 2, 3, 4, 5, 6]
    # Barrier to ensure all ranks complete
    dist.barrier()
    subgroup_start = time.time()
    
    if rank < 7:
        subgroup = dist.new_group(ranks=subgroup_ranks, backend='nccl')
        subgroup_init_time = time.time() - subgroup_start
        # do all reduce for the subgroup, and test all reduce performance
        # test all reduce performance
        all_reduce_start = time.time()
        tensor = torch.ones(1000, 1000, device=f'cuda:{rank}')
        dist.all_reduce(tensor, group=subgroup)
        all_reduce_end = time.time()
        all_reduce_time = all_reduce_end - all_reduce_start 
        # Verify the subgroup works
        # if rank == 0:
        #     tensor = torch.ones(1000, 1000, device=f'cuda:{rank}')
        # else:
        #     tensor = torch.zeros(1000, 1000, device=f'cuda:{rank}')
        
        # dist.broadcast(tensor, src=0, group=subgroup)
        
        # # Verify correctness
        # assert torch.allclose(tensor, torch.ones(1000, 1000, device=f'cuda:{rank}')), \
        #     f"Rank {rank}: Broadcast verification failed!"
        
        print(f"[Rank {rank}] WITH split - Default PG init: {default_pg_init_time:.4f}s, "
              f"Subgroup init: {subgroup_init_time:.4f}s, Total: {default_pg_init_time + subgroup_init_time:.4f},"
              f"All reduce: {all_reduce_time:.4f}s"
              )
    else:
        # Rank 7 still needs to participate in new_group call
        dist.new_group(ranks=subgroup_ranks, backend='nccl')
        print(f"[Rank {rank}] WITH split - Default PG init: {default_pg_init_time:.4f}s, "
              f"Not in subgroup")
    
    # Barrier to ensure all ranks complete
    dist.barrier()
    
    total_time = time.time() - start_time
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"WITH ncclCommSplit optimization - Total time: {total_time:.4f}s")
        print(f"{'='*60}\n")
    
    cleanup()
    return default_pg_init_time, subgroup_init_time if rank < 7 else 0, total_time

def main():
    """Main test function."""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size != 8:
        if rank == 0:
            print(f"ERROR: This test requires exactly 8 GPUs, but got {world_size}")
            print("Usage: torchrun --nproc_per_node=8 test_with_split_optimization.py")
        return
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        if rank == 0:
            print("ERROR: CUDA is not available!")
        return
    
    if torch.cuda.device_count() < 8:
        if rank == 0:
            print(f"ERROR: This test requires 8 GPUs, but only {torch.cuda.device_count()} available")
        return
    
    # Set CUDA device for this rank
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print("="*80)
        print("NCCL Initialization Performance Test - WITH ncclCommSplit Optimization")
        print("="*80)
        print(f"World size: {world_size}")
        print(f"Testing: Creating a 7-GPU subgroup (ranks 0-6) from 8-GPU setup")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"NCCL version: {torch.cuda.nccl.version()}")
        print("="*80)
        print()
    
    # Run test
    with_split_times = test_with_split_optimization()
    
    # Print summary
    if rank == 0:
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Default PG init: {with_split_times[0]:.4f}s")
        print(f"Subgroup init:   {with_split_times[1]:.4f}s")
        print(f"Total time:      {with_split_times[2]:.4f}s")
        print("="*80)

if __name__ == '__main__':
    main()
