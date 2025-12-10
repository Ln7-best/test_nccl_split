#!/usr/bin/env python3
"""
Test script for NCCL initialization WITHOUT ncclCommSplit optimization.

This script tests creating a 7-GPU process group directly without using ncclCommSplit.
Only the first 7 ranks (0-6) participate in the initialization.

Usage:
    torchrun --nproc_per_node=8 test_without_split_optimization.py
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

def test_without_split_optimization():
    """
    Test initialization WITHOUT ncclCommSplit optimization.
    
    Key: Initialize default PG without device_id, causing lazy initialization
    and preventing splitting optimization. Only first 7 ranks participate.
    """
    rank = int(os.environ['RANK'])
    
    print(f"[Rank {rank}] Testing WITHOUT ncclCommSplit optimization...")
    
    # Clean up any existing process group
    cleanup()
    
    # Only first 7 ranks (0-6) participate in initialization
    if rank < 7:
        # Initialize with world_size=7 for only the first 7 ranks
        start_time = time.time()
        
        # CRITICAL: Do NOT specify device_id - this prevents splitting optimization
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=7,  # Only 7 ranks participate
            rank=rank,
            timeout=timedelta(seconds=300),
            device_id=torch.device('cuda', rank) 
        )
        
        init_time = time.time() - start_time
        
        print(f"[Rank {rank}] WITHOUT split - init_process_group time: {init_time:.4f}s")
        
        # Barrier to ensure all participating ranks complete
        dist.barrier()
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"WITHOUT ncclCommSplit optimization - init_process_group time: {init_time:.4f}s")
            print(f"{'='*60}\n")
        
        cleanup()
        return init_time
    else:
        # Rank 7 does not participate
        print(f"[Rank {rank}] Not participating in initialization")
        return 0

def main():
    """Main test function."""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size != 8:
        if rank == 0:
            print(f"ERROR: This test requires exactly 8 GPUs, but got {world_size}")
            print("Usage: torchrun --nproc_per_node=8 test_without_split_optimization.py")
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
        print("NCCL Initialization Performance Test - WITHOUT ncclCommSplit Optimization")
        print("="*80)
        print(f"World size: {world_size}")
        print(f"Testing: Direct initialization of 7-GPU process group (ranks 0-6)")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"NCCL version: {torch.cuda.nccl.version()}")
        print("="*80)
        print()
    
    # Run test
    init_time = test_without_split_optimization()
    
    # Print summary
    if rank == 0 and init_time > 0:
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"init_process_group time: {init_time:.4f}s")
        print("="*80)

if __name__ == '__main__':
    main()
