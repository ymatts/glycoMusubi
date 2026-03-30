#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parallel.py

Multiprocessing utilities for the glycoMusubi pipeline.
Provides helpers for parallel processing with ProcessPoolExecutor.
"""

import os
import random
import tempfile
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterator, List, Optional, Tuple, TypeVar
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


def get_worker_count(requested: Optional[int] = None, default: Optional[int] = None) -> int:
    """
    Get the number of worker processes to use.
    
    Args:
        requested: Explicitly requested worker count (from CLI).
        default: Default from config (None means auto-detect).
    
    Returns:
        Number of workers to use.
    """
    if requested is not None and requested > 0:
        return requested
    if default is not None and default > 0:
        return default
    return os.cpu_count() or 1


def chunked(iterable: List[T], chunk_size: int) -> Iterator[List[T]]:
    """
    Split an iterable into chunks of specified size.
    
    Args:
        iterable: List to split.
        chunk_size: Maximum size of each chunk.
    
    Yields:
        Lists of at most chunk_size elements.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]


def chunked_by_count(iterable: List[T], n_chunks: int) -> Iterator[Tuple[int, List[T]]]:
    """
    Split an iterable into approximately n_chunks equal parts.
    
    Args:
        iterable: List to split.
        n_chunks: Number of chunks to create.
    
    Yields:
        Tuples of (chunk_index, chunk_data).
    """
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive")
    
    total = len(iterable)
    if total == 0:
        return
    
    chunk_size = max(1, (total + n_chunks - 1) // n_chunks)
    
    for i, start in enumerate(range(0, total, chunk_size)):
        yield i, iterable[start:start + chunk_size]


def seed_worker(worker_id: int, base_seed: int = 42):
    """
    Seed random number generators for deterministic behavior in workers.
    
    Args:
        worker_id: Unique identifier for the worker.
        base_seed: Base seed value.
    """
    seed = base_seed + worker_id
    random.seed(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def run_in_process_pool(
    func: Callable,
    tasks: List[Any],
    total_items: int,
    desc: str,
    workers: int,
    show_progress: bool = True
) -> Iterator[Any]:
    """
    Run tasks in a ProcessPoolExecutor with progress tracking.
    
    Args:
        func: Worker function to execute. Must be a top-level function.
        tasks: List of task arguments to pass to func.
        total_items: Total number of items being processed (for progress bar).
        desc: Description for the progress bar.
        workers: Number of worker processes.
        show_progress: Whether to show progress bar.
    
    Yields:
        Results from each completed task.
    """
    if not tasks:
        return
    
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        show_progress = False
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for task in tasks:
            future = executor.submit(func, task)
            if isinstance(task, tuple) and len(task) > 0 and isinstance(task[0], list):
                futures[future] = len(task[0])
            else:
                futures[future] = 1
        
        if show_progress and has_tqdm:
            with tqdm(total=total_items, desc=desc) as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        pbar.update(futures[future])
                        yield result
                    except Exception as e:
                        logger.error(f"Worker task failed: {e}")
                        pbar.update(futures[future])
                        yield {"error": str(e)}
        else:
            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    completed += futures[future]
                    if not show_progress:
                        logger.info(f"{desc}: {completed}/{total_items}")
                    yield result
                except Exception as e:
                    logger.error(f"Worker task failed: {e}")
                    completed += futures[future]
                    yield {"error": str(e)}


def run_sequential(
    func: Callable,
    tasks: List[Any],
    total_items: int,
    desc: str,
    show_progress: bool = True
) -> Iterator[Any]:
    """
    Run tasks sequentially (non-parallel fallback).
    
    Args:
        func: Worker function to execute.
        tasks: List of task arguments to pass to func.
        total_items: Total number of items being processed.
        desc: Description for the progress bar.
        show_progress: Whether to show progress bar.
    
    Yields:
        Results from each completed task.
    """
    if not tasks:
        return
    
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        show_progress = False
    
    if show_progress and has_tqdm:
        with tqdm(total=total_items, desc=desc) as pbar:
            for task in tasks:
                try:
                    result = func(task)
                    if isinstance(task, tuple) and len(task) > 0 and isinstance(task[0], list):
                        pbar.update(len(task[0]))
                    else:
                        pbar.update(1)
                    yield result
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    if isinstance(task, tuple) and len(task) > 0 and isinstance(task[0], list):
                        pbar.update(len(task[0]))
                    else:
                        pbar.update(1)
                    yield {"error": str(e)}
    else:
        completed = 0
        for task in tasks:
            try:
                result = func(task)
                if isinstance(task, tuple) and len(task) > 0 and isinstance(task[0], list):
                    completed += len(task[0])
                else:
                    completed += 1
                logger.info(f"{desc}: {completed}/{total_items}")
                yield result
            except Exception as e:
                logger.error(f"Task failed: {e}")
                completed += 1
                yield {"error": str(e)}


def atomic_write(path: Path, write_fn: Callable[[Any], None], mode: str = 'w'):
    """
    Write to a file atomically using a temporary file and rename.
    
    Args:
        path: Target file path.
        write_fn: Function that takes a file handle and writes content.
        mode: File mode ('w' for text, 'wb' for binary).
    """
    path = Path(path)
    tmp_dir = path.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = path.suffix or '.tmp'
    
    with tempfile.NamedTemporaryFile(
        mode=mode,
        delete=False,
        dir=tmp_dir,
        suffix=suffix
    ) as f:
        tmp_name = f.name
        write_fn(f)
    
    os.replace(tmp_name, path)


def merge_dicts_with_dedup(
    dicts: List[dict],
    merge_fn: Optional[Callable[[dict, dict], dict]] = None
) -> dict:
    """
    Merge multiple dictionaries, handling duplicate keys.
    
    Args:
        dicts: List of dictionaries to merge.
        merge_fn: Optional function to merge values for duplicate keys.
                  If None, later values overwrite earlier ones.
    
    Returns:
        Merged dictionary.
    """
    result = {}
    
    for d in dicts:
        for key, value in d.items():
            if key in result and merge_fn is not None:
                result[key] = merge_fn(result[key], value)
            else:
                result[key] = value
    
    return result


class ParallelConfig:
    """Configuration for parallel processing."""
    
    def __init__(
        self,
        enabled: bool = True,
        workers: Optional[int] = None,
        seed: int = 42,
        batch_size: int = 100
    ):
        self.enabled = enabled
        self.workers = workers
        self.seed = seed
        self.batch_size = batch_size
    
    def get_effective_workers(self, requested: Optional[int] = None) -> int:
        """Get effective worker count considering all sources."""
        return get_worker_count(requested, self.workers)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for passing to workers."""
        return {
            "enabled": self.enabled,
            "workers": self.workers,
            "seed": self.seed,
            "batch_size": self.batch_size,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ParallelConfig':
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            workers=data.get("workers"),
            seed=data.get("seed", 42),
            batch_size=data.get("batch_size", 100),
        )
