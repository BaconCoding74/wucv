import torch

from contextlib import contextmanager
from time import perf_counter

@contextmanager
def benchmark(name: str, use_cuda: bool = False):
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = perf_counter()
    yield

    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = perf_counter()

    elapsed_time = end_time - start_time
    print(f"\033[33m[BENCHMARK] {name}: {elapsed_time:.6f} seconds\033[0m")