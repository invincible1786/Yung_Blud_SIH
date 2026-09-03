"""
Benchmark runner entrypoint shim.
Forwards to src/benchmark/runner.py.
"""
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmark.runner import run_benchmarks, get_algo_key, ALGO_KEY_MAP

if __name__ == "__main__":
    run_benchmarks()
