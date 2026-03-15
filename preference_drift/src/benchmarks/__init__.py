"""Benchmark registry and factory."""

from .crows_pairs import CrowsPairsBenchmark

BENCHMARK_REGISTRY = {
    "crows_pairs": CrowsPairsBenchmark,
}


def create_benchmark(name: str, params: dict):
    """Create a benchmark instance by name."""
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available: {list(BENCHMARK_REGISTRY.keys())}"
        )
    return BENCHMARK_REGISTRY[name](params)
