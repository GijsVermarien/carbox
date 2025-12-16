"""Comparison tools for benchmark results."""

from .compare_abundances import compare_abundances
from .compare_performance import (
    generate_multi_network_comparison,
    generate_performance_report,
)

__all__ = [
    "compare_abundances",
    "generate_performance_report",
    "generate_multi_network_comparison",
]
