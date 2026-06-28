"""Deep legal analysis package for Persian contract review."""

from .orchestrator import run_deep_analysis
from .presentation import generate_persian_presentation

__all__ = ["run_deep_analysis", "generate_persian_presentation"]
