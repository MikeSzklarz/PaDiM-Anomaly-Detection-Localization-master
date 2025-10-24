"""Compatibility shim for older imports.

This module used to contain plotting, embedding and miscellaneous helper
functions. The functionality has been moved into smaller modules under
``utils`` to make the code easier to navigate. To preserve backwards
compatibility for code that imports from ``utils.helpers`` we re-export
the most commonly used helpers here.
"""

from .embeddings import get_batch_embeddings, concatenate_embeddings
from .misc import denormalize_image_for_display
from .plotting import (
    plot_summary_visuals,
    plot_mean_anomaly_maps,
    plot_individual_visualizations,
    plot_patch_score_distributions,
)

__all__ = [
    "get_batch_embeddings",
    "concatenate_embeddings",
    "denormalize_image_for_display",
    "plot_summary_visuals",
    "plot_mean_anomaly_maps",
    "plot_individual_visualizations",
    "plot_patch_score_distributions",
]
