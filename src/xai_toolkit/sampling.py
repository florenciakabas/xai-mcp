"""Batch sampling strategies — select which rows to explain (Layer 2).

Pure functions, no framework dependency. Strategies:
  - "random": uniform random selection (reproducible via random_state)
  - "uncertainty": prefer samples near the decision boundary (probability 0.3–0.7)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def select_samples(
    model,
    X: pd.DataFrame,
    n_samples: int = 100,
    strategy: str = "random",
    random_state: int = 42,
) -> list[int]:
    """Select sample indices for batch explanation.

    Args:
        model: A fitted model with predict_proba().
        X: Feature matrix to sample from.
        n_samples: Number of samples to select.
        strategy: "random" or "uncertainty".
        random_state: Seed for reproducibility.

    Returns:
        List of integer indices into X.
    """
    n_available = len(X)

    if n_samples >= n_available:
        if n_samples > n_available:
            logger.warning(
                "Requested %d samples but dataset has only %d rows. "
                "Returning all indices.",
                n_samples, n_available,
            )
        return list(range(n_available))

    rng = np.random.RandomState(random_state)

    if strategy == "random":
        return sorted(rng.choice(n_available, size=n_samples, replace=False).tolist())

    if strategy == "uncertainty":
        return _select_uncertain(model, X, n_samples, rng)

    raise ValueError(
        f"Unknown sampling strategy '{strategy}'. "
        "Supported: 'random', 'uncertainty'."
    )


def _select_uncertain(
    model,
    X: pd.DataFrame,
    n_samples: int,
    rng: np.random.RandomState,
) -> list[int]:
    """Select samples near the decision boundary (probability 0.3–0.7).

    If not enough uncertain samples exist, falls back to random for the
    remainder.
    """
    probas = model.predict_proba(X)[:, 1]

    # Indices where model is uncertain (near decision boundary)
    uncertain_mask = (probas >= 0.3) & (probas <= 0.7)
    uncertain_indices = np.where(uncertain_mask)[0]

    if len(uncertain_indices) == 0:
        logger.warning(
            "No samples with probability in [0.3, 0.7]. "
            "Falling back to random sampling."
        )
        return sorted(rng.choice(len(X), size=n_samples, replace=False).tolist())

    if len(uncertain_indices) >= n_samples:
        # Enough uncertain samples — pick closest to 0.5
        distances = np.abs(probas[uncertain_indices] - 0.5)
        closest = np.argsort(distances)[:n_samples]
        return sorted(uncertain_indices[closest].tolist())

    # Not enough uncertain samples — take all uncertain + fill with random
    logger.info(
        "Only %d uncertain samples (need %d). "
        "Filling remainder with random samples.",
        len(uncertain_indices), n_samples,
    )
    remaining_indices = np.setdiff1d(np.arange(len(X)), uncertain_indices)
    n_fill = n_samples - len(uncertain_indices)
    fill = rng.choice(remaining_indices, size=n_fill, replace=False)
    return sorted(np.concatenate([uncertain_indices, fill]).tolist())
