import random
from typing import Iterable

import numpy as np

BIG_NUMBER = 1e7
SMALL_NUMBER = 1e-7


def pick_indices_from_probs(probs: np.ndarray, num_picks: int, use_sampling: bool=False,
                            temperature: float=0.5) -> Iterable[int]:
    """Given an array of probabilities, pick up to num_samples unique indices from it."""
    if use_sampling:
        # First, consider the temperature for sampling:
        probs = probs ** (1.0 / temperature)
        normaliser = np.sum(probs)
        probs = probs / normaliser

        probs_cum = np.cumsum(probs)
        probs_cum[-1] = 1.0  # To protect against floating point oddness
        picked_indices = set()
        remaining_picks = num_picks * 10
        while len(picked_indices) < num_picks and remaining_picks > 0:
            remaining_picks -= 1
            picked_val = random.random()
            picked_index = np.argmax(probs_cum >= picked_val)  # type: int
            if picked_index not in picked_indices and probs[picked_index] > SMALL_NUMBER:
                picked_indices.add(picked_index)
        return picked_indices
    else:
        num_samples = min(num_picks, len(probs))
        top_k_indices = np.argpartition(probs, -num_samples)[-num_samples:]
        top_k_indices = [index for index in top_k_indices if probs[index] > SMALL_NUMBER]
        return top_k_indices
