from typing import Any

import numpy as np

__all__ = ["Population"]


class Population:
    """Initializes a population with values from a gaussian distribution."""

    def __init__(self, population_size: int, n_vars: int = 32, seed: int = 42) -> None:
        self.popsize = population_size
        self.n_vars = n_vars
        self.rng = np.random.default_rng(seed)

    def initialize(self) -> None:
        """Each gene value is sampled from a gaussian distribution."""
        return self.rng.normal(size=(self.popsize, self.n_vars))
