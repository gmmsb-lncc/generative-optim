import os

import numpy as np
import torch

from problems import DecoderInterface


class Population:
    """Define a population of individuals."""

    def __init__(self, size: int, n_var: int, seed: int, xl: float, xu: float):
        self.popsize = size
        self.n_vars = n_var
        self.rng = np.random.default_rng(seed)
        self.xl = xl
        self.xu = xu

    def _clip(self, population: np.ndarray) -> np.ndarray:
        """Clips the population values to the upper and lower bounds plus a delta."""
        population[population > self.xu] = self.xu - np.random.random()
        population[population < self.xl] = self.xl + np.random.random()
        return population

    def initialize(self) -> None:
        """Each gene value is sampled from a gaussian distribution."""
        population = self.rng.normal(size=(self.popsize, self.n_vars))
        population = self._clip(population)
        return population

    def export_population(
        self, population: np.ndarray, filename: str, decoder: DecoderInterface
    ) -> np.ndarray:
        """Writes the final population of SMILES strings to a file."""
        sols = torch.from_numpy(population)
        smiles = decoder.decode(torch.as_tensor(sols, dtype=torch.float32))

        self._ensure_dirs_exist(filename)
        with open(filename, "w") as f:
            for s in smiles:
                f.write(s + "\n")

    def _ensure_dirs_exist(self, path: str) -> None:
        """Ensures that the directories in the path exist, otherwise create them."""
        dirpath = os.path.dirname(path)

        if dirpath:
            os.makedirs(os.path.dirname(path), exist_ok=True)
