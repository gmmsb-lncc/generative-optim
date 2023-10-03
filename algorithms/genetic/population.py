import numpy as np
import torch

from hgraph.hiervae import HierVAEDecoder

__all__ = ["Population"]


class Population:
    """Initializes a population with values from a gaussian distribution."""

    def __init__(
        self,
        population_size: int,
        n_vars: int = 32,
        seed: int = 42,
        xl: float = -2.5,
        xu: float = 2.5,
        decoder=HierVAEDecoder(),
    ) -> None:
        self.popsize = population_size
        self.n_vars = n_vars
        self.xl = xl
        self.xu = xu
        self.rng = np.random.default_rng(seed)
        self.decoder = decoder

    def _clip(self, population: np.ndarray) -> np.ndarray:
        """Clips the population values to the upper and lower bounds."""
        population[population > self.xu] = self.xu - np.random.random()
        population[population < self.xl] = self.xl + np.random.random()
        return population

    def export_population(self, population: np.ndarray, filename: str) -> np.ndarray:
        """Writes the final population of SMILES strings to a file."""
        sols = torch.from_numpy(population)
        smiles = self.decoder.decode(torch.as_tensor(sols, dtype=torch.float32))
        with open(filename, "w") as f:
            for s in smiles:
                f.write(s + "\n")

    def initialize(self) -> None:
        """Each gene value is sampled from a gaussian distribution."""
        population = self.rng.normal(size=(self.popsize, self.n_vars))
        population = self._clip(population)
        return population
