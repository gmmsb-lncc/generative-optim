from abc import ABC, abstractmethod
from pymoo.core.problem import Problem
import numpy as np

__all__ = ["MolecularProblem"]


class MolecularProblem(ABC, Problem):
    """Abstract base class for molecular optimization problems."""

    @abstractmethod
    def calc_property(self, x: np.ndarray) -> np.ndarray:
        """Calculate the molecular property for a batch of compounds."""
        pass

    @abstractmethod
    def get_decoder(self):
        """Return the decoder object."""
        pass

    @abstractmethod
    def get_min_property_history(self):
        """Return the minimum property history."""
        pass

    @abstractmethod
    def get_avg_property_history(self):
        """Return the average property history."""
        pass
