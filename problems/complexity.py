"""Optimze the Bottcher score (complexity) of a small molecule."""

from typing import Any, List

import numpy as np

from utils.bottchscore import calculate_bottchscore_from_smiles

from .molecular_problem import DecoderInterface, MolecularProblem

__all__ = ["ComplexityProblem"]


class ComplexityProblem(MolecularProblem):
    """Optimize the Bottcher score (complexity) of a small molecule.

    Minimize the squared difference between the Bottcher score (complexity) and
    the target value in [0, +inf).
    """

    def __init__(
        self,
        target_value: float,
        n_var: int,
        lbound: float,
        ubound: float,
        decoder: DecoderInterface,
        *args,
        **kwargs
    ):
        super().__init__(target_value, n_var, lbound, ubound, decoder, *args, **kwargs)

    def evaluate_mols(self, mols: List[str]) -> np.ndarray:
        """Calculates the fitness of a list of molecules based on the target value."""
        scores = np.array([calculate_bottchscore_from_smiles(m) for m in mols])
        fitness = np.square(scores - self.target)
        return fitness

    def _evaluate(self, x, out, *args, **kwargs):
        mols = self.decode_population(x)
        out["F"] = self.evaluate_mols(mols)
