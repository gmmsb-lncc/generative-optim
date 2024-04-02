"""Optimze the molecular weight of a molecule."""

from typing import Any, List

import numpy as np
from rdkit import Chem

from utils.sa_score import calculateScore

from .molecular_problem import DecoderInterface, MolecularProblem

__all__ = ["SAProblem"]


class SAProblem(MolecularProblem):
    """Optimize the SA (synthetic accessibility) score of a molecule.

    Minimize the squared difference between the molecular weight of a molecule and
    the target value [1, 10].
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

    def calculate_property(self, mols: List[str]) -> np.ndarray:
        """Calculates the SA score of a list of molecules."""
        mols = [Chem.MolFromSmiles(m) for m in mols]
        scores = np.array([calculateScore(mol) for mol in mols])
        return scores

    def evaluate_mols(self, mols: List[str]) -> np.ndarray:
        """Calculates the fitness of a list of molecules based on the target value."""
        scores = self.calculate_property(mols)
        fitness = np.square(scores - self.target)
        return fitness

    def _evaluate(self, x, out, *args, **kwargs):
        mols = self.decode_population(x)
        out["F"] = self.evaluate_mols(mols)
