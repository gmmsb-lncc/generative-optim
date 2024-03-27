"""Optimze the molecular weight of a molecule."""

from typing import Any, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED

from .molecular_problem import DecoderInterface, MolecularProblem

__all__ = ["QEDProblem"]


class QEDProblem(MolecularProblem):
    """Optimize the QED (quantitative estimate of drug-likeness) of a molecule.

    Minimize the abs difference between the QED of a molecule and the target value in
    [0, 1].
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
        mols = [Chem.MolFromSmiles(m) for m in mols]
        qeds = np.array([QED.qed(mol) for mol in mols])
        fitness = np.abs(qeds - self.target)
        return fitness

    def _evaluate(self, x, out, *args, **kwargs):
        mols = self.decode_population(x)
        out["F"] = self.evaluate_mols(mols)
