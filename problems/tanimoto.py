from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

from .molecular_problem import DecoderInterface, MolecularProblem

__all__ = ["TanimotoSimilarity"]


class TanimotoSimilarity(MolecularProblem):
    """
    Optimize the similarity of a molecule to a given target molecule.

    Calculate the Tanimoto similarity between the molecular fingerprint of a molecule
    and the target molecule.
    """

    def __init__(
        self,
        target_value: str,
        n_var: int,
        lbound: float,
        ubound: float,
        decoder: DecoderInterface,
        *args,
        **kwargs
    ):
        super().__init__(target_value, n_var, lbound, ubound, decoder, *args, **kwargs)
        self.target_mol = Chem.MolFromSmiles(target_value)
        self.target_fp = AllChem.GetMorganFingerprint(self.target_mol, 2)

    def evaluate_mols(self, mols: List[str]) -> np.ndarray:
        """
        Calculates the similarity of a list of molecules to the target molecule.
        """
        mols_fp = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(m), 2) for m in mols]
        similarities = np.array(
            [BulkTanimotoSimilarity(self.target_fp, [fp])[0] for fp in mols_fp]
        )

        return 1 - similarities

    def _evaluate(self, x, out, *args, **kwargs):
        mols = self.decode_population(x)
        out["F"] = self.evaluate_mols(mols)
