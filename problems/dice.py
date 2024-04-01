from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from .molecular_problem import DecoderInterface, MolecularProblem

# from rdkit.DataStructs import BulkTanimotoSimilarity


__all__ = ["DiceSimProblem", "DiceDissimProblem"]


class DiceSimProblem(MolecularProblem):
    """
    Optimize the similarity of a molecule to a given target molecule.

    Maximize the Dice similarity between the molecular fingerprint of a molecule
    and the target molecule (SMILES representation). This problem is formulated as a
    minimization problem by returning 1 - similarity.
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
        self.target_fp = AllChem.GetMorganFingerprintAsBitVect(
            self.target_mol, radius=2
        )

    @staticmethod
    def bulk_dice_similarity(target_fp, fps):
        """
        Calculates the Dice similarity between a target fingerprint and a list of fingerprints.

        Parameters:
        - target_fp: The target fingerprint as an RDKit fingerprint object.
        - fps: A list of RDKit fingerprint objects to compare against the target.

        Returns:
        - A numpy array of Dice similarity scores.
        """
        target_arr = np.array(target_fp)
        fps_arr = [np.array(fp) for fp in fps]

        # Calculate Dice similarity using the formula: 2 * |A ∩ B| / (|A| + |B|)
        # where |A ∩ B| is the bitwise AND (np.bitwise_and) and |A| and |B| are counts of bits set to 1
        intersection = np.array(
            [np.sum(np.bitwise_and(target_arr, fp_arr)) for fp_arr in fps_arr]
        )
        target_count = np.sum(target_arr)
        fps_count = np.array([np.sum(fp_arr) for fp_arr in fps_arr])
        dice_scores = 2 * intersection / (target_count + fps_count)

        return dice_scores

    def evaluate_mols(self, mols: List[str]) -> np.ndarray:
        """
        Calculates the similarity of a list of molecules to the target molecule.
        """
        mols_fp = [
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(m), radius=2)
            for m in mols
        ]
        similarities = self.bulk_dice_similarity(self.target_fp, mols_fp)

        # since the objective is to maximize the similarity, we return 1 - similarity
        return 1 - similarities

    def _evaluate(self, x, out, *args, **kwargs):
        mols = self.decode_population(x)
        out["F"] = self.evaluate_mols(mols)


class DiceDissimProblem(DiceSimProblem):
    """
    Optimize the dissimilarity of a molecule to a given target molecule.

    Maximize the Dice dissimilarity between the molecular fingerprint of a molecule
    and the target molecule (SMILES representation). This problem is formulated as a
    minimization problem by returning the similarity itself.
    """

    def evaluate_mols(self, mols: List[str]) -> np.ndarray:
        """
        Calculates the dissimilarity of a list of molecules to the target molecule.
        """
        mols_fp = [
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(m), 2)
            for m in mols
        ]
        similarities = self.bulk_dice_similarity(self.target_fp, mols_fp)

        # since the objective is to minimize the similarity, we return the similarity itself
        return similarities
