from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from pymoo.core.problem import Problem
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Lipinski import NumRotatableBonds

from utils.sa_score import calculateScore

__all__ = ["MolecularProblem"]


class MolecularProblem(ABC, Problem):
    """Abstract base class for molecular optimization problems."""

    @abstractmethod
    def get_min_property_history(self):
        """Return the minimum property history."""
        pass

    @abstractmethod
    def get_avg_property_history(self):
        """Return the average property history."""
        pass

    def vector2molecule(self, population: np.ndarray, decoder) -> List:
        """Convert a population of latent vectors to a molecule.

        Returns:
            A list of RDKit molecules.
        """
        # assert population.ndim == 2, "population must be two-dimensional"
        sols = torch.from_numpy(population)
        smiles = decoder.decode(torch.as_tensor(sols, dtype=torch.float32))
        return [Chem.MolFromSmiles(s) for s in smiles]

    def veber(self, molecule: Chem.rdchem.Mol) -> bool:
        """Apply Veber's rule to a molecule."""
        rotatable_bonds = NumRotatableBonds(molecule)
        tpsa = QED.properties(molecule).PSA
        return rotatable_bonds <= 10 and tpsa <= 140

    def molecular_weight(self, molecule: Chem.rdchem.Mol) -> float:
        """Calculate the molecular weight of a molecule."""
        return MolWt(molecule)

    def synthetic_accessibility(self, molecule: Chem.rdchem.Mol) -> float:
        """Calculate the synthetic accessibility score of a molecule."""
        return calculateScore(molecule)

    def qed(self, molecule: Chem.rdchem.Mol) -> float:
        """Calculate the QED score of a molecule."""
        return QED.qed(molecule)
