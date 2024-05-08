import sys
from typing import Any, List

import numpy as np

from .complexity import ComplexityProblem
from .molecular_problem import DecoderInterface, MolecularProblem
from .qed import QEDProblem
from .sascore import SAProblem
from .tanimoto import TanimotoDissimProblem, TanimotoSimProblem

__all__ = ["CxQEDSAProblem", "CxQEDSASimDissimProblem"]


class CxQEDSAProblem(MolecularProblem):
    """This is simply the summation of the Complexity, QED and SA problems."""

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
        # define targets manually here
        self.complexity = ComplexityProblem(0.0, n_var, lbound, ubound, decoder)
        self.qed = QEDProblem(1.0, n_var, lbound, ubound, decoder)
        self.sa = SAProblem(1.0, n_var, lbound, ubound, decoder)

    def evaluate_mols(self, mols: List[str]) -> np.ndarray:
        """Calculates the fitness of a list of molecules based on the target value."""
        c_fitness = self.complexity.evaluate_mols(mols)
        qed_fitness = self.qed.evaluate_mols(mols)
        sa_fitness = self.sa.evaluate_mols(mols)
        fitness = c_fitness + qed_fitness + sa_fitness
        return fitness

    def _evaluate(self, x, out, *args, **kwargs):
        mols = self.decode_population(x)
        out["F"] = self.evaluate_mols(mols)


class CxQEDSASimDissimProblem(MolecularProblem):
    """This is the aggregation of the Complexity, QED, SA, TanimotoSim and TanimotoDissim problems."""

    def __init__(
        self,
        target_value: float,
        n_var: int,
        lbound: float,
        ubound: float,
        decoder: DecoderInterface,
        weights: List[float] = None,
        *args,
        **kwargs
    ):
        super().__init__(target_value, n_var, lbound, ubound, decoder, *args, **kwargs)
        # define targets manually here
        self.complexity = ComplexityProblem(0.0, n_var, lbound, ubound, decoder)
        self.qed = QEDProblem(1.0, n_var, lbound, ubound, decoder)
        self.sa = SAProblem(1.0, n_var, lbound, ubound, decoder)
        self.tanimoto_sim = TanimotoSimProblem(
            "CCCC1=C(C(=CC(=C1)C2=NC(CO2)C(=O)NO)OC)OC", n_var, lbound, ubound, decoder
        )
        self.tanimoto_dissim = TanimotoDissimProblem(
            "O=C(NO)[C@@H](Cc1ccc2ccccc2c1)NS(=O)(=O)c1ccc2ccccc2c1",
            n_var,
            lbound,
            ubound,
            decoder,
        )
        self.weights = weights

    def get_normalization_fn(self, max_value: float, min_value: float) -> Any:
        return lambda x: (x - min_value) / (max_value - min_value)

    def evaluate_mols(self, mols: List[str]) -> np.ndarray:
        """Calculates the fitness of a list of molecules based on the target value."""
        c_fitness = self.complexity.evaluate_mols(mols)
        qed_fitness = self.qed.evaluate_mols(mols)
        sa_fitness = self.sa.evaluate_mols(mols)
        tanimoto_sim_fitness = self.tanimoto_sim.evaluate_mols(mols)
        tanimoto_dissim_fitness = self.tanimoto_dissim.evaluate_mols(mols)

        norm_c = self.get_normalization_fn(1e6 + 1e-3, 0)
        norm_sa = self.get_normalization_fn(81 + 1e-3, 0)

        fitness = (
            norm_c(c_fitness) * self.weights[0]
            + qed_fitness * self.weights[1]
            + norm_sa(sa_fitness) * self.weights[2]
            + tanimoto_sim_fitness * self.weights[3]
            + tanimoto_dissim_fitness * self.weights[4]
        )

        return fitness

    def _evaluate(self, x, out, *args, **kwargs):
        mols = self.decode_population(x)
        out["F"] = self.evaluate_mols(mols)
