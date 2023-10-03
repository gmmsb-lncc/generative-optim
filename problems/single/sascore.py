import numpy as np
import torch
from rdkit import Chem

from hgraph.hiervae import HierVAEDecoder
from problems.molecular_problem import MolecularProblem
from utils.sa_score import calculateScore

__all__ = ["SAScore"]


class SAScore(MolecularProblem):
    """Generate compounds with target SA."""

    def __init__(
        self, sa_target, n_var=32, xl=-1, xu=1, decoder: str = "HierVAEDecoder"
    ):
        self.sa_target = sa_target
        self.decoder = eval(decoder)()
        self.min_property_history = list()
        self.avg_property_history = list()

        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
        )

    def get_decoder(self) -> np.ndarray:
        return self.decoder

    def get_min_property_history(self) -> np.ndarray:
        return np.array(self.min_property_history)

    def get_avg_property_history(self) -> np.ndarray:
        return np.array(self.avg_property_history)

    def calc_property(self, x: np.ndarray) -> np.ndarray:
        """Calculate the SA score of a batch of compounds."""
        assert x.ndim == 2, "x must be two-dimensional"
        assert x.shape[1] == self.n_var, "x must have the correct number of variables"

        sols = torch.from_numpy(x)
        smiles = self.decoder.decode(torch.as_tensor(sols, dtype=torch.float32))
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        sas = [calculateScore(m) for m in mols]

        return np.array(sas)

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        properties = self.calc_property(x)
        fitness = np.square(self.sa_target - properties)

        self.min_property_history.append(properties[fitness.argmin()])
        self.avg_property_history.append(properties.mean())

        out["F"] = fitness
