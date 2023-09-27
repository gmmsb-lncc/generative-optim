import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from utils.sa_score import calculateScore

from hgraph.hiervae import HierVAEDecoder
from .molproblem import MolecularProblem


__all__ = ["MolecularWeight", "SAScore"]


class MolecularWeight(MolecularProblem):
    """Generate compounds with a molecular weight constraint.

    The pymoo `Problem` class evaluates a batch of solutions at once (ideal for
    vectorized operations). The actual function evaluation takes place in the _evaluate
    method, which aims to fill the out dictionary with the corresponding data.

    In pymoo, each objective function is supposed to be minimized, and each constraint
    needs to be provided in the form of <= 0.

    """

    def __init__(
        self, mw_target, n_var=32, xl=-1, xu=1, decoder: str = "HierVAEDecoder"
    ):
        self.mw_target = mw_target
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
        """Calculate the molecular weight of a batch of compounds."""
        assert x.ndim == 2, "x must be two-dimensional"
        assert x.shape[1] == self.n_var, "x must have the correct number of variables"

        sols = torch.from_numpy(x)
        smiles = self.decoder.decode(torch.as_tensor(sols, dtype=torch.float32))
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        mws = [MolWt(m) for m in mols]

        return np.array(mws)

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        """Returns the squared difference between the target and molecular weight.

        The function values are supposed to be written into `out["F"]` and the
        constraints into `out["G"]` if `n_constr` is greater than zero.

        The input `x` is a two-dimensional array of size (n_sols, n_var).
        `out["F"]` should be a vector of size `n_var` (since we have one obj only).

        """
        properties = self.calc_property(x)
        fitness = np.square(self.mw_target - properties)

        self.min_property_history.append(properties[fitness.argmin()])
        self.avg_property_history.append(properties.mean())

        out["F"] = fitness


class SAScore(MolecularProblem):
    """Generate compounds with maximum SA."""

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
