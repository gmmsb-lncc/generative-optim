import numpy as np
import torch
from rdkit import Chem

from hgraph.hiervae import HierVAEDecoder
from problems.molecular_problem import MolecularProblem
from utils.sa_score import calculateScore

__all__ = ["MWSA"]


class MWSA(MolecularProblem):
    """Optim problem with two objs: molecular weight and synthetic accessibility.

    PROPERTY 1: molecular weight
    PROPERTY 2: synthetic accessibility
    """

    def __init__(
        self,
        prop_targets: tuple = (800.0, 1.0),  # (mw_target, sa_target)
        n_var=32,
        xl=-2.5,
        xu=2.5,
        decoder: str = "HierVAEDecoder",
    ):
        self.mw_target = prop_targets[0]
        self.sa_target = prop_targets[1]
        self.decoder = eval(decoder)()
        self.min_property_history = list()
        self.avg_property_history = list()

        super().__init__(
            n_var=n_var,
            n_obj=2,  # two objectives: molecular weight and synthetic accessibility
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        """Calculate the fitness function for a batch of compounds.

        The F1 is the squared difference between the target molecular weight and the
        molecular weight of the compound. The F2 is the squared difference between the
        target sa score and the sa score of the compound.

        PROPERTY 1: molecular weight
        PROPERTY 2: synthetic accessibility
        """
        props = self.calc_property(x)
        f1 = np.square(self.mw_target - props[:, 0])
        f2 = np.square(self.sa_target - props[:, 1])

        best_mw = props[:, 0][f1.argmin()]
        best_sa = props[:, 1][f2.argmin()]
        self.min_property_history.append((best_mw, best_sa))
        self.avg_property_history.append((props[:, 0].mean(), props[:, 1].mean()))

        out["F"] = [f1, f2]

    def calc_property(self, x: np.ndarray) -> np.ndarray:
        """Calculate the molecular property for a batch of compounds.

        Returns:
            np.ndarray: shape (n_var, n_obj)
        """
        # get molecules from latent space vectors
        sols = torch.from_numpy(x)
        smiles = self.decoder.decode(torch.as_tensor(sols, dtype=torch.float32))
        mols = [Chem.MolFromSmiles(s) for s in smiles]

        # calculate properties
        props = [(Chem.Descriptors.MolWt(m), calculateScore(m)) for m in mols]
        return np.array(props)

    def get_decoder(self):
        return self.decoder

    def get_min_property_history(self):
        """Return the minimum property history."""
        return np.array(self.min_property_history)

    def get_avg_property_history(self):
        """Return the average property history."""
        return np.array(self.avg_property_history)
