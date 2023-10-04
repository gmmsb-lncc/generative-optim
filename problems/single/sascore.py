import numpy as np

from hgraph.hiervae import HierVAEDecoder
from problems.molecular_problem import MolecularProblem

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

    def get_min_property_history(self) -> np.ndarray:
        return np.array(self.min_property_history)

    def get_avg_property_history(self) -> np.ndarray:
        return np.array(self.avg_property_history)

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        mols = self.vector2molecule(x, self.decoder)
        properties = np.array([self.synthetic_accessibility(m) for m in mols])
        fitness = np.square(self.sa_target - properties)

        self.min_property_history.append(properties[fitness.argmin()])
        self.avg_property_history.append(properties.mean())

        out["F"] = fitness
