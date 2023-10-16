import numpy as np

from hgraph.hiervae import HierVAEDecoder
from problems.molecular_problem import MolecularProblem

__all__ = ["QED"]


class QED(MolecularProblem):
    """Optimize QED score."""

    def __init__(
        self,
        qed_target: float,
        n_var=32,
        xl=-1,
        xu=1,
        decoder: str = "HierVAEDecoder",
    ):
        self.qed_target = qed_target
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
        properties = np.array([self.qed(m) for m in mols])
        fitness = np.abs(self.qed_target - properties)

        # log properties
        self.min_property_history.append(properties[fitness.argmin()])
        self.avg_property_history.append(properties.mean())

        out["F"] = fitness
