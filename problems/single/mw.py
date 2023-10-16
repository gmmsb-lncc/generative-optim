import numpy as np

from hgraph.hiervae import HierVAEDecoder
from problems.molecular_problem import MolecularProblem

__all__ = ["MolecularWeight"]


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

    def get_min_property_history(self) -> np.ndarray:
        return np.array(self.min_property_history)

    def get_avg_property_history(self) -> np.ndarray:
        return np.array(self.avg_property_history)

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        """Returns the squared difference between the target and molecular weight.

        The function values are supposed to be written into `out["F"]` and the
        constraints into `out["G"]` if `n_constr` is greater than zero.

        The input `x` is a two-dimensional array of size (n_sols, n_var).
        `out["F"]` should be a vector of size `n_var` (since we have one obj only).

        """
        # properties = self.calc_property(x)
        mols = self.vector2molecule(x, self.decoder)
        properties = np.array([self.molecular_weight(m) for m in mols])

        fitness = np.square(self.mw_target - properties)

        self.min_property_history.append(properties[fitness.argmin()])
        self.avg_property_history.append(properties.mean())

        out["F"] = fitness
