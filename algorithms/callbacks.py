from pymoo.core.callback import Callback
from aim import Run, Text
import numpy as np

from problems import MolecularProblem

__all__ = ["AimCallback"]


class AimCallback(Callback):
    def __init__(self, run: Run, problem: MolecularProblem) -> None:
        super().__init__()
        self.run = run
        self.problem = problem

    def notify(self, algorithm):
        min_f_idx = algorithm.pop.get("F").argmin()
        min_f = algorithm.pop.get("F").min()
        avg_f = algorithm.pop.get("F").mean()
        n_gen = algorithm.n_gen
        min_property = self.problem.get_min_property_history()[-1]
        avg_property = self.problem.get_avg_property_history()[-1]

        best_solution_vector = algorithm.pop[min_f_idx].X[np.newaxis, :]
        best_solution_smiles = self.problem.get_decoder().decode(best_solution_vector)

        self.run.track(min_f, name="min_fitness", step=n_gen)
        self.run.track(avg_f, name="avg_fitness", step=n_gen)
        self.run.track(min_property, name="min_property", step=n_gen)
        self.run.track(avg_property, name="avg_property", step=n_gen)

        solutions = {
            "best_solution_smiles": best_solution_smiles,
            "best_solution_vector": best_solution_vector.squeeze().tolist(),
        }
        self.run.track(Text(str(solutions)), name="best_solution", step=n_gen)
