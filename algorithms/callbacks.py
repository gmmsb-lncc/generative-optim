import argparse
import os
from pymoo.core.callback import Callback
from aim import Run, Text, Image, Figure
import numpy as np

import pandas as pd
from problems.molproblem import MolecularProblem
from pymoo.visualization.scatter import Scatter
from matplotlib import pyplot as plt

__all__ = ["AimCallback"]


def pareto_front_plot(algorithm):
    f = algorithm.pop.get("F")
    fig, axis = plt.subplots()
    axis.set_xlabel("$f_1(x)$")
    axis.set_ylabel("$f_2(x)$")
    axis.scatter(f[:, 0], f[:, 1])
    return fig


class AimCallback(Callback):
    def __init__(
        self, run: Run, problem: MolecularProblem, args: argparse.Namespace
    ) -> None:
        super().__init__()
        self.run = run
        self.problem = problem
        self.args = args

    def notify(self, algorithm):
        n_gen = algorithm.n_gen
        min_property = self.problem.get_min_property_history()[-1]
        avg_property = self.problem.get_avg_property_history()[-1]
        fitness = algorithm.pop.get("F")

        # adjust shape for single objective problems
        if self.problem.n_obj == 1:
            fitness = fitness[:, np.newaxis]
            min_property = np.array([min_property])
            avg_property = np.array([avg_property])

        for i in range(self.problem.n_obj):
            self.run.track(fitness[:, i].min(), name=f"min_fitness_{i}", step=n_gen)
            self.run.track(fitness[:, i].mean(), name=f"avg_fitness_{i}", step=n_gen)
            self.run.track(min_property[i], name=f"min_property_{i}", step=n_gen)
            self.run.track(avg_property[i], name=f"avg_property_{i}", step=n_gen)

        # create a pandas dataframe for the current generation
        df = pd.DataFrame()
        df["chrm"] = algorithm.pop.get("X").tolist()
        for i in range(self.problem.n_obj):
            df[f"f{i}"] = fitness[:, i].squeeze()

        # save dataframe as a csv and track
        repo = os.path.join(self.run.repo.path, "meta/chunks", self.run.hash)
        df.to_csv(os.path.join(repo, f"gen={n_gen}.csv"), index=False)
        # self.run.track(Text(df.to_csv(index=False)), name="population", step=n_gen)

        # save pareto front plot
        if self.problem.n_obj > 1 and n_gen == self.args.max_gens:
            fig = pareto_front_plot(algorithm)
            self.run.track(Figure(fig), name="pareto_front")
