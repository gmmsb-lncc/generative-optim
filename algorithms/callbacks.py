import os

import numpy as np
import pandas as pd
from aim import Run
from pymoo.core.callback import Callback

__all__ = ["AimCallback"]


class AimCallback(Callback):
    def __init__(
        self,
        run: Run,
        n_obj: int,  # number of objectives
    ) -> None:
        super().__init__()
        self.run = run
        self.n_obj = n_obj

    def notify(self, algorithm):
        n_gen = algorithm.n_gen
        fitness = algorithm.pop.get("F")

        if self.n_obj == 1:
            fitness = fitness[:, np.newaxis]

        for i in range(self.n_obj):
            self.run.track(
                fitness[:, i].min(), name=f"min_fitness_obj={i+1}", step=n_gen
            )
            self.run.track(
                fitness[:, i].mean(), name=f"avg_fitness_obj={i+1}", step=n_gen
            )

        df = pd.DataFrame()
        df["individual"] = algorithm.pop.get("X").tolist()
        for i in range(self.n_obj):
            df[f"f{i}"] = fitness[:, i].squeeze()

        repo = os.path.join(self.run.repo.path, "meta/chunks", self.run.hash)
        df.to_csv(os.path.join(repo, f"gen={n_gen}.csv"), index=False)
