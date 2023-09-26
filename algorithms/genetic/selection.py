import numpy as np
from pymoo.operators.selection.tournament import TournamentSelection

__all__ = ["BinaryTournament"]


class BinaryTournament(TournamentSelection):
    def __init__(self, *args, **kwargs):
        super().__init__(pressure=2, func_comp=self.func_comp, *args, **kwargs)

    def func_comp(self, population, tournaments_idxs: np.ndarray, *args, **kwargs):
        n_tournaments, n_parents = tournaments_idxs.shape

        # res = np.full(n_tournaments, np.inf)
        res = np.full(n_tournaments, -1, dtype=np.int)
        for t in range(n_tournaments):
            a, b = tournaments_idxs[t]
            res[t] = a if population[a].F < population[b].F else b

        return res
