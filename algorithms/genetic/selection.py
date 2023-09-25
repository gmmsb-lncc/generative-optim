import numpy as np
from pymoo.operators.selection.tournament import TournamentSelection

__all__ = ["BinaryTournament"]


class BinaryTournament(TournamentSelection):
    def __init__(self, *args, **kwargs):
        super().__init__(pressure=2, func_comp=self.func_comp, *args, **kwargs)

    def func_comp(self, population, tournaments_idx: np.ndarray, *args, **kwargs):
        n_tournaments, n_parents = tournaments_idx.shape

        res = np.full(n_tournaments, np.inf)
        for t in range(n_tournaments):
            a, b = tournaments_idx[t]
            res[t] = a if population[a].F[0] < population[b].F[0] else b

        return res
