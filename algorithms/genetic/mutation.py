import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.core.variable import Real, get
from pymoo.operators.repair.bounds_repair import repair_random_init

__all__ = ["GaussianMutationFix"]


class GaussianMutationFix(Mutation):
    def __init__(self, sigma, prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = sigma
        self.prob = prob

    def gauss_mutation(self, population, xl, xu, sigma, prob):
        n, n_var = population.shape
        assert len(sigma) == n
        assert len(prob) == n

        mutated_pop = np.full(population.shape, np.inf)
        mutated_pop[:, :] = population

        mut = np.random.random(population.shape) < prob[:, None]

        _xl = np.repeat(xl[None, :], population.shape[0], axis=0)[mut]
        _xu = np.repeat(xu[None, :], population.shape[0], axis=0)[mut]
        sigma = sigma[:, None].repeat(n_var, axis=1)[mut]

        mutated_pop[mut] = np.random.normal(population[mut], sigma * abs(_xu - _xl))
        mutated_pop = repair_random_init(mutated_pop, population, xl, xu)

        return mutated_pop

    def _do(self, problem, population, **kwargs):
        population = population.astype(float)
        sigma = get(self.sigma, size=len(population))
        prob_var = get(self.prob, size=len(population))
        mutated_pop = self.gauss_mutation(
            population, problem.xl, problem.xu, sigma, prob_var
        )

        return mutated_pop
