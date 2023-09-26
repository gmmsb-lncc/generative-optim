import argparse
import os

from aim import Run
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.optimize import minimize

from algorithms.genetic import BinaryTournament, GaussianMutationFix, Population
from algorithms import AimCallback
from problems import *


def main(args: argparse.Namespace):
    problem = eval(args.optim_prob)(
        args.property_target,
        n_var=args.num_vars,
        xl=args.lbound,
        xu=args.ubound,
        decoder=args.decoder,
    )

    population = Population(args.population_size, args.num_vars, args.seed).initialize()
    xover = PointCrossover(prob=args.xover_prob, n_points=args.xover_points)
    mutation = GaussianMutationFix(sigma=args.mutation_sigma, prob=args.mutation_prob)
    selection = BinaryTournament()

    algorithm = GA(
        pop_size=args.population_size,
        sampling=population,
        selection=selection,
        crossover=xover,
        mutation=mutation,
    )

    # setup callback
    run = Run(experiment=args.experiment)
    run["hparams"] = vars(args)

    # begin optimization
    termination_criteria = ("n_gen", args.max_gens)
    result = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=termination_criteria,
        seed=args.seed,
        verbose=args.verbose,
        callback=AimCallback(run, problem),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "single-objective optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # script level parameters
    script_args = parser.add_argument_group("optimization arguments")
    script_args.add_argument("--experiment", type=str, help="experiment name", required=True)
    script_args.add_argument("--max-gens", type=int, default=20, help="number of generations")
    script_args.add_argument("--seed", type=int, default=42, help="random seed")
    script_args.add_argument("--verbose", action="store_true", help="print progress")

    # problem parameters
    problem_args = parser.add_argument_group("problem arguments")
    problem_args.add_argument("--property-target", type=float, default=800, help="molecular property target value")
    problem_args.add_argument("--optim-prob", type=str, default="MolecularWeight", help="optimization problem")
    problem_args.add_argument("--num-vars", type=int, default=32, help="number of variables")
    problem_args.add_argument("--decoder", type=str, default="HierVAEDecoder", help="decoder model")

    # genetic algorithm parameters
    ga_args = parser.add_argument_group("genetic algorithm arguments")
    ga_args.add_argument("--population-size", type=int, default=40, help="population size")
    ga_args.add_argument("--num-offspring", type=int, default=None, help="number of offspring")
    ga_args.add_argument("--mutation-sigma", type=float, default=0.01, help="gaussian mutation strength")
    ga_args.add_argument("--mutation-prob", type=float, default=0.1, help="mutation probability")
    ga_args.add_argument("--xover-points", type=int, default=2, help="number of crossover points")
    ga_args.add_argument("--xover-prob", type=float, default=0.9, help="crossover probability")
    ga_args.add_argument("--ubound", type=float, default=2.5, help="gene value upper bound")
    ga_args.add_argument("--lbound", type=float, default=-2.5, help="gene value lower bound")
    # fmt: on

    args = parser.parse_args()
    main(args)
