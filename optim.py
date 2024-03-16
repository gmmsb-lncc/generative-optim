import argparse
import os
import subprocess
from dataclasses import dataclass

import aim
from aim import Run
from pymoo.core.algorithm import Algorithm
from pymoo.optimize import minimize

from algorithms import AlgorithmFactory
from algorithms.callbacks import AimCallback
from algorithms.operators import BinaryTournament, GaussianMutation_, PointCrossover
from algorithms.population import Population
from hgraph.hiervae import HierVAEDecoder
from problems import MolecularProblem, MolecularWeight, ProblemFactory


def main(args: argparse.Namespace):
    problem = configure_problem(args)
    population = Population(
        args.population_size, args.num_vars, args.seed, xl=args.lbound, xu=args.ubound
    )
    xover = PointCrossover(prob=args.xover_prob, n_points=args.xover_points)
    mutation = GaussianMutation_(sigma=args.mutation_sigma, prob=args.mutation_prob)
    selection = BinaryTournament()

    algorithm_factory = AlgorithmFactory()
    algorithm = algorithm_factory.create_algorithm(
        algorithm_type=args.algorithm,
        pop_size=args.population_size,
        sampling=population.initialize(),
        selection=selection,
        crossover=xover,
        mutation=mutation,
    )

    termination_criteria = ("n_gen", args.max_gens)
    run = configure_callback(args, algorithm)
    result = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=termination_criteria,
        seed=args.seed,
        verbose=args.verbose,
        callback=AimCallback(run, n_obj=len(args.optim_probs)),
    )

    # export final population
    individuals = result.pop.get("X")
    population.export_population(
        individuals,
        f"generated_molecules_run={run.hash}.txt",
        HierVAEDecoder(),
    )


def configure_callback(args: argparse.Namespace, algorithm: Algorithm) -> Run:
    run = Run(experiment=args.experiment)
    args.algorithm = algorithm.__class__.__name__
    args.git_hash = _get_git_revision_hash()
    run["hparams"] = vars(args)
    run.track(_get_this_file(), name="script")
    return run


def configure_problem(args: argparse.Namespace):
    @dataclass
    class ProblemConfig:
        problem: MolecularProblem
        target_value: float

    problem_identifiers = {
        "MolecularWeight": ProblemConfig(MolecularWeight, 100),
    }

    problem_factory = ProblemFactory()
    for k, v in problem_identifiers.items():
        problem_factory.register_problem(k, v.problem)

    problem = problem_factory.create_problem(
        problem_identifiers=args.optim_probs,
        targets=[v.target_value for v in problem_identifiers.values()],
        n_var=args.num_vars,
        lbound=args.lbound,
        ubound=args.ubound,
        decoder=HierVAEDecoder(),
    )

    return problem


def _get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def _get_this_file() -> None:
    this_file = os.path.abspath(__file__)
    with open(this_file, "r") as f:
        file = aim.Text(f.read())

    return file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Multi- and many-objective optimization in generative chemistry model latent spaces.",
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
    problem_args.add_argument("--optim-probs", type=str, nargs="+", default=["MolecularWeight"], help=f"optimization problem")
    problem_args.add_argument("--num-vars", type=int, default=32, help="number of variables")

    # genetic algorithm parameters
    ga_args = parser.add_argument_group("genetic algorithm arguments")
    ga_args.add_argument("--algorithm", type=str, default="GA", help=f"algorithm type; options: {', '.join(AlgorithmFactory.algorithm_map.keys())}")
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
