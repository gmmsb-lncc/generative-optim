import argparse
import importlib
import json
import logging
import os
import random
import subprocess
from dataclasses import dataclass
from typing import Tuple

import aim
import numpy as np
import torch
from aim import Run
from pymoo.core.algorithm import Algorithm
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

import problems
from algorithms import AlgorithmFactory
from algorithms.callbacks import AimCallback
from algorithms.operators import BinaryTournament, GaussianMutation_, PointCrossover
from algorithms.population import Population
from hgraph.hiervae import HierVAEDecoder
from problems.molecular_problem import ProblemFactory


def main(args: argparse.Namespace) -> Run:
    seed_everything(args.seed)
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
        ref_dirs=get_reference_directions(
            args.ref_dirs_method, problem.n_obj, args.ref_dirs_n_points
        ),
    )
    algorithm_factory.check_algorithm_n_objs(algorithm, problem.n_obj)

    termination_criteria = ("n_gen", args.max_gens)
    run = configure_callback(args, algorithm)
    result = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=termination_criteria,
        seed=args.seed,
        verbose=args.verbose,
        callback=AimCallback(run, n_obj=problem.n_obj),
    )

    # export final population
    repo = os.path.join(run.repo.path, "meta/chunks", run.hash)
    individuals = result.pop.get("X")
    population.export_population(
        individuals,
        os.path.join(repo, "generated_mols.txt"),
        HierVAEDecoder(),
    )
    # track final population
    with open(os.path.join(repo, "generated_mols.txt"), "r") as f:
        run.track(aim.Text(f.read()), name="solutions")

    return run


def configure_callback(args: argparse.Namespace, algorithm: Algorithm) -> Run:
    run = Run(experiment=args.experiment)
    args.algorithm = algorithm.__class__.__name__
    args.git_hash = _get_git_revision_hash()
    run["hparams"] = vars(args)
    for file, filename in _get_files(args):
        run.track(file, name=filename)
    return run


def configure_problem(args: argparse.Namespace):
    avail_probs = {p: getattr(problems, p) for p in problems.__all__}

    def determine_type(target):
        try:
            return float(target)
        except ValueError:
            return str(target)

    with open(args.objs_file, "r") as f:
        objs = json.load(f)

    user_problems = {}
    for obj in objs["objectives"]:  # name, target
        if obj["name"] not in avail_probs:
            raise ValueError(
                f"Objective '{obj['name']}' not found in available objectives: {avail_probs.keys()}"
            )

        user_problems[f"{obj['name']}_{obj['target']}"] = (
            avail_probs[obj["name"]],
            determine_type(obj["target"]),
        )
    print("Objectives: ", user_problems)

    problem_factory = ProblemFactory()
    for k, v in user_problems.items():
        problem_factory.register_problem(k, v[0])

    problem = problem_factory.create_problem(
        problem_identifiers=list(user_problems.keys()),
        targets=[v[1] for v in user_problems.values()],
        n_var=args.num_vars,
        lbound=args.lbound,
        ubound=args.ubound,
        decoder=HierVAEDecoder(),
    )

    return problem


def _get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def _get_files(args: argparse.Namespace) -> Tuple[aim.Text, str]:
    files_to_track = [args.objs_file, os.path.abspath(__file__)]

    files = []
    for file in files_to_track:
        with open(file, "r") as f:
            file = aim.Text(f.read())
            files.append(file)
    return [(file, filename) for file, filename in zip(files, files_to_track)]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    problem_args.add_argument("--objs-file", type=str, default="objectives.conf.json", help=f"path to objectives configuration file; objectives must be configured inside the conf file. available options: {', '.join(problems.__all__)}")
    problem_args.add_argument("--num-vars", type=int, default=32, help="number of variables")

    # genetic algorithm parameters
    ga_args = parser.add_argument_group("genetic algorithm arguments")
    ga_args.add_argument("--algorithm", type=str, default="GA", help=f"algorithm type; available options: {', '.join(AlgorithmFactory.algorithm_map.keys())}")
    ga_args.add_argument("--population-size", type=int, default=40, help="population size")
    ga_args.add_argument("--num-offspring", type=int, default=None, help="number of offspring")
    ga_args.add_argument("--mutation-sigma", type=float, default=0.01, help="gaussian mutation strength")
    ga_args.add_argument("--mutation-prob", type=float, default=0.1, help="mutation probability")
    ga_args.add_argument("--xover-points", type=int, default=2, help="number of crossover points")
    ga_args.add_argument("--xover-prob", type=float, default=0.9, help="crossover probability")
    ga_args.add_argument("--ubound", type=float, default=2.5, help="gene value upper bound")
    ga_args.add_argument("--lbound", type=float, default=-2.5, help="gene value lower bound")
    ga_args.add_argument("--ref-dirs-method", type=str, default="das-dennis", help="reference directions method")
    ga_args.add_argument("--ref-dirs-n-points", type=int, default=100, help="number of reference directions")
    # fmt: on

    args = parser.parse_args()
    print("Params:", [f"{k}={v}" for k, v in vars(args).items()])

    logging.basicConfig(level=logging.INFO)
    main(args)
