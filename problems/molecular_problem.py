from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol, Type, Union

import numpy as np
import torch
from pymoo.core.problem import Problem

__all__ = ["ProblemFactory", "CompositeProblem", "MolecularProblem", "DecoderInterface"]


class DecoderInterface(Protocol):
    """Interface for decoders."""

    def decode(self, vecs: torch.Tensor) -> List[str]:
        """Decode a set of latent variables to their molecular representations."""
        pass


class MolecularProblem(Problem, ABC):
    """Abstract base class for single-objective molecular optimization problems.

    Subclasses must implement the evaluate_mols method and _evaluate method (from pymoo).
    """

    def __init__(
        self,
        target_value: Any,
        n_var: int,
        lbound: float,
        ubound: float,
        decoder: DecoderInterface,
        n_constr: int = 0,
        n_obj: int = 1,
    ):
        super().__init__(
            n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=lbound, xu=ubound
        )
        self.target = target_value
        self.decoder = decoder

    @abstractmethod
    def evaluate_mols(self, mols: List[str]) -> np.ndarray:
        """Evaluate a list of molecules based on a target value.

        Implement the evaluation logic specific to this problem.

        Args:
            mols: A list of molecules in string representation.

        Returns:
            An array of fitness values for the input molecules.

        """
        pass

    @abstractmethod
    def _evaluate(
        self, X: np.ndarray, out: Dict[str, np.ndarray], *args: Any, **kwargs: Any
    ) -> None:
        """From pymoo."""
        pass

    def decode_population(self, population: np.ndarray) -> List[str]:
        """Decode a population of solutions to their molecular representations.

        Args:
            population: The population of solutions to be decoded.

        Returns:
            A list of decoded molecular representations in string representation.

        """
        sols = torch.from_numpy(population)
        mols = self.decoder.decode(sols.type(torch.float32))
        return mols


class CompositeProblem(Problem):
    """Represents a composite problem comprising multiple MolecularProblems."""

    def __init__(
        self,
        problems: List[MolecularProblem],
        n_var: int,
        lbound: float,
        ubound: float,
    ):
        self.n_obj = len(problems)
        super().__init__(n_var=n_var, n_obj=self.n_obj, xl=lbound, xu=ubound)
        self.problems = problems

    def _evaluate(
        self, X: np.ndarray, out: Dict[str, np.ndarray], *args: Any, **kwargs: Any
    ) -> None:
        decoded_molecules = self.problems[0].decode_population(X)
        objs = [problem.evaluate_mols(decoded_molecules) for problem in self.problems]
        out["F"] = np.column_stack(objs)


class ProblemFactory:
    """Factory for creating problem instances."""

    def __init__(self):
        self.problem_map: Dict[str, Type[MolecularProblem]] = {}

    def register_problem(
        self, problem_type: str, problem_class: Type[MolecularProblem]
    ) -> None:
        """Allows for dynamic registration of problem types."""
        if problem_type in self.problem_map:
            raise ValueError(f"Problem type '{problem_type}' is already registered.")
        self.problem_map[problem_type] = problem_class

    def create_problem(
        self,
        problem_identifiers: List[str],
        targets: List[Union[int, float]],
        n_var: int,
        lbound: float,
        ubound: float,
        decoder: DecoderInterface,
    ) -> Union[MolecularProblem, CompositeProblem]:
        """Create a problem instance based on the given problem identifiers."""

        invalids = [pid for pid in problem_identifiers if pid not in self.problem_map]
        if invalids:
            raise ValueError(
                f"Unknown problem identifiers: {', '.join(invalids)}. "
                f"Valid identifiers are: {', '.join(self.problem_map.keys())}"
            )

        if len(problem_identifiers) != len(targets):
            raise ValueError(
                "Length of problem_identifiers and targets must be equal. "
                f"Got {len(problem_identifiers)} and {len(targets)} respectively."
            )
        
        problems = [
            self.problem_map[pid](
                target_value=target,
                n_var=n_var,
                lbound=lbound,
                ubound=ubound,
                decoder=decoder,
            )
            for pid, target in zip(problem_identifiers, targets)
        ]

        if len(problems) == 1:
            return problems[0]
        else:
            return CompositeProblem(
                problems=problems,
                n_var=n_var,
                lbound=lbound,
                ubound=ubound,
            )
