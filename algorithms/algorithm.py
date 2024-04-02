import logging
from typing import Dict, Type

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.algorithm import Algorithm

__all__ = ["AlgorithmFactory"]


class AlgorithmFactory:
    """Factory class for creating algorithm instances."""

    algorithm_map: Dict[str, Type[Algorithm]] = {
        "GA": GA,
        "NSGA2": NSGA2,
        "NSGA3": NSGA3,
    }

    def register_algorithm(
        self, algorithm_type: str, algorithm_class: Type[Algorithm]
    ) -> None:
        """Allows for dynamic registration of algorithm types."""
        if algorithm_type in self.algorithm_map:
            raise ValueError(
                f"Algorithm type '{algorithm_type}' is already registered."
            )
        self.algorithm_map[algorithm_type] = algorithm_class

    def create_algorithm(self, algorithm_type: str, *args, **kwargs) -> Algorithm:
        """Create an algorithm instance based on the given algorithm type."""
        if algorithm_type not in self.algorithm_map:
            raise ValueError(
                f"Unknown algorithm type '{algorithm_type}'. "
                f"Valid types are: {', '.join(self.algorithm_map.keys())}"
            )

        if algorithm_type in ("NSGA2", "NSGA3") and "selection" in kwargs:
            kwargs.pop("selection", None)
            logging.info("Popping selection from kwargs for NSGA2/NSGA3!")

        return self.algorithm_map[algorithm_type](*args, **kwargs)

    @staticmethod
    def check_algorithm_n_objs(algorithm: Algorithm, n_objs: int) -> None:
        """Check if number of objs match the algorithm type."""
        if n_objs > 1 and isinstance(algorithm, GA):
            raise ValueError(
                f"GA only supports single-objective optimization, but {n_objs} objectives were provided."
            )
