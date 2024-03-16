import argparse
from dataclasses import dataclass

from problems import (
    MolecularProblem,
    MolecularWeight,
    QEDProblem,
    SAScore,
    TanimotoSimilarity,
)

__all__ = ["ProblemConfig", "define_objectives"]


@dataclass
class ProblemConfig:
    problem: MolecularProblem
    target_value: float


def define_objectives():
    """Configure the objectives for the optimization problem here."""
    problems = {
        "MW": ProblemConfig(MolecularWeight, 800),
        "QED": ProblemConfig(QEDProblem, 1.0),
        "SA": ProblemConfig(SAScore, 1.0),
        "SIM_caffeine": ProblemConfig(
            TanimotoSimilarity, "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        ),
    }
    return problems
