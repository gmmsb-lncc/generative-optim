import pytest

from problems import MolecularProblem, ProblemFactory
from problems.mw import MolecularWeightProblem


def test_problem_factory_creation():
    factory = ProblemFactory()
    factory.problem_map = {"MW": MolecularWeightProblem}

    # test with valid identifiers
    problem = factory.create_problem(
        problem_identifiers=["MW"],
        targets=[5],
        n_var=10,
        lbound=0,
        ubound=1,
        decoder=None,
    )
    assert isinstance(problem, MolecularProblem)

    # test with an invalid identifier
    with pytest.raises(ValueError) as exc_info:
        factory.create_problem(
            problem_identifiers=["Unknown"],
            targets=[5],
            n_var=10,
            lbound=0,
            ubound=1,
            decoder=None,
        )
    assert "Unknown problem identifiers: Unknown" in str(exc_info.value)


def test_problem_registration():
    factory = ProblemFactory()
    factory.register_problem("MW", MolecularWeightProblem)

    # test with valid identifiers
    problem = factory.create_problem(
        problem_identifiers=["MW"],
        targets=[5],
        n_var=10,
        lbound=0,
        ubound=1,
        decoder=None,
    )
    assert isinstance(problem, MolecularProblem)

    # test duplicate registration
    with pytest.raises(ValueError) as exc_info:
        factory.register_problem("MW", MolecularWeightProblem)
