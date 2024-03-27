import numpy as np
import pytest

from problems.complexity import ComplexityProblem, calculate_bottchscore_from_smiles


class MockDecoder:
    def decode_population(self, x):
        return x


@pytest.fixture
def setup_complexity_problem():
    decoder = MockDecoder()

    target_value = 0.0
    problem = ComplexityProblem(
        target_value=target_value, n_var=0, lbound=0, ubound=0, decoder=decoder
    )
    return problem


def test_evaluate_mols(setup_complexity_problem):
    problem = setup_complexity_problem
    test_smiles = [
        "O=C([C@H](CC1=CNC=N1)N)O",  # histidine, score = 171.60
        # "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",  # morphine, score = 359.37
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine, score = 246.63
    ]

    expected_fitness = [29446, 60826.3569]
    fitness = problem.evaluate_mols(test_smiles)

    assert np.allclose(fitness, expected_fitness, atol=0.1)


def test_score_calculation():
    smiles = [
        "O=C([C@H](CC1=CNC=N1)N)O",
        # "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",  # don't know why results for morphine are so different from the [webserver](https://forlilab.org/services/bottcher/).
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ]
    expected_score = [171.60, 246.63]

    for sm, score in zip(smiles, expected_score):
        assert calculate_bottchscore_from_smiles(sm) == pytest.approx(score, 0.1)
