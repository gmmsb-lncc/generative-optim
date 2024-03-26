import numpy as np
import pytest

from problems.tanimoto import TanimotoSimilarity


class MockDecoder:
    def decode_population(self, x):
        return x


@pytest.fixture
def setup_molecular_similarity():
    decoder = MockDecoder()

    target_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # target molecule (caffeine)
    problem = TanimotoSimilarity(
        target_value=target_smiles, n_var=0, lbound=0, ubound=0, decoder=decoder
    )
    return problem


def test_evaluate_mols(setup_molecular_similarity):
    problem = setup_molecular_similarity
    test_smiles = [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(=O)Nc1ccccc1",
        "O=C(O)c1ccccc1C(=O)O",
    ]
    expected_similarities = [
        0.0,
        np.nan,  # idk
        np.nan,
    ]

    similarities = problem.evaluate_mols(test_smiles)

    assert similarities[0] == pytest.approx(
        0.0, 0.01
    ), "self-similarity of the target molecule should be 0"
