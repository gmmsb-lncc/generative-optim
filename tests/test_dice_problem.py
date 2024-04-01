import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# from problems.tanimoto import TanimotoDissimProblem, TanimotoSimProblem
from problems.dice import DiceDissimProblem, DiceSimProblem


class MockDecoder:
    def decode_population(self, x):
        return x


@pytest.fixture
def setup_molecular_similarity():
    decoder = MockDecoder()

    target_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # target molecule (caffeine)
    problem = DiceSimProblem(
        target_value=target_smiles, n_var=0, lbound=0, ubound=0, decoder=decoder
    )
    return problem


@pytest.fixture
def setup_molecular_dissimilarity():
    decoder = MockDecoder()

    target_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    problem = DiceDissimProblem(
        target_value=target_smiles, n_var=0, lbound=0, ubound=0, decoder=decoder
    )
    return problem


def test_dice_implementation_works():
    mol1 = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine
    mol2 = "CC(=O)Nc1ccccc1"

    mol1_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol1), radius=2)
    mol2_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol2), radius=2)

    dice_score = DiceSimProblem.bulk_dice_similarity(mol1_fp, [mol2_fp])[0]
    dice_sim = DataStructs.FingerprintSimilarity(
        mol1_fp, mol2_fp, metric=DataStructs.DiceSimilarity
    )

    assert dice_score == pytest.approx(dice_sim, 0.01)


def test_evaluate_mols_sim(setup_molecular_similarity):
    problem = setup_molecular_similarity
    test_smiles = [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(=O)Nc1ccccc1",
        "O=C(O)c1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "O=C(NO)[C@@H](Cc1ccc2ccccc2c1)NS(=O)(=O)c1ccc2ccccc2c1",
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


def test_evaluate_mols_dissim(setup_molecular_dissimilarity):
    problem = setup_molecular_dissimilarity
    test_smiles = [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(=O)Nc1ccccc1",
        "O=C(O)c1ccccc1C(=O)O",
    ]
    expected_dissimilarities = [
        1.0,
        np.nan,  # idk
        np.nan,
    ]

    dissimilarities = problem.evaluate_mols(test_smiles)

    assert dissimilarities[0] == pytest.approx(
        1.0, 0.01
    ), "self-dissimilarity of the target molecule should be 1.0"
