import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

from hgraph.hiervae import HierVAEDecoder
from problems.mw import MolecularWeight


def test_fitness_func():
    X = np.random.rand(2, 32).astype(np.float64)
    out = {}
    target = 1
    decoder = HierVAEDecoder()
    problem = MolecularWeight(target, 32, -1, 1, decoder)

    smiles = problem.decode_population(X)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fitness = np.square([MolWt(mol) - target for mol in mols])

    problem._evaluate(X, out)
    assert out["F"].shape == (2,)
    assert out["F"].dtype == np.float64
    assert np.allclose(out["F"], fitness)
