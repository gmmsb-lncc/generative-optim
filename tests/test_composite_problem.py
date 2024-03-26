import numpy as np

from hgraph.hiervae import HierVAEDecoder
from problems import CompositeProblem, MolecularProblem
from problems.mw import MolecularWeightProblem


def test_composite_problem_initialization():

    class CustomProblem(MolecularProblem):
        def __init__(self, target_value, n_var, lbound, ubound, decoder):
            super().__init__(target_value, n_var, lbound, ubound, decoder)

        def _evaluate(self, x, out, *args, **kwargs):
            pass

        def evaluate_mols(self, mols):
            pass

    sub_problems = [
        CustomProblem(target_value=i, n_var=10, lbound=0, ubound=1, decoder=None)
        for i in range(2)
    ]

    composite = CompositeProblem(problems=sub_problems, n_var=10, lbound=0, ubound=1)

    assert len(composite.problems) == 2
    assert composite.n_obj == 2


def test_composite_problem_evaluation():
    decoder = HierVAEDecoder()
    X = np.random.rand(2, 32).astype(np.float64)

    mw1 = MolecularWeightProblem(200, n_var=32, lbound=0, ubound=1, decoder=decoder)
    mw2 = MolecularWeightProblem(target_value=0, n_var=32, lbound=0, ubound=1, decoder=decoder)

    composite = CompositeProblem(problems=[mw1, mw2], n_var=32, lbound=0, ubound=1)

    out = {}
    composite._evaluate(X, out)
