"""Generate inputs for the aggregation tree analysis."""

import random
from dataclasses import dataclass

import numpy as np
import torch

from hgraph.hiervae import HierVAEDecoder
from optim import configure_problem


@dataclass
class Args:
    objs_file = "objectives.conf.json"
    num_vars = 32
    lbound = -2.5
    ubound = 2.5


problem = configure_problem(Args())


vecs = np.random.randn(250, 32)
out = {}
problem._evaluate(vecs, out)

print(out["F"].shape)
with open("random-eval.dat", "w") as f:
    for sol in out["F"]:
        f.write(f", ".join(sol.astype(str).tolist()) + "\n")
