import numpy as np
import pytest
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA

from algorithms.algorithm import AlgorithmFactory


def test_successful_algorithm_creation():
    factory = AlgorithmFactory()
    nsga2_instance = factory.create_algorithm("NSGA2")
    assert isinstance(nsga2_instance, NSGA2), "NSGA2 instance not created correctly."


def test_dynamic_registration_and_creation():
    factory = AlgorithmFactory()
    factory.register_algorithm("NSGA3_", NSGA3)
    nsga3_instance = factory.create_algorithm("NSGA3_", ref_dirs=np.random.rand(100, 3))
    assert isinstance(nsga3_instance, NSGA3), "NSGA3_ instance not created correctly."


def test_duplicate_registration():
    factory = AlgorithmFactory()
    with pytest.raises(ValueError) as excinfo:
        factory.register_algorithm("NSGA2", NSGA3)
    assert "already registered" in str(
        excinfo.value
    ), "Duplicate registration did not raise ValueError."


def test_invalid_algorithm_type():
    factory = AlgorithmFactory()
    with pytest.raises(ValueError) as excinfo:
        factory.create_algorithm("InvalidType")
    assert "Unknown algorithm type" in str(
        excinfo.value
    ), "Invalid algorithm type did not raise ValueError."


def test_correct_instance_type():
    factory = AlgorithmFactory()
    ga_instance = factory.create_algorithm("GA")
    assert isinstance(ga_instance, GA), "GA instance is not of type GA."
