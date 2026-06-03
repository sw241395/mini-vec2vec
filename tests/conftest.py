import pytest
import numpy


@pytest.fixture(scope="function")
def A():
    return numpy.loadtxt("./tests/data/A.out", delimiter=",")


@pytest.fixture(scope="function")
def B():
    return numpy.loadtxt("./tests/data/B.out", delimiter=",")
