"""
Evaluating the performance of the vector class compared to tuples !


Options:

* using __slots__
* switch to NamedTuples
* switch to data classes (!! 3.7 + only !!)
* switch to plain dummy tuples !
* switch to numpy
"""
from .context import raytracer

from raytracer import vector

from collections import namedtuple

from raytracer.vector import Vector3
import numpy as np


def test_vector(benchmark):

    v1 = Vector3(1.0, 2.0, 3.0)
    v2 = Vector3(4.0, 5.0, 6.0)
    res = benchmark(v1.__add__, v2)
    assert res == Vector3(5.0, 7.0, 9.0)


def test_vector_slots(benchmark):

    Vector3.__slots__ = ["x", "y", "z"]
    v1 = Vector3(1.0, 2.0, 3.0)
    v2 = Vector3(4.0, 5.0, 6.0)
    res = benchmark(v1.__add__, v2)
    assert res == Vector3(5.0, 7.0, 9.0)


def test_namedtuple(benchmark):
    # Replacing Vector3 with a named tuple
    # => roughly 30% faster than Vector3 !
    VNamedTuple = namedtuple("VNamedTuple", ["x", "y", "z"])

    def add_vnt(v1, v2):
        return v1.x + v2.x, v1.y + v2.y, v1.z + v2.z

    v1 = VNamedTuple(1, 2, 3)
    v2 = VNamedTuple(4, 5, 6)
    res = benchmark(add_vnt, v1, v2)
    assert res == VNamedTuple(5.0, 7.0, 9.0)


def test_tuple_index(benchmark):
    # Replacing Vector3 with plain tuples
    # => roughly 50% faster than Vector3 !

    def add_tuples(v1, v2):
        return v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]

    v1 = 1.0, 2.0, 3.0
    v2 = 4.0, 5.0, 6.0

    res = benchmark(add_tuples, v1, v2)
    assert res == (5.0, 7.0, 9.0)


def test_tuple_decompose(benchmark):
    # Replacing Vector3 with plain tuples
    # but decomposing the tuple instead of using index access
    def add_tuples(v1, v2):
        a, b, d = v1
        e, f, g = v2
        return a + e, b + f, d + g

    v1 = 1.0, 2.0, 3.0
    v2 = 4.0, 5.0, 6.0

    res = benchmark(add_tuples, v1, v2)
    assert res == (5.0, 7.0, 9.0)


def test_numpy_on_array(benchmark):
    # Using a numpy array, using the + operator
    # a bit slower than Vector3, or in the same range

    v1 = np.array((1.0, 2.0, 3.0))
    v2 = np.array((4.0, 5.0, 6.0))

    res = benchmark(v1.__add__, v2)

    assert np.all(res == (5.0, 7.0, 9.0))


def test_numpy(benchmark):
    # Using a numpy array, using np.add
    # a bit slower than Vector3, or in the same range

    v1 = np.array((1.0, 2.0, 3.0))
    v2 = np.array((4.0, 5.0, 6.0))

    res = benchmark(np.add, v1, v2)

    assert np.all(res == (5.0, 7.0, 9.0))
