from .context import raytracer

from raytracer import vector
from raytracer.vector import Vector3


def test_norm():
    v = Vector3(1, 0, 0)
    assert v.norm() == 1

    v = Vector3(0, 5, 0)
    assert v.norm() == 5


def test_normalize():
    v = Vector3(2, 0, 0)
    assert v.normalize() == Vector3(1, 0, 0)


def test_normalize_already_norm():
    v = Vector3(1, 0, 0)
    assert v.normalize() == v


def test_cross_product():
    v1 = Vector3(1, 0, 0)
    v2 = Vector3(0, 1, 0)

    assert v1.cross(v2) == Vector3(0, 0, 1)
    assert v2.cross(v1) == Vector3(0, 0, -1)


def test_dot_product():
    v1 = Vector3(1, 0, 2)
    v2 = Vector3(2, 3, 4)

    assert v1.dot(v2) == 10
    assert v2.dot(v1) == 10


def test_scalar_multiplication():
    v1 = Vector3(1, 0, 2)

    assert v1 * 3 == Vector3(3, 0, 6)
    assert 3 * v1 == Vector3(3, 0, 6)


def test_element_wise_multiplication():
    v1 = Vector3(1, 0, 2)
    v2 = Vector3(2, 3, 4)

    assert v1 * v2 == Vector3(2, 0, 8)
    assert v2 * v1 == Vector3(2, 0, 8)


def test_scalar_addition():
    v1 = Vector3(1, 6, -1)

    obtained = v1 + 2
    assert obtained == Vector3(3, 8, 1)

    obtained = 2 + v1
    assert obtained == Vector3(3, 8, 1)


def test_element_wise_addition():
    v1 = Vector3(1, 6, -1)
    v2 = Vector3(0, 1, 2)

    obtained = v1 + v2
    assert obtained == Vector3(1, 7, 1)

    obtained = v2 + v1
    assert obtained == Vector3(1, 7, 1)


def test_scalar_subtraction():
    v1 = Vector3(1, 6, -1)
    obtained = v1 - 1
    assert obtained == Vector3(0, 5, -2)

    obtained = 1 - v1
    assert obtained == Vector3(0, -5, 2)


def test_element_wise_subtraction():
    v1 = Vector3(1, 6, -1)
    v2 = Vector3(0, 1, 2)
    obtained = v2 - v1
    assert obtained == Vector3(-1, -5, 3)


def test_scalar_division():
    v1 = Vector3(1, 6, -1)
    obtained = v1 / 2
    assert obtained == Vector3(0.5, 3, -0.5)


def test_element_wise_division():
    v1 = Vector3(1, 6, -1)
    v2 = Vector3(1, 2, -0.5)

    obtained = v1 / v2
    assert obtained == Vector3(1, 3, 2)


def test_copy_create():
    v1 = Vector3(1, 6, -1)
    v2 = Vector3(*v1)
    assert id(v1) != id(v2)
    assert v1 == v2
