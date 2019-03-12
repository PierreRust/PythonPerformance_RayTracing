from .context import raytracer

from raytracer import vector


def test_norm():
    v = (1, 0, 0)
    assert vector.norm(v) == 1

    v = (0, 5, 0)
    assert vector.norm(v) == 5


def test_normalize():
    v = (2, 0, 0)
    assert vector.normalize(v) == (1, 0, 0)


def test_normalize_already_norm():
    v = (1, 0, 0)
    assert vector.normalize(v) == v


def test_cross_product():
    v1 = (1, 0, 0)
    v2 = (0, 1, 0)

    assert vector.cross(v1, v2) == (0, 0, 1)
    assert vector.cross(v2, v1) == (0, 0, -1)


def test_dot_product():
    v1 = (1, 0, 2)
    v2 = (2, 3, 4)

    assert vector.dot(v1, v2) == 10
    assert vector.dot(v2, v1) == 10


def test_scalar_multiplication():
    v1 = (1, 0, 2)

    assert vector.mult_scalar(v1, 3) == (3, 0, 6)


def test_element_wise_multiplication():
    v1 = (1, 0, 2)
    v2 = (2, 3, 4)

    assert vector.mult_vec(v1, v2) == (2, 0, 8)
    assert vector.mult_vec(v2, v1) == (2, 0, 8)


def test_scalar_addition():
    v1 = (1, 6, -1)

    assert vector.add_scalar(v1, 2) == (3, 8, 1)


def test_element_wise_addition():
    v1 = (1, 6, -1)
    v2 = (0, 1, 2)

    assert vector.add_vec(v1, v2) == (1, 7, 1)
    assert vector.add_vec(v2, v1) == (1, 7, 1)


def test_element_wise_subtraction():
    v1 = (1, 6, -1)
    v2 = (0, 1, 2)
    assert vector.sub_vec(v2, v1) == (-1, -5, 3)


def test_scalar_division():
    v1 = (1, 6, -1)
    assert vector.div_scalar(v1, 2) == (0.5, 3, -0.5)
