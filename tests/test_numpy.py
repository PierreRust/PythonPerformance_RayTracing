
import numpy as np


def normalize_many_linalg(vectors):
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def normalize_many_einsum(vectors):
    vectors /= np.sqrt(np.einsum("ij,ij->i", vectors, vectors)).reshape(-1, 1)
    return vectors


def test_normalize():

    a = np.random.normal(size=(10,3))

    normalized_linalg = normalize_many_linalg(a)
    assert np.allclose(np.linalg.norm(normalized_linalg, axis=1), 1.)

    normalized_einsum = normalize_many_einsum(a)
    assert np.allclose(np.linalg.norm(normalized_einsum, axis=1), 1.)


    assert np.allclose(normalized_einsum, normalized_linalg)



def test_linalg(benchmark):

    a = np.random.normal(size=(1000,3))

    res = benchmark(normalize_many_linalg, a)


def test_einsum(benchmark):

    # This shows that einsum is faster than linalg (approx. x2 )
    a = np.random.normal(size=(1000,3))

    res = benchmark(normalize_many_einsum, a)



def test_extract():
    a = np.arange(60).reshape(-1, 3)

    cond = np.arange(10) % 2

    ex1 = np.extract(cond.repeat(3), a).reshape(-1, 3)

    ex2 = np.compress(cond, a, axis=0)

    assert np.all(ex1 == ex2)


def test_perf_extract_compress(benchmark):
    # Compress is faster than extract (3x)
    a = np.arange(60).reshape(-1,3)
    cond = np.arange(10) % 2

    def extract_compress(cond, a):
        return np.compress(cond, a, axis=0)

    benchmark(extract_compress, cond, a)


def test_perf_extract_extract(benchmark):
    a = np.arange(60).reshape(-1,3)
    cond = np.arange(10) % 2

    def extract(cond, a):
        return np.extract(cond.repeat(3), a).reshape(-1, 3)

    benchmark(extract, cond, a)
