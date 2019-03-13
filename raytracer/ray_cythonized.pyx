import math
from typing import Tuple

cimport cython


cdef extern from "math.h":
    double sqrt(double m)


cdef extern from "math.h":
    double sqrt(double m)


cpdef (float, float, float) add_vec((float, float, float) a, (float, float, float) b) :
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


cpdef (float, float, float) sub_vec((float, float, float) a, (float, float, float) b) :
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


cpdef (float, float, float) mult_vec((float, float, float) a, (float, float, float) b) :
    return a[0] * b[0], a[1] * b[1], a[2] * b[2]


cpdef float norm((float, float, float) a):
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


cpdef (float, float, float) normalize((float, float, float) a) :
    norm = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    return a[0] / norm, a[1] / norm, a[2] / norm


cpdef (float, float, float) mult_scalar((float, float, float) a, float scalar) :
    return a[0] * scalar, a[1] * scalar, a[2] * scalar


cpdef (float, float, float) add_scalar((float, float, float) a, float scalar) :
    return a[0] + scalar, a[1] + scalar, a[2] + scalar


cpdef (float, float, float) div_scalar((float, float, float) a, float scalar) :
    return a[0] / scalar, a[1] / scalar, a[2] / scalar


cpdef (float, float, float) cross((float, float, float) a, (float, float, float) b) :
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


cpdef float dot((float, float, float) a, (float, float, float) b) :
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

