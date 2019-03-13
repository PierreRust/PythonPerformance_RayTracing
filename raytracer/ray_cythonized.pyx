# cython: language_level=3

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

cpdef kd_intersect_cython(
    node, ray, exclude=None
) :

    # Use a DFS traversal of the tree instead of the recursive version
    open = [node]
    min_d, min_shape = float("inf"), None

    while open:
        node = open.pop()
        s = node.bbox.sphere
        i = sphere_intersect_cython(s.position, s.radius, ray.origin, ray.direction)
        if i > 0:
            leaf = True
            if node.left is not None:
                open.append(node.left)
                leaf = False
            if node.right is not None:
                open.append(node.right)
                leaf = False
            if leaf:
                for shape in node.shapes:
                    if exclude is not None and shape in exclude:
                        continue
                    d_intersect = sphere_intersect_cython(shape.position, shape.radius,
                                                          ray.origin, ray.direction)

                    # d_intersect = shape.intersect(ray)
                    if d_intersect > 0 and d_intersect < min_d:
                        min_d = d_intersect
                        min_shape = shape

    return min_d, min_shape

cpdef float sphere_intersect_cython((float, float, float) position, float radius,
                                    (float, float, float) ray_origin,
                                    (float, float, float) ray_direction) :
    # intersection is a quadratic equation at^2 + bt +c = 0 with:
    origin_position = sub_vec(ray_origin, position)
    a = dot(ray_direction, ray_direction)
    b = 2 * dot(ray_direction, origin_position)
    c = (
        dot(origin_position, origin_position)
        - radius * radius
    )
    # Discriminant
    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return -1
    elif discriminant > 0:
        dis_sqrt = math.sqrt(discriminant)
        t1 = (-b + dis_sqrt) / (2 * a)
        t2 = (-b - dis_sqrt) / (2 * a)

        # Negative t means that the intersection is before the origin of the ray,
        # we don't want these:
        if t1 > 0:
            if t1 < t2 or t2 < 0:
                return t1
            elif t2 > 0:
                return t2
        elif t2 > 0:
            return t2
        return -1
    else:
        # Graze sphere : something special to do ?
        # single intersection
        t1 = -b / (2 * a)
        return t1