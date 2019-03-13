import math
from typing import Tuple



def add_vec(a: Tuple, b: Tuple) -> Tuple:
    ax, ay, az = a
    bx, by, bz = b
    return ax + bx, ay + by, az + bz


def sub_vec(a: Tuple, b: Tuple) -> Tuple:
    ax, ay, az = a
    bx, by, bz = b
    return ax - bx, ay - by, az - bz


def mult_vec(a: Tuple, b: Tuple) -> Tuple:
    ax, ay, az = a
    bx, by, bz = b
    return ax * bx, ay * by, az * bz


def norm(a: Tuple) -> float:
    ax, ay, az = a
    return math.sqrt(ax * ax + ay * ay + az * az)


def normalize(a: Tuple) -> Tuple:
    ax, ay, az = a
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    return ax / norm, ay / norm, az / norm


def mult_scalar(a: Tuple, scalar: float) -> Tuple:
    ax, ay, az = a
    return ax * scalar, ay * scalar, az * scalar


def add_scalar(a: Tuple, scalar: float) -> Tuple:
    ax, ay, az = a
    return ax + scalar, ay + scalar, az + scalar


def div_scalar(a: Tuple, scalar: float) -> Tuple:
    ax, ay, az = a
    return ax / scalar, ay / scalar, az / scalar


def cross(a: Tuple, b: Tuple) -> Tuple:
    ax, ay, az = a
    bx, by, bz = b
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def dot(a: Tuple, b: Tuple) -> float:
    ax, ay, az = a
    bx, by, bz = b
    return ax * bx + ay * by + az * bz
