import math
from typing import Union


class Vector3:
    """
    Vector3 represents a 3-dimensional vector.

    Standard vector operations are provided: cross product, dot product, norm.
    All operators a defined for element-wise and scalar operations.
    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def normalize(self) -> "Vector3":
        """
        Normalize a vector

        Returns
        -------
        Vector3:
            A new vector, which is the input vector normalized.
        """
        norm = self.norm()
        return Vector3(self.x / norm, self.y / norm, self.z / norm)

    def norm(self) -> float:
        """
        Vector's norm.

        Returns
        -------
        float:
            the norm of the vector (i.e. a scalar)
        """
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def cross(self, other: "Vector3") -> "Vector3":
        """
        Vector cross product.

        Parameters
        ----------
        other: Vector3 of scalar

        Returns
        -------
        Vector3:
            The cross product of the two vectors.

        """
        # aka cross product
        nx = self.y * other.z - self.z * other.y
        ny = self.z * other.x - self.x * other.z
        nz = self.x * other.y - self.y * other.x
        return Vector3(nx, ny, nz)

    def dot(self, other: "Vector3") -> float:
        """
        Vector's dot product.

        Parameters
        ----------
        other: Vector3

        Returns
        -------
        float:
            the dot product of the two vectors.

        """
        # aka dot product, inner product, scalar product
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __mul__(self, other: Union["Vector3", float]) -> "Vector3":
        """
        Multiplication, element-wise and scalar

        Parameters
        ----------
        other: Vector3 of scalar

        Returns
        -------
        a Vector3

        """
        if isinstance(other, Vector3):
            # Element-wise multiplication
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return Vector3(other * self.x, other * self.y, other * self.z)

    def __rmul__(self, other: Union["Vector3", float]) -> "Vector3":
        """
        Multiplication, element-wise and scalar

        Parameters
        ----------
        other: Vector3 of scalar

        Returns
        -------
        a Vector3

        """
        if isinstance(other, Vector3):
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return Vector3(other * self.x, other * self.y, other * self.z)

    def __add__(self, other: Union["Vector3", float]) -> "Vector3":
        """
        Addition, element wise and scalar.

        Parameters
        ----------
        other: a Vector3 or a scalar

        Returns
        -------
        a Vector3

        Examples
        --------

        >>> Vector3(1,2,3) + 1
        Vector3D(2, 3, 4)

        >>> Vector3(1,2,3) + Vector3(-1,0,2)
        Vector3D(0, 2, 5)

        """
        if isinstance(other, Vector3):
            return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            return Vector3(self.x + other, self.y + other, self.z + other)

    def __radd__(self, other: Union["Vector3", float]) -> "Vector3":
        """
        Addition, element wise and scalar.

        Parameters
        ----------
        other: a Vector3 or a scalar

        Returns
        -------
        a Vector3

        Examples
        --------

        >>> 1+ Vector3(1,2,3)
        Vector3D(2, 3, 4)

        >>> Vector3(1,2,3) + Vector3(-1,0,2)
        Vector3D(0, 2, 5)

        """
        if isinstance(other, Vector3):
            return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            return Vector3(self.x + other, self.y + other, self.z + other)

    def __sub__(self, other: Union["Vector3", float]) -> "Vector3":
        """
        Subtraction, element wise and scalar.

        Parameters
        ----------
        other: a Vector3 or a scalar

        Returns
        -------
        a Vector3

        """
        if isinstance(other, Vector3):
            return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return Vector3(self.x - other, self.y - other, self.z - other)

    def __rsub__(self, other: Union["Vector3", float]) -> "Vector3":
        """
        Subtraction, element wise and scalar.

        Parameters
        ----------
        other: a Vector3 or a scalar

        Returns
        -------
        a Vector3

        """
        if isinstance(other, Vector3):
            return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return Vector3(other - self.x, other - self.y, other - self.z)

    def __truediv__(self, other: Union["Vector3", float]) -> "Vector3":
        """
        Division, element wise and scalar.

        Parameters
        ----------
        other: a Vector3 or a scalar

        Returns
        -------
        a Vector3

        """
        if isinstance(other, Vector3):
            return Vector3(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            return Vector3(self.x / other, self.y / other, self.z / other)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vector3):
            return False
        return other.x == self.x and other.y == self.y and self.z == other.z

    def __repr__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def as_tuple(self):
        return self.x, self.y, self.z

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
