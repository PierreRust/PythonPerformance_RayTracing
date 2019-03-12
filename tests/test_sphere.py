from .context import raytracer

from raytracer.ray_tracer import Sphere, Ray
from raytracer.vector import Vector3


def test_intersect():

    s = Sphere(Vector3(10, 0, 0), 5, surface=None)

    # Ray is straight to the sphere, we must have two intersection on the x axis:
    intersections = s.intersect(Ray(Vector3(0, 0, 0), Vector3(10, 0, 0)))
    assert intersections is not None
    assert intersections== 5

    # Throw ray on y, while the sphere is on x
    assert not s.intersect(Ray(Vector3(0, 0, 0), Vector3(0, 1, 0)))

    # Throw ray on z, while the sphere is on x
    assert not s.intersect(Ray(Vector3(0, 0, 0), Vector3(0, 0, 1)))

    # More subtle
    assert s.intersect(Ray(Vector3(0, 0, 0), Vector3(1, 0.2, 0.1)))
