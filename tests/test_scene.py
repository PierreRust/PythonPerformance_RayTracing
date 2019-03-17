from .context import raytracer
from pytest import approx

from raytracer.ray_tracer import Sphere, Scene, Ray, Surface

RED = (200, 0, 0)
BLACK = (0, 0, 0)


def test_intersect_single_sphere():
    surface = Surface(color=RED)
    s = Sphere(position=(10, 0, 0), radius=5, surface=surface)

    scene = Scene(objects=[s], background=BLACK)

    ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
    obtained = scene.find_intersect(ray)

    assert obtained == (5, s)


def test_cast_on_single_sphere():

    surface = Surface(color=RED)
    s = Sphere(position=(10, 0, 0), radius=5, surface=surface)

    scene = Scene(objects=[s], background=BLACK)

    ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
    obtained = scene.cast_ray(ray)

    assert obtained[0] != 0
    assert obtained[1] == 0
    assert obtained[2] == 0


def test_cast_miss_single_sphere():

    surface = Surface(color=RED)
    s = Sphere(position=(10, 0, 0), radius=5, surface=surface)

    scene = Scene(objects=[s], background=BLACK)

    # Ray is above the sphere and should miss it
    ray = Ray(origin=(0, 6, 0), direction=(1, 0, 0))
    obtained = scene.cast_ray(ray)

    assert obtained[0] == 0
    assert obtained[1] == 0
    assert obtained[2] == 0


def test_4_spheres():
    surface = Surface(color=RED)
    s1 = Sphere(position=(10, -4, -4), radius=5, surface=surface)
    s2 = Sphere(position=(10, -4, 4), radius=5, surface=surface)
    s3 = Sphere(position=(10, 4, -4), radius=5, surface=surface)
    s4 = Sphere(position=(10, 4, 4), radius=5, surface=surface)

    scene = Scene(objects=[s1, s2, s3, s4], background=BLACK)

    # Center ray, miss all spheres
    ray = Ray(origin=(0, 0, 0), direction=(1, 0, 0))
    d, obtained = scene.find_intersect(ray)
    assert obtained is None

    ray = Ray(origin=(0, 0, 0), direction=(1, 0.2, 0.2))
    d, intersected = scene.find_intersect(ray)
    assert intersected == s4
    assert d == approx(6.967, rel=1e-3)

    ray = Ray(origin=(0, 0, 0), direction=(1, -0.2, 0.2))
    d, intersected = scene.find_intersect(ray)
    assert intersected == s2
    assert d == approx(6.967, rel=1e-3)

    ray = Ray(origin=(0, 0, 0), direction=(1, -0.2, -0.2))
    d, intersected = scene.find_intersect(ray)
    assert intersected == s1
    assert d == approx(6.967, rel=1e-3)

    ray = Ray(origin=(0, 0, 0), direction=(1, 0.2, -0.2))
    d, intersected = scene.find_intersect(ray)
    assert intersected == s3
    assert d == approx(6.967, rel=1e-3)
