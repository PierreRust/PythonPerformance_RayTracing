from .context import raytracer

import pytest

from raytracer.ray_tracer import (
    Sphere,
    Camera,
    PngScreen,
    Scene,
    LightSource,
    Plane,
    Surface,
)


@pytest.mark.skip
def test_centered_single_sphere():

    scene = Scene()
    light2 = LightSource(Vector3(10, 0, -10))
    surface = Surface(color=Vector3(100, 0, 0))
    sphere1 = Sphere(Vector3(20, 0, 0), 3, surface)
    scene.objects.append(sphere1)
    scene.light_sources.append(light2)

    camera = Camera(
        Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 0, 1), screen_distance=10
    )
    screen = PngScreen("test_centered_single_sphere.png", 400, 400)
    camera.set_screen(screen)

    camera.ray_for_pixel(50, 50)

    camera.take_picture(scene)


@pytest.mark.skip
def test_sphere_position():

    scene = Scene()
    light2 = LightSource(Vector3(0, -20, -20))
    surface = Surface(color=Vector3(100, 0, 0))
    sphere1 = Sphere(Vector3(25, -3, -5), 3, surface)
    scene.objects.append(sphere1)
    scene.light_sources.append(light2)

    camera = Camera(
        Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0), screen_distance=10
    )
    screen = PngScreen("test_sphere_position.png", 600, 400)
    camera.set_screen(screen)

    camera.ray_for_pixel(50, 50)

    camera.take_picture(scene)


@pytest.mark.skip
def test_two_spheres():

    scene = Scene()
    light2 = LightSource(Vector3(0, -20, -10))
    surface = Surface(color=Vector3(100, 0, 0))
    sphere1 = Sphere(Vector3(40, 4, 0), 3, surface)
    sphere2 = Sphere(Vector3(30, -4, 0), 3, surface)

    scene.objects.append(sphere1)
    scene.objects.append(sphere2)
    scene.light_sources.append(light2)

    camera = Camera(Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0))
    screen = PngScreen("test_two_spheres.png", 400, 400)
    camera.set_screen(screen)

    camera.ray_for_pixel(50, 50)

    camera.take_picture(scene)
