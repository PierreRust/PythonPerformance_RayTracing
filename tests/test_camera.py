from .context import raytracer

from raytracer.ray_tracer import Camera, Screen
from raytracer.vector import Vector3

from pytest import approx


def test_compute_camera_origin():

    camera = Camera((0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0))

    assert camera.n == Vector3(-1, 0, 0)
    assert camera.v == Vector3(0, 1, 0)
    assert camera.u == Vector3(0, 0, 1)


def test_screen_size_and_position_do_not_depend_on_resolution():
    camera = Camera(
        Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0), screen_distance=5
    )

    screen1 = Screen(40, 30)
    camera.set_screen(screen1)
    screen1_width = camera.screen_width
    screen1_height = camera.screen_height
    screen1_corner = camera.screen_corner

    screen2 = Screen(400, 300)
    camera.set_screen(screen2)
    screen2_width = camera.screen_width
    screen2_height = camera.screen_height
    screen2_corner = camera.screen_corner

    assert screen1_width == screen2_width
    assert screen1_height == screen2_height
    assert screen1_corner == screen2_corner

    print(f" {camera.screen_width} {camera.screen_height} {camera.screen_corner}")


def test_center_pixel_position():
    camera = Camera(
        Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0), screen_distance=5
    )

    screen1 = Screen(40, 30)
    camera.set_screen(screen1)
    print(f" n :{camera.n} u : {camera.u} v: {camera.v}")

    pixel_0_0 = camera._pixel_pos(0, 0)

    assert pixel_0_0 == camera.screen_corner
    print(f"0,0: {pixel_0_0}")

    pixel_40_0 = camera._pixel_pos(30, 0)
    print(f"40,0: {pixel_40_0}")

    pixel_0_30 = camera._pixel_pos(0, 40)
    print(f"0,30: {pixel_0_30}")

    pixel_40_30 = camera._pixel_pos(30, 40)
    print(f"40,30: {pixel_40_30}")

    # check screen is centered
    assert pixel_0_0.y == approx(-pixel_40_30.y)
    assert pixel_0_0.z == approx(-pixel_40_30.z)

    pixel_center = camera._pixel_pos(15, 20)

    assert pixel_center.as_tuple() == approx(Vector3(5, 0, 0).as_tuple())
