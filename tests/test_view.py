from .context import raytracer

from raytracer.ray_tracer import Screen


def test_pixels_has_right_len():
    view = Screen(2, 3)
    pixels = list(view.pixels())

    assert len(pixels) == 6


def test_pixels_starts_at_zero():
    view = Screen(2, 3)
    pixels = list(view.pixels())

    assert (0, 0) in pixels
