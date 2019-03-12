from .context import raytracer

from raytracer.ray_tracer import Sphere, LightSource, Plane
from raytracer.sceneparser import parse_scene
from raytracer.vector import Vector3

"""
Need :
 scene
  objects
  lights
 camera
 screen distance


"""


def test_scene_one_sphere():
    scene_str = """
surfaces:

  s1:
    color: [0,0,253]
    ka: [0.1, 0.2, 0.3]
    kd: [0.4, 0.5, 0.6]
    ks: [0.7, 0.8, 0.9]

objects:
  sphere1:
    type: sphere
    position: [10, 10, 10]
    radius: 5
    surface: s1

lights: {}
camera:
    position: [0, 0, 0]
    direction: [1,-0.45,0]
    up: [1,1,0]

"""
    scene, _ = parse_scene(scene_str)

    assert len(scene.objects) == 1
    assert isinstance(scene.objects[0], Sphere)
    assert scene.objects[0].radius == 5
    assert scene.objects[0].position == (10, 10, 10)

    assert scene.objects[0].surface.diffuse
    assert not scene.objects[0].surface.mirror_reflection
    assert not scene.objects[0].surface.kr
    assert scene.objects[0].surface.ka == (0.1, 0.2, 0.3)
    assert scene.objects[0].surface.kd == (0.4, 0.5, 0.6)
    assert scene.objects[0].surface.ks == (0.7, 0.8, 0.9)
    assert scene.objects[0].surface.color == (0, 0, 253)


def test_one_plane():
    scene_str = """
surfaces:

  s1:
    color: [0,0,253]

objects:
  ground:
    type: plane
    point: [0, 0, 0]
    normal: [0, 1, 0]
    surface: s1

lights: {}
camera:
    position: [0, 0, 0]
    direction: [1,-0.45,0]
    up: [1,1,0]
"""
    scene, _ = parse_scene(scene_str)
    assert len(scene.objects) == 1
    assert isinstance(scene.objects[0], Plane)
    assert scene.objects[0].point == (0, 0, 0)
    assert scene.objects[0].normal == (0, 1, 0)
    assert scene.objects[0].surface.color == (0, 0, 253)


def test_scene_one_light_source():
    scene_str = """
surfaces: {}

objects: {}

lights:
  light1:
    position: [20,50,30]
    power: [1000, 1000, 1000]
camera:
    position: [0, 0, 0]
    direction: [1,-0.45,0]
    up: [1,1,0]
"""
    scene, _ = parse_scene(scene_str)

    assert len(scene.light_sources) == 1
    assert isinstance(scene.light_sources[0], LightSource)
    assert scene.light_sources[0].position == (20, 50, 30)
    assert scene.light_sources[0].power == (1000, 1000, 1000)


def test_camera():
    scene_str = """
surfaces: {}

objects: {}

lights: {}

camera:
    position: [0, 0, 0]
    direction: [1,-0.45,0]
    up: [1,1,0]
    field_of_view: 2.3
    screen_distance: 11
"""
    _, camera = parse_scene(scene_str)

    assert camera.position == (0, 0, 0)
    assert camera.direction == (1, -0.45, 0)
    assert camera.up == (1, 1, 0)
    assert camera.field_of_view == 2.3
    assert camera.screen_distance == 11
