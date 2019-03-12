from typing import Tuple

import yaml

from raytracer.ray_tracer import Scene, Surface, Sphere, Plane, Camera, LightSource


def parse_scene_from_file(file_path: str):

    with open(file_path, encoding="utf8") as f:
        content = f.read()
        return parse_scene(content)


def parse_scene(scene_str) -> Tuple[Scene, Camera]:
    parsed = yaml.load(scene_str)
    surfaces, objects, lights = {}, {}, {}

    for surface in parsed["surfaces"]:
        surfaces[surface] = Surface(**parsed["surfaces"][surface])

    for obj in parsed["objects"]:
        props = parsed["objects"][obj]
        obj_type = props.pop("type")
        props["surface"] = surfaces[props["surface"]]

        if obj_type == "sphere":
            objects[obj] = Sphere(**props)
        elif obj_type == "plane":
            objects[obj] = Plane(**props)

    for light in parsed["lights"]:
        props = parsed["lights"][light]
        lights[light] = LightSource(**props)

    scene_args = parsed["scene"] if "scene" in parsed else {}
    scene = Scene(
        objects=list(objects.values()),
        light_sources=list(lights.values()),
        **scene_args
    )

    camera = Camera(**parsed["camera"])

    return scene, camera
