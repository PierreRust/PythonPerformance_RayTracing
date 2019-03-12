#!/usr/bin/env python3

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import time
from os.path import splitext
from time import sleep

from PIL import Image
from typing import Tuple, Optional
from multiprocessing import Pool

from raytracer.vector import Vector3


"""
A simple Ray tracer.

All ray-tracing code is contained in this file, only vectors and file format
are implemented externally.

"""

MAX_DEPTH = 3
NUDGE = 0.03


class Ray:
    """ a Ray is a line"""

    def __init__(self, origin: Vector3, direction: Vector3):
        self.origin = origin
        # Make sure to always use a unit vector to be able to compare distances
        # on different rays.
        self.direction = direction.normalize()

    def __repr__(self):
        return f"Ray({self.origin}, {self.direction})"


class LightSource:
    def __init__(self, position, power=Vector3(1, 1, 1)):
        self.position = Vector3(*position)
        # Power of the light source, not restricted to [0-1]
        # The value depends on the scale of the scene you are using
        self.power = Vector3(*power)

    def __repr__(self):
        return f"Light({self.position}, {self.power})"


class Surface:
    """
    Describe an object's surface.

    Supports:
    * Phong lighting model with ambient, diffuse and specular lighting
    * Mirror like reflection
    * Refraction with Snell law, Fresnel equation and Schlick's approximation

    """

    def __init__(
        self,
        diffuse=True,
        color=None,
        ka=None,
        kd=None,
        ks=None,
        alpha: int = 16,
        mirror_reflection=None,
        kr: float = None,
    ):
        """

        Parameters
        ----------

        diffuse: Boolean
            If True, the surface is shaded using the Phong model
        color: Vector3
            Base color of the surface, given as RGB (0-255)
        ka: Vector3
            Ambient reflection light coefficient (Phong model)
        kd: Vector3
            Diffuse reflection coefficient (Phong model)
        ks: Vector3
            Specular reflection coefficient (Phong model)
        alpha: int
            Shininess for specular reflexion (Phong model), large alpha produces
            small specular highlights , i.e. mirror like (=64)

        mirror_reflection: Vector3
            coefficient for reflection, used when `mirror` is `True`.
            One [0-1] coefficient must be given for each RGB component.
            If given, the surface acts as a mirror and reflects light.

        kr: float
            Refractive index, used for computing refraction using Snell's Law.
            We assume the objects are placed in air, which has kr = 1.
            If given, the surface is transparent and lighting is made of refracted and
            reflected light, according to kr.


        """
        self.diffuse = diffuse

        # Surface properties for Phong reflection model
        self.color = Vector3(*color) if color is not None else Vector3(0, 0, 0)
        self.ka = Vector3(*ka) if ka is not None else Vector3(0.9, 0.9, 0.9)
        self.kd = Vector3(*kd) if kd is not None else Vector3(0.8, 0.8, 0.8)
        self.ks = Vector3(*ks) if ks is not None else Vector3(1.2, 1.2, 1.2)
        self.alpha = alpha

        self.mirror_reflection = (
            Vector3(*mirror_reflection) if mirror_reflection is not None else None
        )
        self.kr = kr

    def color_at(self, point, ray, hit_normal, scene, depth):
        # Compute the color for a ray touching this surface,
        # using phong, reflection or transparent (reflexion + refraction + fresnel)

        color = Vector3(0, 0, 0)
        if self.diffuse:
            # Phong model for ambient, diffuse and specular reflexion light
            color += self.phong(point, hit_normal, ray, scene)

        if depth < 0:
            return color

        if self.mirror_reflection:
            # Reflexion only, mirror like
            color += self.mirror_reflection * self.reflexion_at(
                point, ray, hit_normal, scene, depth
            )

            return color

        elif self.kr:
            # Refraction
            color += self.refraction_at(point, ray, hit_normal, scene, depth)
        return color

    def diffuse_lightning(self, normal: Vector3, light_dir: Vector3, light_power):
        dot_p = normal.dot(light_dir)
        if dot_p > 0:
            return (self.kd * dot_p) * light_power
        return Vector3(0, 0, 0)

    def specular_reflexion(
        self, ray: Ray, normal: Vector3, light_dir: Vector3, light_power
    ):
        spec_reflexion_dir = 2 * (light_dir.dot(normal)) * normal - light_dir
        view_dir = ray.direction * -1
        spec_coef = view_dir.dot(spec_reflexion_dir)
        if spec_coef > 0:
            return (self.ks * math.pow(spec_coef, self.alpha)) * light_power
        return Vector3(0, 0, 0)

    def phong(self, point, normal, ray, scene):

        # ambient light
        ambient_coef = self.ka * scene.ambient_light

        # For each light source, diffuse and specular reflexion
        lights_coef = Vector3(0, 0, 0)
        for light in scene.light_sources:

            # Direction and distance to light
            light_segment = light.position - point
            light_dir = light_segment.normalize()
            light_power = light.power / (math.pi * light_segment.dot(light_segment))

            # check if there is an object between the light source and the point
            outer_point = point + normal * NUDGE
            _, obj = scene.find_intersect(Ray(outer_point, light_dir), exclude=[self])
            if obj:
                continue

            # Diffuse lightning:
            lights_coef += self.diffuse_lightning(normal, light_dir, light_power)
            # Specular reflexion lightning
            lights_coef += self.specular_reflexion(ray, normal, light_dir, light_power)

        return self.color * (ambient_coef + lights_coef)

    def reflexion_at(self, point, ray, normal, scene, depth):
        reflexion_dir = ray.direction - 2 * (ray.direction.dot(normal)) * normal
        reflexion_ray = Ray(point + normal * NUDGE, reflexion_dir)
        reflexion_color = scene.cast_ray(reflexion_ray, depth - 1)
        return reflexion_color

    def refraction_at(self, point, ray, normal, scene, depth):
        cos_out = normal.dot(ray.direction)
        if cos_out > 0:
            # getting out of the object: invert refraction coefficients
            n1 = self.kr
            n2 = 1
        else:
            # Entering the object
            n1 = 1
            n2 = self.kr
            cos_out = -cos_out

        n12 = n1 / n2

        # Refraction + Reflexion
        # Assume we are moving from air (n= 1) to another material with nt
        # Ratio of reflected light, use Fresnel and Schilck approximation
        r0 = math.pow((n2 - 1) / (n2 + 1), 2)
        r = r0 + (1 - r0) * math.pow(1 - cos_out, 5)

        # Reflexion
        reflexion_dir = ray.direction - 2 * (ray.direction.dot(normal)) * normal
        reflexion_ray = Ray(point + normal * NUDGE, reflexion_dir)
        reflexion_color = scene.cast_ray(reflexion_ray, depth - 1)

        # Refraction
        refraction_color = Vector3(255, 0, 0)
        dis = 1 - n12 * n12 * (1 - cos_out * cos_out)
        if dis > 0:
            # otherwise, no refraction, all is reflected
            refraction_dir = n12 * (
                ray.direction - normal * cos_out
            ) - normal * math.sqrt(dis)

            refraction_ray = Ray(point - normal * NUDGE, refraction_dir)

            # Cast a refraction (aka transparency) ray:
            refraction_color = scene.cast_ray(refraction_ray, depth - 1)
        else:
            r = 1

        color = r * reflexion_color + (1 - r) * refraction_color

        return color


class SceneObject:
    def __init__(self, surface: Surface):
        self.surface = surface

    def intersect(self, ray: Ray) -> Optional[float]:
        raise NotImplementedError()

    def normal_at(self, pt):
        raise NotImplementedError()

    def color_for_ray(self, ray, distance: float, scene: "Scene", depth) -> Vector3:
        point = distance * ray.direction + ray.origin
        normal = self.normal_at(point)
        # Difference normal vs hitNormal !
        cos_out = normal.dot(ray.direction)
        if cos_out > 0:
            hit_normal = -1 * normal
        else:
            hit_normal = normal

        return self.surface.color_at(point, ray, hit_normal, scene, depth)


class Plane(SceneObject):
    def __init__(self, point: Vector3, normal: Vector3, surface: Surface):
        super().__init__(surface)
        self.point = Vector3(*point)
        self.normal = Vector3(*normal).normalize()

    def intersect(self, ray: Ray) -> Optional[float]:
        # The equation of the plane is (point - pt ) . normal = 0
        # we can replace pt by the ray equation : (point -dir * t - origin) . normal = 0
        # t = (point- origin) . normal) / ((dir) . normal)
        d = ray.direction.dot(self.normal)
        if d == 0:
            return None
        n = (self.point - ray.origin).dot(self.normal)
        t = n / d
        if t > 0:
            return t
        return None

    def normal_at(self, pt):
        return self.normal

    def __repr__(self):
        return f"Plane({self.point}, {self.normal})"


class Sphere(SceneObject):
    def __init__(self, position: Vector3, radius: float, surface):
        super().__init__(surface)
        self.radius = radius
        self.position = Vector3(*position)

    def intersect(self, ray: Ray) -> Optional[float]:
        # intersection is a quadratic equation at^2 + bt +c = 0 with:
        a = ray.direction.dot(ray.direction)
        b = 2 * ray.direction.dot((ray.origin - self.position))
        c = (ray.origin - self.position).dot(
            (ray.origin - self.position)
        ) - self.radius * self.radius
        # Discriminant
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None
        elif discriminant > 0:
            dis_sqrt = math.sqrt(discriminant)
            t1 = (-b + dis_sqrt) / (2 * a)
            t2 = (-b - dis_sqrt) / (2 * a)

            # Negative t means that the intersection is before the origin of the ray,
            # we don't want these:
            if t1 > 0:
                if t1 < t2:
                    return t1
                elif t2 > 0:
                    return t2
            elif t2 > 0:
                return t2
            return None
        else:
            # Graze sphere : something special to do ?
            # single intersection
            t1 = -b / (2 * a)
            return t1

    def normal_at(self, pt):
        return (pt - self.position).normalize()

    def __repr__(self):
        return f"Sphere({self.position} , {self.radius}"


class Scene:
    """
    A scene is a collection of scene objects.
    """

    def __init__(
        self, ambient_light=None, background=None, objects=None, light_sources=None
    ):
        self.objects = [] if objects is None else objects
        self.light_sources = [] if light_sources is None else light_sources
        self.ambient_light = (
            Vector3(0.6, 0.6, 0.6) if ambient_light is None else Vector3(*ambient_light)
        )
        self.background = (
            Vector3(0, 0, 0) if background is None else Vector3(*background)
        )

    def find_intersect(
        self, ray: Ray, exclude=None
    ) -> Tuple[float, Optional[SceneObject]]:
        intersections = []
        for obj in self.objects:
            if exclude is not None and obj in exclude:
                continue
            obj_intersec = obj.intersect(ray)
            if obj_intersec:
                intersections.append((obj_intersec, obj))

        if intersections:
            intersections.sort(key=lambda t: t[0])
            return intersections[0]
        else:
            return float("inf"), None

    # def find_intersection(self, ray) :

    def cast_ray(self, ray: Ray, depth=MAX_DEPTH) -> Vector3:
        distance, obj = self.find_intersect(ray)
        if obj:
            color = obj.color_for_ray(ray, distance, self, depth)
        else:
            color = self.background
        return color


class Screen:
    def __init__(self, screen_width: int, screen_height: int):
        # in pixel count
        self.width = screen_width
        self.height = screen_height

    @property
    def ratio(self):
        return self.width / self.height

    def pixels(self):
        """ Generates of pixels coordinate for the view """
        for row in range(0, self.height):
            for col in range(0, self.width):
                yield (row, col)

    def draw_pixel(self, row: int, col: int, color: Vector3):
        raise NotImplementedError()

    def reveal(self):
        raise NotImplementedError()


class PngScreen(Screen):
    def __init__(self, filename, screen_width: int, screen_height: int, mode=None):
        super().__init__(screen_width, screen_height)
        self.filename = filename
        self.screen_width = screen_width
        self.screen_height = screen_height
        # buffer is a grid  or [r, g, b] arrays
        self.mode = mode
        self.buffer = [
            [[0 for _ in range(3)] for _ in range(width)] for _ in range(height)
        ]

    def draw_pixel(self, row: int, col: int, color: Vector3):
        r, g, b = color
        color_array = [
            min(255, max(0, int(r))),
            min(255, max(0, int(g))),
            min(255, max(0, int(b))),
        ]
        self.buffer[self.screen_height - row-1][col] = color_array

        if self.mode == "threads-io":
            # simulate an io operation that would block the thread for 1 ms
            sleep(0.001)

    def reveal(self):
        print(f"Write image to disk: {self.filename}")
        flat_buffer = [c for row in self.buffer for color in row for c in color]
        img = Image.frombytes(
            "RGB", (self.screen_width, self.screen_height), bytes(flat_buffer)
        )
        img.show()
        img.save(self.filename)


def cast_ray_for_pixel(args):
    camera, scene, row, col, direct_draw = args
    ray = camera.ray_for_pixel(row, col)
    color = scene.cast_ray(ray)
    if direct_draw:
        camera.screen.draw_pixel(row, col, color)
    return color


class Camera:
    def __init__(
        self,
        position: Vector3,
        direction: Vector3,
        up: Vector3,
        field_of_view=math.pi * 0.4,
        screen_distance=10,
    ):
        self.position = Vector3(*position)
        self.direction = Vector3(*direction)  # In which we are looking !
        self.up = Vector3(*up)  # viewing orientation
        self.field_of_view = field_of_view  # angle,
        self.screen_distance = screen_distance

        # Compute basis vector at camera position:
        # need : position, coi and v_up
        self.n = -1 * self.direction.normalize()
        self.u = (self.up.cross(self.n)).normalize()
        self.v = self.n.cross(self.u)
        self.screen_3d_width, self.screen_3d_height = 0, 0

    def set_screen(self, screen: Screen):
        self.screen = screen

        # Compute Width and Height of the view, in 3D space:
        self.screen_3d_width = math.tan(self.field_of_view / 2) * (
            2 * self.screen_distance
        )
        self.screen_3d_height = self.screen_3d_width / screen.ratio

        # Compute bottom left point of the view, in 3D space:
        screen_center = self.position - (self.n * self.screen_distance)
        self.screen_corner = (
            screen_center
            - (self.u * (self.screen_3d_width / 2))
            - (self.v * (self.screen_3d_height / 2))
        )

    def take_picture(self, scene: Scene, parallel=None):
        start = time.time()

        if parallel.startswith("threads"):
            # Parallel pixels generation, with threads:
            with ThreadPoolExecutor() as executor:
                for r, c in self.screen.pixels():
                    executor.submit(cast_ray_for_pixel, (self, scene, r, c, True))
        elif parallel.startswith("process"):
            # multiprocess generation
            with Pool() as pool:
                colors = pool.map(
                    cast_ray_for_pixel,
                    [(self, scene, r, c, False) for r, c in self.screen.pixels()],
                    chunksize=int(self.screen.height * self.screen.width / 4),
                )
            for (r, c), color in zip(self.screen.pixels(), colors):
                self.screen.draw_pixel(r, c, color)
        else:
            # Sequential generation of pixels:
            for row, col in self.screen.pixels():
                # Get ray for a pixel
                ray = self.ray_for_pixel(row, col)
                # Send the ray on the scene to get the color
                color = scene.cast_ray(ray)
                # Draw that pixel on the screen
                self.screen.draw_pixel(row, col, color)

        end = time.time()
        duration = end - start
        pixel_count = self.screen.width * self.screen.height
        rate = pixel_count / duration
        print(
            f"Image gGenerated in {duration}s - {rate} pixel/s for {pixel_count} pixels"
        )
        self.screen.reveal()

    def pixel_pos(self, row: int, col: int) -> Vector3:
        # the position of a pixel in the 3D space

        return (
            self.screen_corner
            + (col * self.u * self.screen_3d_width / self.screen.width)
            + (row * self.v * self.screen_3d_height / self.screen.height)
        )

    def ray_for_pixel(self, row: int, col: int):
        pixel_pos = self.pixel_pos(row, col)
        return Ray(self.position, pixel_pos - self.position)


def parse_args():
    """
    scene_file: yaml file, scene + camera
    output_file
    size: w*h

    """
    parser = argparse.ArgumentParser(description="Ray Tracer")
    parser.add_argument(
        "scene_file", type=str, help="File containing the scene description un yaml"
    )
    parser.add_argument(
        "--output", type=str, help="name of generated image", default=None
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=str,
        choices=["sequential", "threads", "threads-io", "process"],
        help="Parallel generation mode: sequential or threads",
        default="sequential",
    )
    parser.add_argument(
        "--size",
        type=str,
        help="size, as 'widthxheight', eg 800x600",
        default="800x600",
    )
    args = parser.parse_args()

    if args.output is None:
        file_name, _ = splitext(args.scene_file)
        args.output = f"{file_name}.png"

    img_width, img_height = args.size.split("x")
    img_width, img_height = int(img_width), int(img_height)

    return args.scene_file, args.output, img_width, img_height, args.parallel


if __name__ == "__main__":
    scene_file, output, width, height, parallel = parse_args()

    from raytracer.sceneparser import parse_scene_from_file

    a_scene, a_camera = parse_scene_from_file(scene_file)

    a_screen = PngScreen(output, width, height, parallel)
    a_camera.set_screen(a_screen)

    a_camera.take_picture(a_scene, parallel)
