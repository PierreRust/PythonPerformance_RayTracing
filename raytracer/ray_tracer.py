#!/usr/bin/env python3

import argparse
import functools
import statistics
from concurrent.futures import ThreadPoolExecutor
import math
import time
from os.path import splitext
from time import sleep

from PIL import Image
from typing import Tuple, Optional, List
from multiprocessing import Pool
import numpy as np

# Type alias for tuple-based vector
Vector3 = Tuple[float, float, float]

from raytracer.vector import (
    normalize,
    norm,
    div_scalar,
    sub_vec,
    add_vec,
    add_scalar,
    mult_vec,
    mult_scalar,
    cross,
    dot,
)


"""
A simple Ray tracer.

All ray-tracing code is contained in this file, only vectors and file format
are implemented externally.

"""

MAX_DEPTH = 3
NUDGE = 0.03
MIN_NODE_SHAPE = 4


def normalize_many(vectors):
    vectors /= np.sqrt(np.einsum("ij,ij->i", vectors, vectors)).reshape(-1, 1)
    return vectors


def dot_many(vectors1, vectors2):
    return np.einsum("ij,ij->i", vectors1, vectors2).reshape(-1, 1)


class Ray:
    """ a Ray is a line"""

    def __init__(self, origin: Vector3, direction: Vector3):
        self.origin = tuple(origin)
        # Make sure to always use a unit vector to be able to compare distances
        # on different rays.
        self.direction = normalize(direction)

    def __repr__(self):
        return f"Ray({self.origin}, {self.direction})"


class LightSource:
    def __init__(self, position: Vector3, power=(1, 1, 1)):
        self.position = tuple(position)
        # Power of the light source, not restricted to [0-1]
        # The value depends on the scale of the scene you are using
        self.power = tuple(power)

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
        self.color = tuple(color) if color is not None else (0, 0, 0)
        self.ka = tuple(ka) if ka is not None else (0.9, 0.9, 0.9)
        self.kd = tuple(kd) if kd is not None else (0.8, 0.8, 0.8)
        self.ks = tuple(ks) if ks is not None else (1.2, 1.2, 1.2)
        self.alpha = alpha

        self.mirror_reflection = (
            tuple(mirror_reflection) if mirror_reflection is not None else None
        )
        self.kr = kr

    def color_at_points(
        self, points, rays_origins, rays_directions, hit_normals, scene, depth
    ):
        # return self.color

        if self.diffuse:
            colors = self.phong_many(points, hit_normals, rays_directions, scene)
        else:
            colors = np.zeros(rays_directions.shape)

        if depth < 0:
            return colors

        if self.mirror_reflection or self.kr:
            angles = dot_many(rays_directions, hit_normals)

        if self.mirror_reflection:
            reflection_colors = self.reflexion_at_many(
                points, rays_directions, hit_normals, angles, scene, depth
            ).reshape(-1, 3)
            reflection_colors = self.mirror_reflection * reflection_colors
            colors += reflection_colors
            pass

        elif self.kr:
            colors += self.refraction_at_many(
                points, rays_directions, hit_normals, angles, scene, depth
            )
            pass

        return colors

    def phong_many(self, points, normals, rays_directions, scene):
        """
        Phong model for many rays at once

        Parameters
        ----------
        points
        normals
        rays_origins
        rays_directions
        scene

        Returns
        -------
        np.array
            an array with a rgb color for each ray.
        """

        # ambient light
        lights_coefs = np.ones(rays_directions.shape) * self.ka * scene.ambient_light

        # For each light source, diffuse and specular reflexion
        for light in scene.light_sources:

            # Direction and distance to light
            lights_dirs = light.position - points
            attenuation = math.pi * dot_many(lights_dirs, lights_dirs)
            normalize_many(lights_dirs)
            lights_powers = light.power / attenuation.reshape(-1, 1)

            # check if there is an object between the light source and the point
            outer_points = points + normals * NUDGE
            hit_distances = scene.intersections_distances(outer_points, lights_dirs)
            hit_distances = functools.reduce(np.minimum, hit_distances)
            hit_mask = (hit_distances == FARAWAY).reshape(-1, 1)

            lights_angles = dot_many(normals, lights_dirs)

            # FIXME : use compress & place to only consider a subset of rays ?

            # Diffuse lightning:
            diffuse_coefs = self.kd * lights_angles * lights_powers
            lights_coefs += np.where(
                (lights_angles > 0 & ~hit_mask).reshape(-1, 1), diffuse_coefs, (0, 0, 0)
            )

            # Specular reflexion lightning
            spec_reflexion_dirs = (normals * 2 * lights_angles) - lights_dirs
            spec_coefs = dot_many(-rays_directions, spec_reflexion_dirs)
            spec = (
                self.ks
                * np.power(spec_coefs, self.alpha).reshape(-1, 1)
                * lights_powers
            )
            lights_coefs += np.where(
                (spec_coefs > 0 & ~hit_mask).reshape(-1, 1), spec, (0, 0, 0)
            )

        return self.color * lights_coefs

    def reflexion_at_many(self, points, rays_directions, normals, angles, scene, depth):
        # Normalized reflexion vectors:
        reflexions_dirs = rays_directions - (normals * 2 * angles)
        normalize_many(reflexions_dirs)
        # Origin for reflexion rays, shifted by NUDGE
        reflexions_origins = points + normals * NUDGE

        reflexion_colors = scene.cast_rays(
            reflexions_origins, reflexions_dirs, depth - 1
        )
        return reflexion_colors

    def refraction_at_many(
        self, points, rays_directions, normals, angles, scene, depth
    ):
        cond = angles > 0
        n1 = np.where(cond, self.kr, 1)
        n2 = np.where(cond, 1, self.kr)
        angles = np.where(cond, angles, -angles)
        n12 = n1 / n2

        # Refraction + Reflexion
        # Assume we are moving from air (n= 1) to another material with nt
        # Ratio of reflected light, use Fresnel and Schilck approximation
        r0 = np.power((n2 - 1) / (n2 + 1), 2)
        r = r0 + (1 - r0) * np.power(1 - angles, 5)

        # Reflexion directions, normalized
        reflexions_dirs = rays_directions - (normals * 2 * angles)
        normalize_many(reflexions_dirs)
        # Shift the origin by NUDGE
        reflexions_origins = points + normals * NUDGE

        reflexion_colors = scene.cast_rays(
            reflexions_origins, reflexions_dirs, depth - 1
        )

        # Refraction
        # TODO: improve, we could cast only rays where dis > 0
        #  (I doubt the gain would be big)
        dis = 1 - n12 * n12 * (1 - angles * angles)
        refractions_dirs = (rays_directions - normals * angles) * n12 - (
            normals * np.sqrt(dis)
        )
        normalize_many(refractions_dirs)
        refractions_origins = points - normals * NUDGE
        refractions_colors = scene.cast_rays(
            refractions_origins, refractions_dirs, depth - 1
        )

        cond = (dis > 0)
        refractions_colors = np.where(cond, refractions_colors, (0, 0, 0))
        r = np.where(cond, r, (1))

        colors = reflexion_colors * r + refractions_colors * (1 - r)
        return colors


class SceneObject:
    def __init__(self, surface: Surface):
        self.surface = surface

    def intersect(self, ray: Ray) -> Optional[float]:
        raise NotImplementedError()

    def normal_at(self, pt):
        raise NotImplementedError()

    def normal_for_points(self, pts):
        raise NotImplementedError()

    def middle_point(self):
        raise NotImplementedError()

    def color_for_ray(self, ray, distance: float, scene: "Scene", depth) -> Vector3:
        point = add_vec(mult_scalar(ray.direction, distance), ray.origin)
        normal = self.normal_at(point)
        # Difference normal vs hitNormal !
        cos_out = dot(normal, ray.direction)
        if cos_out > 0:
            hit_normal = mult_scalar(normal, -1)
        else:
            hit_normal = normal

        return self.surface.color_at(point, ray, hit_normal, scene, depth)

    def color_for_rays(
        self, rays_origins, rays_directions, distances, scene: "Scene", depth
    ) -> Vector3:
        """
        get colors for many rays at once
        """

        points = rays_directions * distances.reshape(-1, 1) + rays_origins

        #
        normals = self.normal_for_points(points)

        # Difference normal vs hitNormal: hit_normal points _outside_.
        # dot product
        cos_out = np.einsum("ij,ij->i", normals, rays_directions)
        # hit_normals = normals * np.where(cos_out > 0, 1, -1).repeat(3).reshape(-1, 3)
        hit_normals = normals * np.where(cos_out > 0, -1, 1).reshape(-1, 1)

        return self.surface.color_at_points(
            points, rays_origins, rays_directions, hit_normals, scene, depth
        )


class Plane(SceneObject):
    def __init__(self, point: Vector3, normal: Vector3, surface: Surface):
        super().__init__(surface)
        self.point = tuple(point)
        self.normal = normalize(normal)

    def intersect(self, ray: Ray) -> Optional[float]:
        # The equation of the plane is (point - pt ) . normal = 0
        # we can replace pt by the ray equation : (point -dir * t - origin) . normal = 0
        # t = (point- origin) . normal) / ((dir) . normal)
        d = dot(ray.direction, self.normal)
        if d == 0:
            return None
        n = dot(sub_vec(self.point, ray.origin), self.normal)
        t = n / d
        if t > 0:
            return t
        return None

    def normal_at(self, pt):
        return self.normal

    def normal_for_points(self, pts):
        return self.normal

    def __repr__(self):
        return f"Plane({self.point}, {self.normal})"


class Sphere(SceneObject):
    def __init__(self, position: Vector3, radius: float, surface):
        super().__init__(surface)
        self.radius = radius
        self.position = tuple(position)

    def intersect(self, ray: Ray) -> Optional[float]:
        # intersection is a quadratic equation at^2 + bt +c = 0 with:
        origin_position = sub_vec(ray.origin, self.position)
        a = dot(ray.direction, ray.direction)
        b = 2 * dot(ray.direction, origin_position)
        c = dot(origin_position, origin_position) - self.radius * self.radius
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
                if t1 < t2 or t2 < 0:
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

    def intersect_rays(self, rays_origins, ray_directions):
        """
        Compute the hit distances of a set of ray on that object.

        Parameters
        ----------
        rays_origins: np array of 3d vectors
            origins of the rays
        ray_directions: np array of 3d vectors
            directions of the rays

        Returns
        -------
        np.array
            an array of distances (float), where distance is FARAWAY if a ray does not
            hit that object.
        """
        # intersection is a quadratic equation at^2 + bt +c = 0 with:
        origins_positions = rays_origins - self.position
        # a = np.einsum("ij,ij->i", ray_directions, ray_directions)
        b = 2 * np.einsum("ij,ij->i", ray_directions, origins_positions)
        c = (
            np.einsum("ij,ij->i", origins_positions, origins_positions)
            - self.radius * self.radius
        )
        # b = 2 * dot_many( ray_directions, origins_positions)
        # c = (
        #     dot_many(origins_positions, origins_positions)
        #     - self.radius * self.radius
        # )

        discriminant = b * b - 4 * c  # * a

        sq = np.sqrt(np.maximum(0, discriminant))
        h0 = (-b - sq) / (2)  # * a)
        h1 = (-b + sq) / (2)  # * a)

        # only keep the smallest positive distance, or FARAWAY
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (discriminant > 0) & (h > 0)
        return np.where(pred, h, FARAWAY)

        np.where( )

    def normal_at(self, pt):
        return normalize(sub_vec(pt, self.position))

    def normal_for_points(self, pts):
        return normalize_many(pts - self.position)

    def middle_point(self):
        return self.position

    def __repr__(self):
        return f"Sphere({self.position} , {self.radius}"


FARAWAY = 1.0e39  # an implausibly huge distance


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
            (0.6, 0.6, 0.6) if ambient_light is None else tuple(ambient_light)
        )
        self.background = (0, 0, 0) if background is None else tuple(background)
        self.kdtree = None

    def set_intersect_mode(self, mode):
        self.mode = mode
        if mode.startswith("kdtree"):
            self.kdtree = build_kdtree(self.objects)

    def find_intersect(
        self, ray: Ray, exclude=None
    ) -> Tuple[float, Optional[SceneObject]]:

        if not self.kdtree:
            # basic O(n) linear intersection implementation
            return self.linear_intersect(ray, exclude)
        elif self.mode == "kdtree":
            # Recursive KD tree intersect
            return self.kdtree.intersect(ray, exclude)
        elif self.mode == "kdtree_iter":
            # Recursive KD tree intersect
            return kd_intersect(self.kdtree, ray, exclude)

    def linear_intersect(self, ray, exclude):
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

    def intersections_distances(self, rays_origins, rays_directions):
        # check the intersection of many rays with a single object
        # return an array of array of distance : for each object,
        # an array containing the distance for each ray
        distances = [
            obj.intersect_rays(rays_origins, rays_directions) for obj in self.objects
        ]
        return distances

    def cast_rays(self, rays_origins, rays_directions, depth=MAX_DEPTH):
        """
        Cast many rays at once on the scene.

        Parameters
        ----------
        rays_origins: np.array of 3d vectors
            origins of rays
        rays_directions: np.array of 3d vectors
            a numpy array (ray_count, 3) containing the directions of the rays
        depth:
            recursive call count

        Returns
        -------
        numpy array:
            an array of colors, one for each ray casted, with shape (ray_count, 3)
        """

        # distances is an array that contains, for each object,
        # an np.array of distance for the intersection for each ray
        distances = self.intersections_distances(rays_origins, rays_directions)

        # reduce take the minimum of the distances : for each ray it's the distance
        # to the nearest object
        nearest = functools.reduce(np.minimum, distances)

        # buffer = np.zeros(rays_directions.shape)
        buffer = (
            np.tile(self.background, len(rays_directions))
            .astype("float64")
            .reshape(-1, 3)
        )

        # Analyse rays for each scene object `obj`
        for d_rays, obj in zip(distances, self.objects):

            # hit is a vector mask : an array containing True for the rays
            # that hits this object the nearest:
            hit = (nearest != FARAWAY) & (d_rays == nearest)

            if np.any(hit):
                # Get the hit distance for each of the ray => act as a mask
                # reduces the number of ray we look at
                hit_distances = np.extract(hit, d_rays)

                # Direction and origin of the hitting rays
                hit_rays_directions = np.compress(hit, rays_directions, axis=0)
                hit_rays_origins = np.compress(hit, rays_origins, axis=0)

                # Color for each ray hitting `obj`
                colors = obj.color_for_rays(
                    hit_rays_origins, hit_rays_directions, hit_distances, self, depth
                )

                # Place the color correctly !!
                np.place(buffer, hit.repeat(3), colors)

        return buffer


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

    def draw_pixels(self, colors):
        raise NotImplementedError()

    def reveal(self):
        raise NotImplementedError()


class PngScreen(Screen):
    def __init__(self, filename, screen_width: int, screen_height: int, mode=None):
        super().__init__(screen_width, screen_height)
        self.filename = filename
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.mode = mode
        # buffer is a grid  or [r, g, b] arrays
        self.buffer = [
            [[0 for _ in range(3)] for _ in range(width)] for _ in range(height)
        ]
        self.flat_buffer = None

    def draw_pixel(self, row: int, col: int, color: Vector3):
        r, g, b = color
        color_array = [
            min(255, max(0, int(r))),
            min(255, max(0, int(g))),
            min(255, max(0, int(b))),
        ]
        self.buffer[self.screen_height - row - 1][col] = color_array

        if self.mode == "threads-io":
            # simulate an io operation that would block the thread for 1 ms
            sleep(0.001)

    def draw_pixels(self, colors):
        self.flat_buffer = colors

    def reveal(self):
        print(f"Write image to disk: {self.filename}")
        if self.flat_buffer is None:
            self.flat_buffer = [
                c for row in self.buffer for color in row for c in color
            ]
            img = Image.frombytes(
                "RGB", (self.screen_width, self.screen_height), bytes(self.flat_buffer)
            )
        else:

            bufforig = self.flat_buffer
            bufforig2 = np.clip(self.flat_buffer, 0, 255)
            bufforig2_int = np.clip(self.flat_buffer, 0, 255).astype("uint8")

            buff = self.flat_buffer.astype("uint8")
            buff = bufforig2_int.reshape((self.screen_height, self.screen_width, 3))
            img = Image.fromarray(buff, mode="RGB")

        img.show()
        img.save(self.filename)


def cast_ray_for_pixel(camera, scene, direct_draw, pixel):
    # Function that computes the color for a pixel,
    # and optionally, draw it on the screen.
    row, col = pixel
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
        self.position = tuple(position)
        self.direction = tuple(direction)  # In which we are looking !
        self.up = tuple(up)  # viewing orientation
        self.field_of_view = field_of_view  # angle,
        self.screen_distance = screen_distance

        # Compute basis vector at camera position:
        # need : position, coi and v_up
        self.n = normalize(mult_scalar(self.direction, -1))
        self.u = normalize(cross(self.up, self.n))
        self.v = cross(self.n, self.u)
        self.screen_3d_width, self.screen_3d_height = 0, 0

    def set_screen(self, screen: Screen):
        self.screen = screen

        # Compute Width and Height of the view, in 3D space:
        self.screen_3d_width = math.tan(self.field_of_view / 2) * (
            2 * self.screen_distance
        )
        self.screen_3d_height = self.screen_3d_width / screen.ratio

        # Compute bottom left point of the view, in 3D space:
        screen_center = sub_vec(
            self.position, mult_scalar(self.n, self.screen_distance)
        )
        self.screen_corner = sub_vec(
            sub_vec(screen_center, mult_scalar(self.u, (self.screen_3d_width / 2))),
            mult_scalar(self.v, (self.screen_3d_height / 2)),
        )

    def take_picture(self, scene: Scene, parallel=None):
        start = time.time()

        if parallel.startswith("threads"):
            # Parallel pixels generation, with threads:
            # We need a task function to pass to the worker threads
            partial_cast = functools.partial(cast_ray_for_pixel, self, scene, True)

            with ThreadPoolExecutor() as executor:
                for r, c in self.screen.pixels():
                    executor.submit(partial_cast, (r, c))
        elif parallel.startswith("process"):
            # multiprocess generation
            with Pool() as pool:

                partial_cast = functools.partial(cast_ray_for_pixel, self, scene, False)

                colors = pool.map(
                    partial_cast,
                    [(r, c) for r, c in self.screen.pixels()],
                    chunksize=int(self.screen.height * self.screen.width / 4),
                )
            for (r, c), color in zip(self.screen.pixels(), colors):
                self.screen.draw_pixel(r, c, color)
        elif parallel.startswith("matrix"):
            rays_directions = self.generate_rays_vector()
            rays_positions = np.tile(self.position, (len(rays_directions), 1))
            colors = scene.cast_rays(rays_positions, rays_directions)
            self.screen.draw_pixels(colors)

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

    def generate_rays_vector(self):
        # generate all rays with numpy
        # all rays have the same origin : self.position
        position = np.fromiter(self.position, float)
        corner = np.fromiter(self.screen_corner, float)

        # Scale base vector according to pixel's size
        u = np.fromiter(self.u, float) * (self.screen_3d_width / self.screen.width)
        v = np.fromiter(self.v, float) * (self.screen_3d_height / self.screen.height)
        unit_vect = np.stack([v, u])

        # All screen points as the cartesian product of rows and columns:
        rows = np.arange(self.screen.height)
        cols = np.arange(self.screen.width)
        screen_pts = np.transpose(
            [np.repeat(rows, self.screen.width), np.tile(cols, self.screen.height)]
        )

        # Position of screen pixels in scene coordinates:
        pixel_pos = corner + np.matmul(screen_pts, unit_vect)

        # Rays directions, with normalization:
        rays_dirs = pixel_pos - position
        normalize_many(rays_dirs)
        return rays_dirs


class BoundingBox:
    # shape for our bounding box: sphere
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius
        self.sphere = Sphere(position, radius, None)

    def intersect(self, ray):
        if self.radius == 0:
            return False
        return self.sphere.intersect(ray) is not None


class KDNode:
    # a KD node
    def __init__(self, shapes, node_left=None, node_right=None):
        self.shapes = shapes
        self.left = node_left
        self.right = node_right
        self.bbox = build_bounding_box(shapes)

    def intersect(self, ray, exclude=None) -> Tuple[float, Optional[SceneObject]]:
        if self.bbox.intersect(ray):

            if self.left is not None or self.right is not None:

                d_left, o_left = (
                    self.left.intersect(ray, exclude)
                    if self.left.shapes
                    else (float("inf"), None)
                )
                d_right, o_right = (
                    self.right.intersect(ray, exclude)
                    if self.right.shapes
                    else (float("inf"), None)
                )

                if d_left < d_right:
                    return d_left, o_left
                if d_right < d_left:
                    return d_right, o_right

            else:
                # We are on a leaf:
                min_d, min_shape = float("inf"), None
                for shape in self.shapes:
                    if exclude is not None and shape in exclude:
                        continue
                    d_intersec = shape.intersect(ray)
                    if d_intersec is not None and d_intersec < min_d:
                        min_d = d_intersec
                        min_shape = shape
                return min_d, min_shape

        return float("inf"), None


def kd_intersect(
    node: KDNode, ray: Ray, exclude=None
) -> Tuple[float, Optional[SceneObject]]:

    # Use a DFS traversal of the tree instead of the recursive version
    open = [node]
    min_d, min_shape = float("inf"), None

    while open:
        node = open.pop()
        if node.bbox.intersect(ray):
            leaf = True
            if node.left is not None:
                open.append(node.left)
                leaf = False
            if node.right is not None:
                open.append(node.right)
                leaf = False
            if leaf:
                for shape in node.shapes:
                    if exclude is not None and shape in exclude:
                        continue
                    d_intersect = shape.intersect(ray)
                    if d_intersect is not None and d_intersect < min_d:
                        min_d = d_intersect
                        min_shape = shape

    return min_d, min_shape


def build_bounding_box(shapes: List[Sphere]) -> BoundingBox:
    if not shapes:
        return BoundingBox((0, 0, 0), 0)

    # start with first shape and expand for all other shapes
    position = shapes[0].position
    radius = shapes[0].radius

    for shape in shapes[1:]:
        # expand the bounding box
        position_distance = norm(sub_vec(shape.position, position))
        if position_distance + shape.radius <= radius:
            # The shape is already fully contained in the current bbox.
            continue
        if position_distance + radius <= shape.radius:
            # The bounding box is fully contained in shape,
            # use the shape as the new bounding box.
            position, radius = shape.position, shape.radius
            continue
        # Smallest sphere containing the shape and the box:
        new_radius = (radius + shape.radius + position_distance) / 2
        position = add_vec(
            position,
            mult_scalar(
                normalize(sub_vec(shape.position, position)), (new_radius - radius)
            ),
        )
        radius = new_radius

    return BoundingBox(position, radius)


def is_left(axis, threshold, pt):
    ptx, pty, ptz = pt
    if axis == 0:
        return threshold < ptx
    if axis == 1:
        return threshold < pty
    if axis == 2:
        return threshold < ptz


def mean_tuples(pts, axis):
    avg = statistics.mean(pt[axis] for pt in pts)
    return avg


def select_axis(middle_points):
    # Axes : x : 0, y: 1, z: 2
    var_x = statistics.variance(p[0] for p in middle_points)
    var_y = statistics.variance(p[1] for p in middle_points)
    var_z = statistics.variance(p[2] for p in middle_points)
    var_max = max(var_x, var_y, var_z)
    if var_x == var_max:
        return 0  # x axis
    if var_y == var_max:
        return 1  # y axis
    if var_z == var_max:
        return 2  # z axis


def build_kdtree(shapes: List[SceneObject]) -> KDNode:
    """
    Build a a kd-tree for scene objects.

    Very simple KD tree implementation, we select the axis with the highest variance
    and split on average coordinate on this axis.

    Parameters
    ----------
    shapes:
        list of SceneObjects

    Returns
    -------
    KDNode:
        the root of our kd tree.
    """

    if len(shapes) <= MIN_NODE_SHAPE:
        # Stop condition : when there are only MIN_NODE_SHAPE shape in a node
        return KDNode(shapes)

    # select axis on the largest variance of objects coordinates
    middle_points = [shape.middle_point() for shape in shapes]
    axis = select_axis(middle_points)

    # split on mean of the middle point of the shapes
    split_threshold = mean_tuples(middle_points, axis)

    # assign shapes to left or right child
    left, right = [], []
    for shape in shapes:
        if is_left(axis, split_threshold, shape.middle_point()):
            left.append(shape)
        else:
            right.append(shape)

    # root is a KDNode that contains all shapes
    return KDNode(shapes, build_kdtree(left), build_kdtree(right))


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
        choices=["sequential", "threads", "threads-io", "process", "matrix"],
        help="Parallel generation mode: sequential or threads",
        default="sequential",
    )
    parser.add_argument(
        "--intersect",
        "-i",
        type=str,
        choices=["linear", "kdtree", "kdtree_iter"],
        help="Mode for intersection computation",
        default="linear",
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

    return (
        args.scene_file,
        args.output,
        img_width,
        img_height,
        args.parallel,
        args.intersect,
    )


if __name__ == "__main__":
    scene_file, output, width, height, parallel, intersect = parse_args()

    from raytracer.sceneparser import parse_scene_from_file

    a_scene, a_camera = parse_scene_from_file(scene_file)
    a_scene.set_intersect_mode(intersect)

    a_screen = PngScreen(output, width, height, parallel)
    a_camera.set_screen(a_screen)

    a_camera.take_picture(a_scene, parallel)
