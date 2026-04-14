"""Simple recursive ray tracer for the assignment goals.

Implements:
- Basic recursive ray tracing
- Reflection
- Refraction
- Glossy reflection
- Soft shadows
- Multiple viewpoints

Usage examples:
    python rayTrace.py
    python rayTrace.py --mode reflection --view front
    python rayTrace.py --mode glossy --view left --width 400 --height 300
    python rayTrace.py --mode all --view all --seed 7
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


EPSILON = 1e-4
INF = float("inf")


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < EPSILON:
        return v
    return v / n


def clamp01(v: np.ndarray) -> np.ndarray:
    return np.clip(v, 0.0, 1.0)


def reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return normalize(direction - 2.0 * np.dot(direction, normal) * normal)


def refract(direction: np.ndarray, normal: np.ndarray, ior: float) -> Optional[np.ndarray]:
    cosi = np.clip(np.dot(direction, normal), -1.0, 1.0)
    etai = 1.0
    etat = ior
    n = normal.copy()
    if cosi < 0.0:
        cosi = -cosi
    else:
        etai, etat = etat, etai
        n = -normal
    eta = etai / etat
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    if k < 0.0:
        return None
    return normalize(eta * direction + (eta * cosi - math.sqrt(k)) * n)


def random_in_unit_sphere(rng: np.random.Generator) -> np.ndarray:
    while True:
        p = rng.uniform(-1.0, 1.0, 3)
        if np.dot(p, p) <= 1.0:
            return p


@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray

    def point_at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction


@dataclass
class Material:
    color: np.ndarray
    ambient: float = 0.12
    diffuse: float = 0.75
    specular: float = 0.25
    shininess: float = 32.0
    reflectivity: float = 0.0
    transparency: float = 0.0
    ior: float = 1.5
    roughness: float = 0.0


@dataclass
class Hit:
    t: float
    point: np.ndarray
    normal: np.ndarray
    material: Material


class SceneObject:
    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Optional[Hit]:
        raise NotImplementedError


class Sphere(SceneObject):
    def __init__(self, center: np.ndarray, radius: float, material: Material):
        self.center = center
        self.radius = radius
        self.material = material

    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Optional[Hit]:
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None

        sqrtd = math.sqrt(disc)
        root = (-b - sqrtd) / (2.0 * a)
        if root < t_min or root > t_max:
            root = (-b + sqrtd) / (2.0 * a)
            if root < t_min or root > t_max:
                return None

        p = ray.point_at(root)
        n = normalize(p - self.center)
        return Hit(root, p, n, self.material)


class Plane(SceneObject):
    def __init__(self, point: np.ndarray, normal: np.ndarray, material: Material):
        self.point = point
        self.normal = normalize(normal)
        self.material = material

    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Optional[Hit]:
        denom = np.dot(ray.direction, self.normal)
        if abs(denom) < EPSILON:
            return None

        t = np.dot(self.point - ray.origin, self.normal) / denom
        if t < t_min or t > t_max:
            return None

        p = ray.point_at(t)
        n = self.normal if np.dot(ray.direction, self.normal) < 0 else -self.normal
        return Hit(t, p, n, self.material)


@dataclass
class Light:
    position: np.ndarray
    color: np.ndarray
    intensity: float = 1.0
    radius: float = 0.0
    samples: int = 1

    def sample_positions(self, rng: np.random.Generator, soft_shadows: bool) -> List[np.ndarray]:
        if not soft_shadows or self.radius <= 0.0 or self.samples <= 1:
            return [self.position]

        pts = []
        for _ in range(self.samples):
            angle = rng.uniform(0.0, 2.0 * math.pi)
            r = self.radius * math.sqrt(rng.uniform(0.0, 1.0))
            offset = np.array([r * math.cos(angle), 0.0, r * math.sin(angle)])
            pts.append(self.position + offset)
        return pts


class Camera:
    def __init__(
        self,
        eye: np.ndarray,
        look_at: np.ndarray,
        up: np.ndarray,
        fov_degrees: float,
        aspect: float,
    ):
        self.eye = eye
        self.fov = fov_degrees
        self.aspect = aspect

        forward = normalize(look_at - eye)
        right = normalize(np.cross(forward, up))
        true_up = normalize(np.cross(right, forward))

        self.forward = forward
        self.right = right
        self.up = true_up

    def generate_ray(self, px: int, py: int, width: int, height: int) -> Ray:
        x = (2.0 * ((px + 0.5) / width) - 1.0) * self.aspect
        y = 1.0 - 2.0 * ((py + 0.5) / height)

        scale = math.tan(math.radians(self.fov * 0.5))
        direction = normalize(self.forward + x * scale * self.right + y * scale * self.up)
        return Ray(self.eye, direction)


@dataclass
class RenderConfig:
    width: int = 1024
    height: int = 768
    max_depth: int = 4
    background_top: np.ndarray = field(default_factory=lambda: np.array([0.70, 0.82, 1.00]))
    background_bottom: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.08, 0.12]))
    enable_reflection: bool = False
    enable_refraction: bool = False
    enable_glossy: bool = False
    enable_soft_shadows: bool = False
    glossy_samples: int = 8


class Scene:
    def __init__(self):
        self.objects: List[SceneObject] = []
        self.lights: List[Light] = []

    def hit(self, ray: Ray, t_min: float = EPSILON, t_max: float = INF) -> Optional[Hit]:
        closest_t = t_max
        closest_hit = None
        for obj in self.objects:
            h = obj.intersect(ray, t_min, closest_t)
            if h is not None and h.t < closest_t:
                closest_t = h.t
                closest_hit = h
        return closest_hit


class Renderer:
    def __init__(self, scene: Scene, config: RenderConfig, rng: np.random.Generator):
        self.scene = scene
        self.cfg = config
        self.rng = rng

    def background(self, direction: np.ndarray) -> np.ndarray:
        t = 0.5 * (direction[1] + 1.0)
        return (1.0 - t) * self.cfg.background_bottom + t * self.cfg.background_top

    def is_in_shadow(self, point: np.ndarray, to_light: np.ndarray, max_t: float) -> bool:
        shadow_ray = Ray(point + EPSILON * normalize(to_light), normalize(to_light))
        blocker = self.scene.hit(shadow_ray, EPSILON, max_t - EPSILON)
        return blocker is not None

    def shade_local(self, hit: Hit, view_dir: np.ndarray) -> np.ndarray:
        m = hit.material
        color = m.ambient * m.color

        for light in self.scene.lights:
            samples = light.sample_positions(self.rng, self.cfg.enable_soft_shadows)
            sample_acc = np.zeros(3)

            for lp in samples:
                to_light = lp - hit.point
                dist = np.linalg.norm(to_light)
                if dist < EPSILON:
                    continue
                ldir = to_light / dist

                if self.is_in_shadow(hit.point + hit.normal * EPSILON, to_light, dist):
                    continue

                ndotl = max(0.0, np.dot(hit.normal, ldir))
                diffuse = m.diffuse * ndotl * m.color

                half_vec = normalize(ldir + view_dir)
                spec_term = max(0.0, np.dot(hit.normal, half_vec)) ** m.shininess
                specular = m.specular * spec_term * np.ones(3)

                sample_acc += (diffuse + specular) * light.color * light.intensity

            color += sample_acc / max(1, len(samples))

        return color

    def trace(self, ray: Ray, depth: int) -> np.ndarray:
        if depth <= 0:
            return np.zeros(3)

        hit = self.scene.hit(ray)
        if hit is None:
            return self.background(ray.direction)

        view_dir = normalize(-ray.direction)
        local = self.shade_local(hit, view_dir)
        m = hit.material

        refl = np.zeros(3)
        refr = np.zeros(3)

        if self.cfg.enable_reflection and m.reflectivity > 0.0:
            reflect_dir = reflect(ray.direction, hit.normal)
            if self.cfg.enable_glossy and m.roughness > 0.0:
                glossy_sum = np.zeros(3)
                for _ in range(max(1, self.cfg.glossy_samples)):
                    jitter = random_in_unit_sphere(self.rng) * m.roughness
                    d = normalize(reflect_dir + jitter)
                    rr = Ray(hit.point + hit.normal * EPSILON, d)
                    glossy_sum += self.trace(rr, depth - 1)
                refl = glossy_sum / max(1, self.cfg.glossy_samples)
            else:
                rr = Ray(hit.point + hit.normal * EPSILON, reflect_dir)
                refl = self.trace(rr, depth - 1)

        if self.cfg.enable_refraction and m.transparency > 0.0:
            refr_dir = refract(ray.direction, hit.normal, m.ior)
            if refr_dir is not None:
                rr = Ray(hit.point - hit.normal * EPSILON, refr_dir)
                refr = self.trace(rr, depth - 1)
            elif self.cfg.enable_reflection:
                # Total internal reflection fallback.
                reflect_dir = reflect(ray.direction, hit.normal)
                rr = Ray(hit.point + hit.normal * EPSILON, reflect_dir)
                refl += self.trace(rr, depth - 1)

        remaining = max(0.0, 1.0 - m.reflectivity - m.transparency)
        return local * remaining + refl * m.reflectivity + refr * m.transparency

    def render(self, camera: Camera) -> np.ndarray:
        h = self.cfg.height
        w = self.cfg.width
        image = np.zeros((h, w, 3), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                ray = camera.generate_ray(x, y, w, h)
                image[y, x, :] = clamp01(self.trace(ray, self.cfg.max_depth))

        return image


def make_scene() -> Scene:
    scene = Scene()

    red_mirror = Material(
        color=np.array([0.92, 0.22, 0.18]),
        reflectivity=0.35,
        roughness=0.12,
        shininess=64.0,
    )
    green_matte = Material(
        color=np.array([0.20, 0.80, 0.28]),
        reflectivity=0.06,
        shininess=24.0,
    )
    glass = Material(
        color=np.array([0.90, 0.95, 1.00]),
        ambient=0.05,
        diffuse=0.25,
        specular=0.45,
        shininess=100.0,
        reflectivity=0.15,
        transparency=0.70,
        ior=1.50,
    )
    floor_mat = Material(
        color=np.array([0.72, 0.72, 0.76]),
        ambient=0.12,
        diffuse=0.80,
        specular=0.08,
        shininess=12.0,
        reflectivity=0.03,
    )

    scene.objects.append(Sphere(np.array([-1.25, -0.05, -5.6]), 0.95, red_mirror))
    scene.objects.append(Sphere(np.array([1.20, 0.00, -4.35]), 0.90, green_matte))
    scene.objects.append(Sphere(np.array([0.00, 0.90, -6.70]), 0.85, glass))
    scene.objects.append(Plane(np.array([0.0, -1.10, 0.0]), np.array([0.0, 1.0, 0.0]), floor_mat))

    scene.lights.append(
        Light(
            position=np.array([3.8, 5.3, -1.2]),
            color=np.array([1.0, 0.98, 0.95]),
            intensity=1.08,
            radius=0.8,
            samples=24,
        )
    )
    scene.lights.append(
        Light(
            position=np.array([-3.0, 2.0, -2.0]),
            color=np.array([0.58, 0.65, 0.90]),
            intensity=0.35,
            radius=0.0,
            samples=1,
        )
    )
    return scene


def make_camera(view: str, aspect: float) -> Camera:
    if view == "front":
        eye = np.array([0.0, 0.6, 1.8])
        look = np.array([0.0, -0.1, -5.0])
    elif view == "left":
        eye = np.array([-2.4, 0.8, 0.7])
        look = np.array([0.0, 0.0, -5.2])
    elif view == "right":
        eye = np.array([2.6, 0.8, 0.8])
        look = np.array([0.0, 0.0, -5.1])
    elif view == "top":
        eye = np.array([0.0, 3.8, -2.4])
        look = np.array([0.0, -0.3, -5.4])
    else:
        raise ValueError(f"Unsupported view: {view}")

    return Camera(eye, look, np.array([0.0, 1.0, 0.0]), fov_degrees=50.0, aspect=aspect)


def mode_to_config(mode: str, width: int, height: int) -> RenderConfig:
    cfg = RenderConfig(width=width, height=height)
    if mode == "basic":
        return cfg
    if mode == "reflection":
        cfg.enable_reflection = True
        return cfg
    if mode == "refraction":
        cfg.enable_reflection = True
        cfg.enable_refraction = True
        return cfg
    if mode == "glossy":
        cfg.enable_reflection = True
        cfg.enable_glossy = True
        return cfg
    if mode == "softshadow":
        cfg.enable_soft_shadows = True
        return cfg
    if mode == "all":
        cfg.enable_reflection = True
        cfg.enable_refraction = True
        cfg.enable_glossy = True
        cfg.enable_soft_shadows = True
        return cfg
    raise ValueError(f"Unsupported mode: {mode}")


def render_one(mode: str, view: str, width: int, height: int, seed: int) -> str:
    scene = make_scene()
    cfg = mode_to_config(mode, width, height)
    rng = np.random.default_rng(seed)
    renderer = Renderer(scene, cfg, rng)

    aspect = width / float(height)
    cam = make_camera(view, aspect)
    image = renderer.render(cam)

    out = (clamp01(image) * 255.0).astype(np.uint8)
    out_name = f"render_{mode}_{view}.png"
    Image.fromarray(out).save(out_name)
    return out_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment ray tracer")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["basic", "reflection", "refraction", "glossy", "softshadow", "all"],
        help="Rendering mode to run",
    )
    parser.add_argument(
        "--view",
        default="all",
        choices=["front", "left", "right", "top", "all"],
        help="Camera viewpoint",
    )
    parser.add_argument("--width", type=int, default=1280, help="Output width")
    parser.add_argument("--height", type=int, default=720, help="Output height")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = [args.mode] if args.mode != "all" else ["basic", "reflection", "refraction", "glossy", "softshadow", "all"]
    views = [args.view] if args.view != "all" else ["front", "left", "right", "top"]

    outputs: List[Tuple[str, str, str]] = []
    for mode in modes:
        for view in views:
            filename = render_one(mode, view, args.width, args.height, args.seed)
            outputs.append((mode, view, filename))

    for mode, view, filename in outputs:
        print(f"[{mode:10s}] [{view:5s}] -> {filename}")


if __name__ == "__main__":
    main()
