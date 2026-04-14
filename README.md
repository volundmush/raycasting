# Raycasting Assignment

This repo includes a compact recursive ray tracer in `rayTrace.py` with differing modes:

1. Basic recursive ray tracing
2. Reflection
3. Refraction
4. Glossy reflection
5. Multiple viewpoints
6. Soft shadows

## WARNING:
It might be that my algorithm is bad or my 980ti can't handle this too well making 1920x1080 pictures, but generating these images takes a while, so pre-rendered ones have been prepared.

But if you have a bunch of threads to spare, just run ./generateImages.sh

## Quick Start

Default render resolution is **1920x1080**.

Run one mode:

```bash
python rayTrace.py --mode basic --view front
```

Render all features from all views:

```bash
python rayTrace.py --mode all --view all
```

Useful options:

```bash
python rayTrace.py --mode glossy --view left --width 1920 --height 1080 --seed 7
```

Modes:
- `basic`
- `reflection`
- `refraction`
- `glossy`
- `softshadow`
- `all`

Views:
- `front`
- `left`
- `right`
- `top`
- `all`

Output files are named:

`render_<mode>_<view>.png`

---

## Mental Model

Think of the renderer like this:

1. For each pixel, shoot one ray from the camera into the scene.
2. Find the closest object that ray hits.
3. Compute local lighting at that hit point (ambient + diffuse + specular).
4. Optionally spawn new rays for mirror/refraction behavior.
5. Combine all returned colors and write the pixel.

So the image is built from many tiny "what do I see in this direction?" questions.

Kind of like a Cathode Ray Tube image but in reverse?

---

## Core Math

### 1) Ray equation

$$
\mathbf{p}(t) = \mathbf{e} + t\,\mathbf{d}
$$

- $\mathbf{e}$: ray origin (camera or bounced hit point)
- $\mathbf{d}$: ray direction (unit vector)
- $t > 0$: distance along ray

Plain English: start at origin and walk forward in a direction by distance $t$.

### 2) Sphere intersection

Sphere implicit form:

$$
\|\mathbf{p}-\mathbf{c}\|^2 - r^2 = 0
$$

Substitute $\mathbf{p}(t)$ and solve quadratic:

$$
At^2 + Bt + C = 0
$$

$$
t = \frac{-B \pm \sqrt{B^2 - 4AC}}{2A}
$$

Plain English: if discriminant $B^2-4AC$ is negative, ray misses. Otherwise, the smallest positive $t$ is the first visible hit.

### 3) Plane intersection

Plane through point $\mathbf{p}_1$ with normal $\mathbf{n}$:

$$
(\mathbf{p}-\mathbf{p}_1)\cdot\mathbf{n}=0
$$

Ray-plane solve:

$$
t = \frac{(\mathbf{p}_1-\mathbf{e})\cdot\mathbf{n}}{\mathbf{d}\cdot\mathbf{n}}
$$

Plain English: if denominator is near zero, ray is parallel to plane (no useful hit).

### 4) Diffuse + specular shading (Phong style)

Diffuse:

$$
I_d = k_d\,\max(0,\mathbf{n}\cdot\mathbf{l})\,\mathbf{c}
$$

Specular (Blinn-Phong half-vector form):

$$
I_s = k_s\,\max(0,\mathbf{n}\cdot\mathbf{h})^{\alpha}
$$

Total local light:

$$
I_{local} = I_a + \sum_{lights}(I_d + I_s)
$$

Plain English:
- diffuse = "how directly the surface faces the light"
- specular = "highlight sparkle"

### 5) Reflection ray

$$
\mathbf{r} = \mathbf{d} - 2(\mathbf{d}\cdot\mathbf{n})\mathbf{n}
$$

Plain English: bounce direction, just like a mirror.

### 6) Refraction ray (Snell-based)

Using indices of refraction $\eta_i$ and $\eta_t$, compute transmitted direction.

If the term inside the square root goes negative, total internal reflection occurs (no transmission ray).

Plain English: transparent materials bend rays when entering/exiting. Good ol' refraction. Because nothing is perfectly transparent

---

## How Each Assignment Goal Is Implemented

### 1) Basic recursive ray tracing

- The main function traces one ray per pixel.
- If no hit, it returns a sky-like gradient background.
- If hit, it computes local lighting.
- Recursion depth limit prevents infinite bounces.

### 2) Reflection

- Materials have `reflectivity` in $[0,1]$.
- A reflection ray is cast from the hit point.
- Reflected color is blended with local color.

### 3) Refraction

- Materials have `transparency` and `ior`.
- Refracted direction is computed from normal + incoming ray.
- Total internal reflection is handled as a fallback.

### 4) Glossy reflection

- Start with mirror direction.
- Add small random jitter controlled by `roughness`.
- Cast multiple rays and average them.

This creates blurry reflections instead of perfect mirrors.

### 5) Multiple viewpoints

- Camera presets are available (`front`, `left`, `right`, `top`).
- `--view all` renders all camera positions.

### 6) Soft shadows

- Light has a radius and multiple samples.
- Multiple shadow rays are cast to different light sample points.
- Fraction of unblocked samples determines how soft/dark shadow is.
