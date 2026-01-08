import random
from pathlib import Path
from functools import lru_cache
import json
import copy
import math
import gc

import multiprocessing as mp
import mitsuba as mi
import numpy as np
import drjit as dr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


mi.set_variant("cuda_ad_spectral_polarized")
mi.set_log_level(mi.LogLevel.Warn)

ASSET_DIR = Path(r"F:\3DAssets\AdobeStockModels")          # OBJ models
MATERIAL_DIR = Path(r"F:\3DAssets\AdobeStockMaterials_polar")    # PBR textures
OUTPUT_DIR = Path(__file__).resolve().parent / "data_polarized_30000"

NUM_SCENES = 30000
IMAGE_RES = 512
SAMPLES_PER_PIXEL = 128
RENDER_TIMEOUT_SEC = 1800  # safety cutoff per scene (seconds)
seed = 1234
rng = random.Random(seed)

# TARGET_EXTENT_RANGE = (10.0, 11.0)
TARGET_EXTENT = 8.0
TEXTURED_PROB = 1.0

SCENE_CENTER = [0.0, 0.0, 0.0]

ROTATION = True

CAMERA_CFG = {
    "distance_range": (7.5, 8.5),
    "elevation_deg_range": (0.0, 6.0),
    "azimuth_deg_range": (-8.0, 8.0),
    "base_scale": 5.0,
}

LIGHT_CFG = {
    "num_lights": 6,
    "distance_scale": 1.5,
    "cos_theta_min": 0.4,
    "cos_theta_max": 0.7,
}

BLIND_LIGHT_CFG = {
    "enabled": False,
    "irradiance": {"type": "d65", "scale": float(rng.uniform(0.15, 0.8))},
    "require_up": True
}

POLARIZER_CFG = {
    "distance_fraction": 0.01,
    "size_factor": 1.3,
}

ORTHO_FRAME_CFG = {
    "enabled": True,
    "fill_fraction": 0.8,
    "vertical_bias_frac": 0.17,  
    "clamp_scale": (0.25, 6.0),
}


CONDUCTOR_MATERIAL_PRESETS = [
    "a-C", "Ag", "Al", "AlAs", "AlAs_palik", "AlSb", "AlSb_palik", "Au", "Be", "Be_palik",
    "Cr", "CsI", "CsI_palik", "Cu", "Cu_palik", "Cu2O", "Cu2O_palik", "CuO", "CuO_palik",
    "d-C", "d-C_palik", "Hg", "Hg_palik", "HgTe", "HgTe_palik", "Ir", "K", "K_palik", "Li",
    "Li_palik", "MgO", "MgO_palik", "Mo", "Na_palik", "Nb", "Nb_palik", "Ni_palik", "Rh",
    "Rh_palik", "Se", "Se_palik", "SiC", "SiC_palik", "SnTe", "SnTe_palik", "Ta", "Ta_palik",
    "Te", "Te_palik", "ThF4", "ThF4_palik", "TiC", "TiC_palik", "TiN", "TiN_palik", "TiO2",
    "TiO2_palik", "VC", "VC_palik", "V_palik", "VN", "VN_palik", "W"
]

# CBOX_DIR = Path(r"D:\mitsuba3_code\cbox")

# CBOX_FILES = {
#     "floor": CBOX_DIR / "cbox_floor.obj",
#     "ceiling": CBOX_DIR / "cbox_ceiling.obj",
#     "back": CBOX_DIR / "cbox_back.obj",
#     "left": CBOX_DIR / "cbox_leftwall.obj",
#     "right": CBOX_DIR / "cbox_rightwall.obj",
# }

# -------------------------
# Texture 
# -------------------------
def discover_textures(root):
    entries = []
    for bc in root.rglob("*_baseColor.*"):
        stem = bc.stem.replace("_baseColor", "")
        suf = bc.suffix
        dir_ = bc.parent

        def opt(name):
            p = dir_ / f"{stem}_{name}{suf}"
            return p if p.exists() else None

        entries.append({
            "stem": stem,
            "base_color": bc,
            #"height": opt("height"),
            #"normal": opt("normal"),
            #"roughness": opt("roughness")
        })
    return entries

# -------------------------
# Helpers
# -------------------------

def point_emitter(name, origin, rgb, rng):
    scale_range = (0.5, 2.0)
    s = rng.uniform(*scale_range)
    rgb_scaled = tuple(c * s for c in rgb)

    return {
        name: {
            "type": "point",
            "to_world": mi.ScalarTransform4f.translate(origin),
            "intensity": {"type": "rgb", "value": rgb_scaled},
        }
    }

def make_random_roughconductor(rng, tex_pool=None):
    height_scale = rng.uniform(0.01, 0.03)
    a = float(rng.uniform(0.01, 0.2))

    core = {
        "type": "roughconductor",
        "material": rng.choice(CONDUCTOR_MATERIAL_PRESETS),
        "distribution": "ggx",
    }

    use_tex = bool(tex_pool) and rng.random() < TEXTURED_PROB
    if use_tex:
        tex = rng.choice(tex_pool)

        core["specular_reflectance"] = {
            #"type": "bitmap",
            #"filename": str(tex["base_color"]),
            "type": "rgb", 
            "value": [float(rng.uniform(0.8, 1.0)) for _ in range(3)] 
        }

        if tex.get("roughness"):
            core["alpha"] = {"type": "bitmap", "filename": str(tex["roughness"]), "raw": True}
        else:
            core["alpha_u"] = a
            core["alpha_v"] = a

        if tex.get("normal"):
            core = {
                "type": "normalmap",
                "normalmap": {"type": "bitmap", "filename": str(tex["normal"]), "raw": True},
                "bsdf": core,
            }
        if tex.get("height"):
            core = {
                "type": "bumpmap",
                "scale": height_scale,
                "texture": {"type": "bitmap", "filename": str(tex["height"]), "raw": True},
                "bsdf": core,
            }
        return core

    core["specular_reflectance"] = {
            "type": "rgb", 
            "value": [float(rng.uniform(0.8, 1.0)) for _ in range(3)]
    }
    core["alpha_u"] = a
    core["alpha_v"] = a

    return core


def make_random_pplastic(rng, tex_pool=None):
    height_scale = rng.uniform(0.01, 0.03)
    alpha = float(rng.uniform(0.01, 0.3)) 

    core = {
        "type": "pplastic",
        "distribution": "ggx",
        "alpha": alpha,
        "int_ior": rng.uniform(1.45, 1.6),
        "ext_ior": 1.0,
    }

    use_tex = bool(tex_pool) and rng.random() < TEXTURED_PROB
    if use_tex:
        tex = rng.choice(tex_pool)

        core["diffuse_reflectance"] = {
            "type": "bitmap",
            "filename": str(tex["base_color"]),
        }

        if tex.get("normal"):
            core = {
                "type": "normalmap",
                "normalmap": {"type": "bitmap", "filename": str(tex["normal"]), "raw": True},
                "bsdf": core,
            }
        if tex.get("height"):
            core = {
                "type": "bumpmap",
                "scale": height_scale,
                "texture": {"type": "bitmap", "filename": str(tex["height"]), "raw": True},
                "bsdf": core,
            }
        return core

def summarize_material(bsdf):
    def add_tex(textures, role, node):
        if isinstance(node, dict) and node.get("type") == "bitmap":
            textures[role] = node.get("filename")
            return True
        return False

    def _summ(b):
        t = b.get("type")
        textured_local = False
        params = {}
        textures = {}
        mat_type = t

        if t == "roughconductor":
            mat_type = "roughconductor"
            params["ior_preset"] = b.get("material")
            params["distribution"] = b.get("distribution")

            a  = b.get("alpha")
            au = b.get("alpha_u")
            av = b.get("alpha_v")

            if a is not None:
                params["roughness_mode"] = "alpha"
                if isinstance(a, dict):
                    textured_local |= add_tex(textures, "roughness", a)
                    params["alpha_type"] = "bitmap"
                else:
                    params["alpha_type"] = "scalar"
                    params["alpha"] = a
            else:
                params["roughness_mode"] = "alpha_u_v"

                if isinstance(au, dict):
                    textured_local |= add_tex(textures, "alpha_u", au)
                    params["alpha_u_type"] = "bitmap"
                else:
                    params["alpha_u_type"] = "scalar"
                    params["alpha_u"] = au

                if isinstance(av, dict):
                    textured_local |= add_tex(textures, "alpha_v", av)
                    params["alpha_v_type"] = "bitmap"
                else:
                    params["alpha_v_type"] = "scalar"
                    params["alpha_v"] = av

            spec = b.get("specular_reflectance")
            if isinstance(spec, dict):
                st = spec.get("type")
                if st == "bitmap":
                    textured_local |= add_tex(textures, "specular", spec)
                    params["specular_type"] = "bitmap"
                elif st == "rgb":
                    params["specular_type"] = "rgb"
                    params["specular_rgb"] = spec.get("value")

        elif t == "pplastic":
            mat_type = "pplastic"

            diff_ref = b.get("diffuse_reflectance", {})
            if isinstance(diff_ref, dict) and diff_ref.get("type") == "rgb":
                params["diffuse_type"] = "rgb"
                params["base_color_rgb"] = diff_ref.get("value")
            elif isinstance(diff_ref, dict) and diff_ref.get("type") == "bitmap":
                textured_local |= add_tex(textures, "base_color", diff_ref)
                params["diffuse_type"] = "bitmap"

            alpha = b.get("alpha")
            params["alpha_type"] = "scalar"
            params["alpha"] = alpha

            params["distribution"] = b.get("distribution")
            params["int_ior"] = b.get("int_ior")
            params["ext_ior"] = b.get("ext_ior")

        elif t == "normalmap":
            inner = b.get("bsdf", {})
            inner_type, inner_textured, inner_params = _summ(inner)
            mat_type = inner_type
            params = dict(inner_params)
            textures = dict(inner_params.get("textures", {}))
            textured_local = bool(inner_textured)

            nm = b.get("normalmap", {})
            if add_tex(textures, "normal", nm):
                textured_local = True

        elif t == "bumpmap":
            inner = b.get("bsdf", {})
            inner_type, inner_textured, inner_params = _summ(inner)
            mat_type = inner_type
            params = dict(inner_params)
            textures = dict(inner_params.get("textures", {}))
            textured_local = bool(inner_textured)

            params["bump_scale"] = b.get("scale")
            tex = b.get("texture", {})
            if add_tex(textures, "height", tex):
                textured_local = True

        if textures:
            params["textures"] = textures

        return mat_type, textured_local, params

    mat_type, textured, params = _summ(bsdf)
    if mat_type == "pplastic" and ("textures" in params):
        mat_type = "pplastic_textured"
    return mat_type, textured, params

@lru_cache(maxsize=None)
def _cached_bbox(path_str):
    def load_obj(use_face_normals=False):
        params = {"type": "obj", "filename": path_str}
        if use_face_normals:
            params["face_normals"] = True
        return mi.load_dict(params)

    try:
        shape = load_obj()
    except Exception:
        shape = load_obj(use_face_normals=True)

    bbox = shape.bbox()
    bb_min = [float(bbox.min[i]) for i in range(3)]
    bb_max = [float(bbox.max[i]) for i in range(3)]
    return bb_min, bb_max

def _unit_dir_uniform_sphere_from_random(rng):
    u = rng.random()
    v = rng.random()

    z = 2.0 * u - 1.0   # [-1, 1]
    t = 2.0 * math.pi * v  # azimuth in [0, 2pi)
    r = math.sqrt(max(0.0, 1.0 - z*z)) # radius in xy plane

    x = r * math.cos(t)
    y = r * math.sin(t)
    return np.array([x, y, z], dtype=float)

def sample_lights_on_front_hemisphere(rng, scene_center, cam_origin, scene_radius, cfg=None):
    if cfg is None:
        cfg = LIGHT_CFG

    center = np.asarray(scene_center, dtype=float)
    cam_origin = np.asarray(cam_origin, dtype=float)

    num_lights = cfg.get("num_lights", 6)
    distance_scale = cfg.get("distance_scale", 1.5)

    cam_vec = cam_origin - center
    cam_dist = np.linalg.norm(cam_vec)

    base_radius = distance_scale * max(cam_dist, scene_radius + 1.0)

    # Hemisphere axis: from center -> camera
    if cam_dist < 1e-8:
        to_cam_hat = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        to_cam_hat = cam_vec / cam_dist

    positions = []
    for _ in range(num_lights):
        d = _unit_dir_uniform_sphere_from_random(rng)

        # Keep camera-side hemisphere
        if np.dot(d, to_cam_hat) < 0.0:
            d = -d

        radius = base_radius * (0.9 + 0.2 * rng.random())  # uniform in [0.9, 1.1]
        pos = center + radius * d
        positions.append(pos.tolist())

    return positions

def _camera_basis_from_lookat(cam_origin, cam_target):
    o = np.asarray(cam_origin, dtype=float)
    t = np.asarray(cam_target, dtype=float)
    forward = t - o
    f_norm = np.linalg.norm(forward)
    if f_norm < 1e-6:
        forward = np.array([0.0, 0.0, -1.0], dtype=float)
        f_norm = 1.0
    forward = forward / f_norm

    world_up = np.array([0.0, 1.0, 0.0], dtype=float)
    right = np.cross(forward, world_up)
    r_norm = np.linalg.norm(right)
    if r_norm < 1e-6:
        right = np.array([0.0, 0.0, 1.0], dtype=float)
        r_norm = 1.0
    right = right / r_norm
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    return right, up, forward

# -------------------------
# Lights and polarizers
# -------------------------
def _blind_light_direction(cam_origin, cam_target, rng=None, require_up=False, max_tries=1000):
    
    if rng is None:
        rng = np.random.default_rng()

    right, up, forward = _camera_basis_from_lookat(cam_origin, cam_target)

    for _ in range(max_tries):
        x = rng.normal(size=3)
        a, b, c = float(x[0]), float(x[1]), float(x[2])
        d = a * right + b * up + c * forward

        if np.dot(d, forward) <= 0:
            continue

        if require_up and np.dot(d, up) >= 0:
            continue

        n = np.linalg.norm(d)
        if n > 1e-8:
            d = d / n
            return d.tolist()

    d = -right - up + forward
    d = d / (np.linalg.norm(d) + 1e-8)
    return d.tolist()
    

def add_blind_directional_light(cam_origin, cam_target, cfg=None, rng=None):
    if cfg is None:
        cfg = BLIND_LIGHT_CFG
    if not cfg.get("enabled", True):
        return None

    direction = _blind_light_direction(
        cam_origin, cam_target, rng=rng,
        require_up=cfg.get("require_up", False),
    )

    irr = cfg.get("irradiance")

    return {
        "type": "directional",
        "direction": direction,
        "irradiance": irr,
    }


def per_light_polarizer_for_point(name, light_pos, scene_center, scene_radius, theta, cfg=None):
    if cfg is None:
        cfg = POLARIZER_CFG

    L = np.asarray(light_pos, dtype=float)
    C = np.asarray(scene_center, dtype=float)
    direction = C - L
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        direction = np.array([0.0, 0.0, -1.0], dtype=float)
        dist = 1.0
    direction = direction / dist

    frac = cfg.get("distance_fraction", 0.05)
    frac = max(0.0, min(1.0, frac))
    plate_center = L + direction * (dist * frac)

    origin = plate_center.tolist()
    target = (plate_center + direction).tolist()

    size_factor = cfg.get("size_factor", 1.3)

    R = max(scene_radius, 1e-6)
    D = dist
    s = dist * frac
    
    den = max(D*D - R*R, 1e-6)
    r_needed = s * (R / math.sqrt(den))
    
    size = size_factor * r_needed  

    pol_to_world = (
        mi.ScalarTransform4f.look_at(origin=origin, target=target, up=[0.0, 1.0, 0.0])
        @ mi.ScalarTransform4f.scale([size, size, size])
    )

    return {
        f"{name}_pol": {
            "type": "disk",
            "to_world": pol_to_world,
            "bsdf": {
                "type": "polarizer",
                "theta": float(theta),
                "transmittance": {"type": "spectrum", "value": 1.0},
            },
        }
    }

def random_light_setup(rng, scene_center, cam_origin, scene_radius):
    COLOR_MAP = {
        "R": [100, 0, 0],
        "G": [0, 100, 0],
        "B": [0, 0, 100],
    }
    color_labels = ["R", "R", "G", "G", "B", "B"]  
    rng.shuffle(color_labels)

    allowed_thetas = [0.0, 45.0, 90.0, 135.0]
    thetas = [None] * len(color_labels)
    for color in ["R", "G", "B"]:
        indices = [i for i, c in enumerate(color_labels) if c == color]
        chosen_angles = rng.sample(allowed_thetas, len(indices))
        for idx_light, angle in zip(indices, chosen_angles):
            thetas[idx_light] = angle

    positions = sample_lights_on_front_hemisphere(
        rng, scene_center, cam_origin, scene_radius, cfg=LIGHT_CFG
    )

    light_setup = []
    for idx, (pos, c_label, angle) in enumerate(zip(positions, color_labels, thetas), start=1):
        rgb = COLOR_MAP[c_label]
        name = f"L{idx}"
        light_setup.append((name, pos, rgb, angle, c_label))
    return light_setup


def _shuffle_inplace(rng, xs):
    try:
        rng.shuffle(xs)
    except AttributeError:
        for i in range(len(xs) - 1, 0, -1):
            j = int(rng.random() * (i + 1))
            xs[i], xs[j] = xs[j], xs[i]


def layout_objects(rng, obj_paths, center=None, x_range=(-8.0, 8.0), y_range=(-1.0, 1.0), z_range=(-8.0, 8.0),
                   recenter_axes=("x", "z"),     # recenter along these axes to avoid all-left/all-right
                   y_offset=-0.7,                # global vertical offset for the whole cluster
                   ):
    
    n = len(obj_paths)
    assert n > 0

    if center is None:
        center = SCENE_CENTER
    center = np.asarray(center, dtype=float)

    # Shared target size for all objects 
    scene_target_extent = TARGET_EXTENT
    scene_target_extent = float(scene_target_extent)

    # Sample positions in a 3D box 
    base_positions = []
    for _ in range(n):
        x = rng.uniform(x_range[0], x_range[1])
        y = rng.uniform(y_range[0], y_range[1])
        z = rng.uniform(z_range[0], z_range[1])
        base_positions.append([x, y, z])

    base_np = np.asarray(base_positions, dtype=float)

    # Recenter along selected axes to prevent drift to one side
    axis_map = {"x": 0, "y": 1, "z": 2}
    for ax in recenter_axes:
        k = axis_map[ax]
        base_np[:, k] -= base_np[:, k].mean()

    # Place around requested center and apply optional vertical offset
    base_np += center
    base_np[:, 1] += float(y_offset)

    # Coarse radius estimate for camera/light heuristics
    approx_obj_r = 0.5 * scene_target_extent
    scene_radius = float(np.max(np.linalg.norm(base_np - center, axis=1) + approx_obj_r))

    scales = [None] * n

    return (
        base_np.tolist(),
        scales,
        center.tolist(),
        scene_radius,
        scene_target_extent,
    )


def sample_camera(rng, scene_center, scene_radius=None, cfg=None, target=None):

    if cfg is None:
        cfg = CAMERA_CFG

    elev = rng.uniform(*cfg["elevation_deg_range"])
    azim = rng.uniform(*cfg["azimuth_deg_range"])

    elev_rad = math.radians(elev)
    azim_rad = math.radians(azim)

    cx, cy, cz = scene_center

    if scene_radius is None:
        r = rng.uniform(*cfg["distance_range"])
    else:
        distance_scale = float(cfg.get("distance_scale", 2.0))
        min_scene_radius = float(cfg.get("min_scene_radius", 1.0))
        r = distance_scale * max(float(scene_radius), min_scene_radius)

    origin = [
        cx + r * math.cos(elev_rad) * math.sin(azim_rad),
        cy + r * math.sin(elev_rad),
        cz + r * math.cos(elev_rad) * math.cos(azim_rad),
    ]

    if target is None:
        target = list(scene_center)

    return origin, list(target)


def random_transform(rng, base_translate, obj_path, scale, rotate=False):
    
    bb_min, bb_max = _cached_bbox(str(obj_path))
    bb_min = np.asarray(bb_min, float)
    bb_max = np.asarray(bb_max, float)

    rot_deg = rng.uniform(0.0, 360.0) if rotate else 0.0

    center_x = 0.5 * (bb_min[0] + bb_max[0])
    center_y = bb_min[1]
    center_z = 0.5 * (bb_min[2] + bb_max[2])

    return (
        mi.ScalarTransform4f.translate(base_translate)
        @ mi.ScalarTransform4f.rotate([0, 1, 0], rot_deg)
        @ mi.ScalarTransform4f.scale(scale)
        @ mi.ScalarTransform4f.translate([-center_x, -center_y, -center_z])
    )


def ortho_bounds_about_origin(P_centers, R, origin, right, up):
    
    P = np.asarray(P_centers, dtype=float)
    R = np.asarray(R, dtype=float)
    o = np.asarray(origin, dtype=float)

    D = P - o
    pr = D @ right
    pu = D @ up

    min_r = float(np.min(pr - R))
    max_r = float(np.max(pr + R))
    min_u = float(np.min(pu - R))
    max_u = float(np.max(pu + R))
    return min_r, max_r, min_u, max_u


def build_scene(obj_paths, rng, tex_pool):
    
    mat_records = []
    shapes = {}

    # ---------------------------------------------------------------------
    # Layout
    # ---------------------------------------------------------------------
    base_positions, _, layout_center, _, scene_target_extent = layout_objects(
        rng,
        obj_paths,
        center=SCENE_CENTER,
        x_range=(-8.0, 8.0),
        y_range=(-1.0, 1.0),
        z_range=(-8.0, 8.0),
        recenter_axes=("x", "z"),
        y_offset=0.0,  
    )

    C = np.asarray(layout_center, dtype=float)

    # ---------------------------------------------------------------------
    # Compute per-object scale/radius/center for framing 
    # ---------------------------------------------------------------------
    scales = []
    radii = []
    half_heights = []
    center_positions = []

    for path, bp in zip(obj_paths, base_positions):
        bb_min, bb_max = _cached_bbox(str(path))
        bb_min = np.asarray(bb_min, float)
        bb_max = np.asarray(bb_max, float)

        extents = bb_max - bb_min
        max_extent = float(np.max(extents))

        # aligned size: largest bbox dimension becomes scene_target_extent
        s = float(scene_target_extent) / max_extent
        scales.append(float(s))

        r = 0.5 * max_extent * s
        radii.append(float(r))

        hh = 0.5 * float(extents[1]) * s
        half_heights.append(float(hh))

        center_positions.append([bp[0], bp[1] + float(hh), bp[2]])

    # Accurate scene_radius from centers + per-object radii
    scene_radius = 0.0
    for cp, r in zip(center_positions, radii):
        scene_radius = max(scene_radius, float(np.linalg.norm(np.asarray(cp) - C) + r))

    # ---------------------------------------------------------------------
    # Camera + orthographic framing 
    # ---------------------------------------------------------------------
    target = np.mean(np.asarray(center_positions, float), axis=0).tolist()
    cam_origin, cam_target = sample_camera(
        rng,
        scene_center=layout_center,
        scene_radius=scene_radius,
        target=target
    )

    right, up, _forward = _camera_basis_from_lookat(cam_origin, cam_target)
    right = np.asarray(right, float)
    up = np.asarray(up, float)

    sensor_offset = [0.0, 0.0]
    sensor_scale = float(CAMERA_CFG.get("base_scale", 1.0))
    cluster_fit_scale = 1.0  # no cluster scaling

    vertical_bias_frac = float(ORTHO_FRAME_CFG.get("vertical_bias_frac", 0.0))

    if ORTHO_FRAME_CFG.get("enabled", True):
        fill_fraction = float(ORTHO_FRAME_CFG.get("fill_fraction", 0.94))
        smin, smax = ORTHO_FRAME_CFG.get("clamp_scale", (0.25, 6.0))

        min_r, max_r, min_u, max_u = ortho_bounds_about_origin(
            center_positions, radii, cam_origin, right, up
        )

        mid_r = 0.5 * (min_r + max_r)
        mid_u = 0.5 * (min_u + max_u)

        half_r = 0.5 * (max_r - min_r)
        half_u = 0.5 * (max_u - min_u)
        base_half = max(half_r, half_u, 1e-6)

        sensor_scale = base_half / max(fill_fraction, 1e-6)
        sensor_scale = float(np.clip(sensor_scale, smin, smax))

        mid_u += vertical_bias_frac * sensor_scale
        sensor_offset = [mid_r, mid_u]
    else:
        sensor_offset[1] += vertical_bias_frac * sensor_scale

    sensor_to_world = (
        mi.ScalarTransform4f.look_at(origin=cam_origin, target=cam_target, up=[0, 1, 0])
        @ mi.ScalarTransform4f.translate([sensor_offset[0], sensor_offset[1], 0.0])
        @ mi.ScalarTransform4f.scale([sensor_scale, sensor_scale, 1.0])
    )

    # ---------------------------------------------------------------------
    # Base scene dict
    # ---------------------------------------------------------------------
    scene_dict = {
        "type": "scene",
        "integrator": {
            "type": "stokes",
            "integrator": {"type": "volpath", "max_depth": 8, "rr_depth": 64},
        },
        "sensor": {
            "type": "orthographic",
            "to_world": sensor_to_world,
            "sampler": {"type": "independent", "sample_count": SAMPLES_PER_PIXEL},
            "film": {
                "type": "hdrfilm",
                "width": IMAGE_RES,
                "height": IMAGE_RES,
                "pixel_format": "rgb",
                "rfilter": {"type": "gaussian"},
            },
        },
    }

    # ---------------------------------------------------------------------
    # Materials: half roughconductor/ half pplastic
    # ---------------------------------------------------------------------
    n = len(obj_paths)
    n_rough = (n + 1) // 2
    n_plas = n - n_rough
    mat_families = (["roughconductor"] * n_rough) + (["pplastic"] * n_plas)
    _shuffle_inplace(rng, mat_families)

    # ---------------------------------------------------------------------
    # Add shapes
    # ---------------------------------------------------------------------
    for idx, (path, base_pos, scale, mat_family) in enumerate(
        zip(obj_paths, base_positions, scales, mat_families), 1
    ):
        if mat_family == "roughconductor":
            bsdf = make_random_roughconductor(rng, tex_pool=tex_pool)
        else:
            bsdf = make_random_pplastic(rng, tex_pool=tex_pool)

        material_type, textured, material_params = summarize_material(bsdf)

        path_obj = Path(path)
        try:
            asset_relpath = str(path_obj.relative_to(ASSET_DIR))
        except ValueError:
            asset_relpath = path_obj.name

        mat_records.append({
            "object": path_obj.name,
            "object_path": str(path_obj),
            "asset_relpath": asset_relpath,
            "material_type": material_type,
            "textured": bool(textured),
            "scale": float(scale),
            "base_position": list(base_pos),
            "material_params": material_params,
            "bsdf": copy.deepcopy(bsdf),
        })

        shapes[f"shape_{idx}"] = {
            "type": "obj",
            "filename": str(path),
            "to_world": random_transform(
                rng, base_pos, path,
                scale=float(scale),
                rotate=ROTATION, 
            ),
            "bsdf": bsdf,
        }

    scene_dict.update(shapes)

    # ---------------------------------------------------------------------
    # Lights 
    # ---------------------------------------------------------------------
    blind_light_info = add_blind_directional_light(cam_origin, cam_target)
    if blind_light_info is not None:
        scene_dict["blind_light"] = blind_light_info

    lights = random_light_setup(rng, layout_center, cam_origin, scene_radius)
    light_records = []
    for name, pos, rgb, theta, color_label in lights:
        scene_dict.update(point_emitter(name, origin=pos, rgb=rgb, rng=rng))
        scene_dict.update(
            per_light_polarizer_for_point(
                name,
                light_pos=pos,
                scene_center=layout_center,
                scene_radius=scene_radius,
                theta=theta,
            )
        )
        light_records.append({
            "name": name,
            "type": "point",
            "position": pos,
            "rgb": rgb,
            "theta": theta,
            "color_label": color_label,
        })

    # ---------------------------------------------------------------------
    # Scene meta
    # ---------------------------------------------------------------------
    scene_meta = {
        "scene_center": layout_center,
        "scene_radius": float(scene_radius),
        "scene_target_extent": float(scene_target_extent),
        "cluster_fit_scale": float(cluster_fit_scale),
        "sensor_offset": [float(sensor_offset[0]), float(sensor_offset[1])],
        "camera": {
            "origin": cam_origin,
            "target": cam_target,
            "sensor_scale": float(sensor_scale),
        },
        "blind_light": blind_light_info,
    }

    return scene_dict, mat_records, light_records, scene_meta



def render_scene(scene_dict, seed, out_path):
    out_path = Path(out_path).with_suffix(".exr")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, seed=seed)
    integrator = scene.integrator()
    channel_names = ["R", "G", "B"] + integrator.aov_names()
    bitmap32 = mi.Bitmap(image, channel_names=channel_names)
    arr32 = np.array(bitmap32, copy=False)
    arr32 = arr32.astype(np.float32)
    bitmap32 = mi.Bitmap(arr32, channel_names=channel_names)
    bitmap32.write(str(out_path))

def _serialize_transforms(obj):
    """Replace Mitsuba transforms with plain data so multiprocessing can pickle the scene."""
    if isinstance(obj, mi.ScalarTransform4f):
        mat = np.array(obj.matrix, dtype=float).reshape(-1).tolist()
        return {"__transform__": True, "matrix": mat}
    if isinstance(obj, dict):
        return {k: _serialize_transforms(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_transforms(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_serialize_transforms(v) for v in obj)
    return obj

def _deserialize_transforms(obj):
    """Rebuild Mitsuba transforms from serialized placeholders."""
    if isinstance(obj, dict):
        if obj.get("__transform__") and "matrix" in obj:
            mat = np.array(obj["matrix"], dtype=float).reshape(4, 4)
            return mi.ScalarTransform4f(mat)
        return {k: _deserialize_transforms(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize_transforms(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_deserialize_transforms(v) for v in obj)
    return obj


def _render_scene_pair(scene_dict, scene_seed, stokes_path, normal_path):
    """Render stokes and normals together (used inside a timeout-protected process)."""
    mi.set_variant("cuda_ad_spectral_polarized")
    mi.set_log_level(mi.LogLevel.Warn)
    scene = _deserialize_transforms(scene_dict)
    render_scene(scene, seed=scene_seed, out_path=stokes_path)
    render_normals(scene, out_path=normal_path)

def render_normals(base_scene_dict, out_path):
    scene_norm = dict(base_scene_dict)
    scene_norm = {k: v for k, v in scene_norm.items() if not k.endswith("_pol")}
    scene_norm["integrator"] = {"type": "aov", "aovs": "nn:geo_normal"}

    sensor = dict(base_scene_dict.get("sensor", {}))
    sensor["film"] = {
        "type": "hdrfilm",
        "width": IMAGE_RES,
        "height": IMAGE_RES,
        "pixel_format": "rgb",
        "rfilter": {"type": "gaussian"},
    }
    scene_norm["sensor"] = sensor

    out_path = Path(out_path).with_suffix(".exr")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene = mi.load_dict(scene_norm)
    img = mi.render(scene)
    bmp32 = mi.Bitmap(img)
    arr32 = np.array(bmp32, copy=False)
    arr32 = arr32.astype(np.float32)
    bmp32 = mi.Bitmap(arr32)
    bmp32.write(str(out_path))

def save_light_positions_plot(light_records, scene_meta, out_png, compress=1.0, margin_frac=0.0, ndp=2, legend_width=0.25, debug=False):
    
    def swap_yz(v):
        v = np.asarray(v, dtype=float)
        return np.array([v[0], v[2], v[1]], dtype=float)

    C_w = np.array(scene_meta["scene_center"], dtype=float)
    cam_w = np.array(scene_meta["camera"]["origin"], dtype=float)

    P_true_w = (
        np.array([lr["position"] for lr in light_records], dtype=float)
        if light_records else np.empty((0, 3), dtype=float)
    )

    P_vis_w = C_w + float(compress) * (P_true_w - C_w) if P_true_w.shape[0] > 0 else P_true_w

    C = swap_yz(C_w)
    cam = swap_yz(cam_w)
    P_vis = np.array([swap_yz(p) for p in P_vis_w], dtype=float) if P_vis_w.shape[0] > 0 else P_vis_w

    bl = scene_meta.get("blind_light")
    d_w = None
    incoming_plot = None
    arrow_tip = None

    if bl and bl.get("type") == "directional":
        d_w = np.array(bl["direction"], dtype=float)
        d_w = d_w / (np.linalg.norm(d_w) + 1e-12)

        d_plot = swap_yz(d_w)
        d_plot = d_plot / (np.linalg.norm(d_plot) + 1e-12)

        incoming_plot = d_plot

        R = float(max(scene_meta["scene_radius"], 1.0))
        L = 1.5 * R
        arrow_tip = C + incoming_plot * L

        if debug:
            print(f"[blind_light] direction world (travel) = {d_w}")
            print(f"[blind_light] direction plotted (travel) = {d_plot}")
            print(f"[blind_light] incoming plotted = {incoming_plot}")

    pts_for_bounds = [C[None, :], cam[None, :]]
    if P_vis.shape[0] > 0:
        pts_for_bounds.append(P_vis)
    if arrow_tip is not None:
        pts_for_bounds.append(arrow_tip[None, :])

    all_pts = np.vstack(pts_for_bounds)
    xyz_min = all_pts.min(axis=0)
    xyz_max = all_pts.max(axis=0)

    span = xyz_max - xyz_min
    span[span < 1e-6] = 1.0
    pad = float(margin_frac) * span
    xyz_min -= pad
    xyz_max += pad

    fig = plt.figure()
    split = 1.0 - float(legend_width)
    split = max(0.50, min(0.97, split))

    plot_left = -0.03  
    ax = fig.add_axes([plot_left, 0.0, split - plot_left, 1.0], projection="3d")
    ax_leg = fig.add_axes([split, 0.0, 1.0 - split, 1.0])
    ax_leg.axis("off")

    ax.xaxis.labelpad = 0
    ax.yaxis.labelpad = 0
    ax.zaxis.labelpad = 0
    ax.tick_params(pad=0)

    legend_handles = []
    for lr, p_vis_plot, p_true in zip(light_records, P_vis, P_true_w):
        rgb = np.array(lr.get("rgb", [1, 1, 1]), dtype=float)
        mx = float(rgb.max())
        if mx > 1.0:
            rgb = rgb / mx

        ax.scatter([p_vis_plot[0]], [p_vis_plot[1]], [p_vis_plot[2]], c=[rgb.tolist()])

        label = f"{lr['name']}: ({p_true[0]:.{ndp}f}, {p_true[1]:.{ndp}f}, {p_true[2]:.{ndp}f})"
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markerfacecolor=rgb.tolist(),
                markeredgecolor="black",
                markersize=8,
                label=label,
            )
        )

    ax.scatter([C[0]], [C[1]], [C[2]], c=[[0.0, 0.0, 0.0]])
    ax.text(C[0], C[1], C[2], "Center")

    ax.scatter([cam[0]], [cam[1]], [cam[2]], c=[[0.35, 0.35, 0.35]])
    ax.text(cam[0], cam[1], cam[2], "Camera")

    if incoming_plot is not None:
        R = float(max(scene_meta["scene_radius"], 1.0))
        L = 1.5 * R

        ax.quiver(
            C[0], C[1], C[2],
            incoming_plot[0], incoming_plot[1], incoming_plot[2],
            length=L,
            normalize=True
        )
        ax.text(arrow_tip[0], arrow_tip[1], arrow_tip[2], "dir")

    ax.set_xlim(xyz_min[0], xyz_max[0])
    ax.set_ylim(xyz_min[1], xyz_max[1])
    ax.set_zlim(xyz_min[2], xyz_max[2])

    try:
        ax.set_box_aspect((xyz_max - xyz_min))
    except Exception:
        pass

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")

    title = "Light positions\n"
    fig.suptitle(title, x=0.5, y=0.995, ha="center", va="top")

    # ----- LEGEND -----
    if legend_handles:
        ax_leg.legend(
            handles=legend_handles,
            title="Lights",
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
            borderaxespad=0.0,
            frameon=False,
            handletextpad=0.4,
            labelspacing=0.1,
            fontsize=8,
            title_fontsize=9,
        )

    fig.savefig(str(out_png), dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def _flush_gpu_memory():
    try:
        dr.flush_malloc_cache()
    except Exception:
        pass
    gc.collect()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_entries = []
    log_path = OUTPUT_DIR / "materials_log.json"

    if log_path.exists():
        try:
            log_entries = json.loads(log_path.read_text())
            print(f"Resuming after {len(log_entries)} completed scenes from existing log.")
        except Exception:
            print("Existing log unreadable; starting fresh.")
            log_entries = []

    assets = sorted(ASSET_DIR.glob("*.obj"))
    if len(assets) < 6:
        raise RuntimeError(f"Need at least 6 OBJ files in {ASSET_DIR}")

    tex_pool = discover_textures(MATERIAL_DIR)
    if not tex_pool:
        print(f"No texture sets found under {MATERIAL_DIR}.")

    idx = len(log_entries) + 1
    attempts = 0
    log_flush_interval = 25

    while idx <= NUM_SCENES:
        attempts += 1
        picked = rng.sample(assets, 6)
        try:
            scene_dict, mat_records, light_records, scene_meta = build_scene(picked, rng, tex_pool)

            # lights_png = OUTPUT_DIR / f"scene_{idx:05d}_lights.png"
            # save_light_positions_plot(light_records, scene_meta, lights_png, compress=1.0)

            stokes_path = OUTPUT_DIR / f"scene_{idx:05d}_stokes.exr"
            normal_path = OUTPUT_DIR / f"scene_{idx:05d}_normal.exr"
            scene_seed = seed + idx - 1

            scene_dict_serial = _serialize_transforms(scene_dict)

            proc = mp.Process(
                target=_render_scene_pair,
                args=(scene_dict_serial, scene_seed, stokes_path, normal_path),
            )
            proc.start()
            proc.join(RENDER_TIMEOUT_SEC)

            if proc.is_alive():
                proc.terminate()
                proc.join()
                print(f"[timeout {idx:05d}] render exceeded {RENDER_TIMEOUT_SEC}s; skipping scene")
                continue

            if proc.exitcode != 0:
                print(f"[skip attempt {attempts}] Render process failed with exit code {proc.exitcode}")
                continue

            picked_names = ", ".join(p.name for p in picked)
            print(f"[{idx:05d}/{NUM_SCENES}] wrote {stokes_path.name} + {normal_path.name} using {picked_names}")

            log_entries.append({
                "scene_id": idx,
                "scene": stokes_path.name,
                "normal": normal_path.name,
                "scene_path": str(stokes_path.relative_to(OUTPUT_DIR)),
                "normal_path": str(normal_path.relative_to(OUTPUT_DIR)),
                "scene_center": scene_meta["scene_center"],
                "scene_radius": scene_meta["scene_radius"],
                "cluster_fit_scale": scene_meta.get("cluster_fit_scale", 1.0),
                "sensor_offset": scene_meta.get("sensor_offset", [0.0, 0.0]),
                "camera": scene_meta["camera"],
                "blind_light": scene_meta.get("blind_light"),
                "render_settings": {
                    "image_res": IMAGE_RES,
                    "spp": SAMPLES_PER_PIXEL,
                    "integrator": "stokes_volpath",
                    "mitsuba_variant": "cuda_ad_spectral_polarized",
                },
                "objects": mat_records,
                "lights": light_records,
            })

            if idx % log_flush_interval == 0 or idx == NUM_SCENES:
                log_path.write_text(json.dumps(log_entries, indent=2))

            idx += 1
        except Exception as e:
            print(f"[skip attempt {attempts}] Failed with {picked}: {e}")
            continue
        finally:
            _flush_gpu_memory()

    log_path.write_text(json.dumps(log_entries, indent=2))
    print(f"Material log written to {log_path}")

if __name__ == "__main__":
    main()
