import mitsuba as mi
import drjit as dr
import numpy as np
import cv2
import matplotlib.colors as mcolors
from pathlib import Path
from drjit.cuda.ad import TensorXf

mi.set_variant("cuda_ad_spectral_polarized")

USE_DENOISER_FOR_VIS = True  # visualization-only denoising 
SAVE_PER_CHANNEL = True  # save grayscale images 
SAVE_AOLP_HUE = True  # hue visualization for AOLP
DENOISE_AOLP = True  
DENOISE_DOLP = True  

def load_stokes(exr_path: Path, channels=("S0","S1","S2","S3")):
    bmp = mi.Bitmap(str(exr_path))
    parts = dict(bmp.split())
    out = {ch: np.array(parts[ch], copy=False).astype(np.float32) for ch in channels if ch in parts}
    if not out:
        raise RuntimeError(f"No Stokes channels found in {exr_path}")
    return out

def load_normal(normal_exr: Path):
    bmp = mi.Bitmap(str(normal_exr))
    n = np.array(bmp, copy=False).astype(np.float32) # [-1, 1]
    return np.clip(0.5 * (n + 1.0), 0.0, 1.0) # -> [0, 1]

def reconstruct_intensities(S0, S1, S2):
    I0   = 0.5 * (S0 + S1)
    I90  = 0.5 * (S0 - S1)
    I45  = 0.5 * (S0 + S2)
    I135 = 0.5 * (S0 - S2)
    return I0, I45, I90, I135

def compute_dolp_aolp(S0, S1, S2, eps=1e-4):
    denom = np.maximum(S0, eps)
    P = np.sqrt(S1*S1 + S2*S2)
    dolp = P / denom
    aolp = 0.5 * np.arctan2(S2, S1)
    return dolp, aolp

def normalize_01(img, clip_percentile=99.0):
    img = img.astype(np.float32, copy=False)
    pos = img[img > 0]
    v = np.percentile(pos, clip_percentile) if pos.size else 1.0
    if v <= 0:
        v = 1.0
    return np.clip(img / v, 0.0, 1.0)

def write_png16(path, img, rgb):
    arr = np.asarray(img, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    u16 = (arr * 65535.0 + 0.5).astype(np.uint16)
    if rgb:
        u16 = cv2.cvtColor(u16, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), u16)

def aolp_to_rgb(aolp_scalar):
    # aolp_scalar in radians [-pi/2, pi/2]
    h = (aolp_scalar + 0.5*np.pi) / np.pi
    h = np.mod(h, 1.0)
    hsv = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = h
    hsv[..., 1] = 1.0
    hsv[..., 2] = 1.0
    return mcolors.hsv_to_rgb(hsv)

# ---------------- OptiX denoiser for VIS outputs ----------------
def make_denoiser(w, h):
    return mi.OptixDenoiser(
        input_size=mi.ScalarVector2u(w, h),
        albedo=False, normals=False, temporal=False, denoise_alpha=False
    )

def denoise_vis_rgb(img01_hwc, denoiser):
    """
    Denoise a normalized [0,1] RGB visualization image (H,W,3).
    """
    img = np.asarray(img01_hwc, dtype=np.float32)
    img_whc = np.transpose(img, (1, 0, 2)).copy() # (W,H,3)
    t = TensorXf(img_whc)
    t_dn = denoiser(t)
    out_whc = np.array(t_dn, copy=False)
    out_hwc = np.transpose(out_whc, (1, 0, 2))
    return np.clip(out_hwc.astype(np.float32, copy=False), 0.0, 1.0)

def denoise_vis_gray(img01_hw, denoiser):
    """
    Denoise a normalized [0,1] grayscale visualization by replicating to RGB.
    """
    g = np.asarray(img01_hw, dtype=np.float32)
    rgb = np.stack([g, g, g], axis=2)
    rgb_dn = denoise_vis_rgb(rgb, denoiser)
    return rgb_dn[..., 0]

# ---------------- Main processing ----------------
def process_stokes_exr(exr_path: Path, out_root: Path):
    base = exr_path.stem.replace("_stokes", "")
    out_dir = out_root / base
    out_dir.mkdir(parents=True, exist_ok=True)

    st = load_stokes(exr_path)
    for req in ("S0", "S1", "S2"):
        if req not in st:
            raise RuntimeError(f"Missing {req} in {exr_path.name}")

    S0, S1, S2 = st["S0"], st["S1"], st["S2"]
    I0, I45, I90, I135 = reconstruct_intensities(S0, S1, S2)
    dolp, aolp = compute_dolp_aolp(S0, S1, S2)

    H, W = S0.shape[0], S0.shape[1]
    denoiser = make_denoiser(W, H) if USE_DENOISER_FOR_VIS else None

    cues = {
        "S0": S0, "S1": S1, "S2": S2,
        "I0": I0, "I45": I45, "I90": I90, "I135": I135
    }
    tags = ["R", "G", "B"]

    # ---- Save S*, I* ----
    for name, cue in cues.items():
        rgb01 = normalize_01(cue)

        if USE_DENOISER_FOR_VIS and denoiser is not None:
            rgb01 = denoise_vis_rgb(rgb01, denoiser)

        write_png16(out_dir / f"{base}_{name}_RGB_u16.png", rgb01, rgb=True)

        if SAVE_PER_CHANNEL:
            for c, tag in enumerate(tags):
                ch01 = normalize_01(cue[..., c])
                if USE_DENOISER_FOR_VIS and denoiser is not None:
                    ch01 = denoise_vis_gray(ch01, denoiser)
                write_png16(out_dir / f"{base}_{name}_{tag}_u16.png", ch01, rgb=False)

    # ---- Save DoLP ----
    dolp01 = np.clip(dolp, 0.0, 1.0)
    if USE_DENOISER_FOR_VIS and denoiser is not None and DENOISE_DOLP:
        dolp01 = denoise_vis_rgb(dolp01, denoiser)

    write_png16(out_dir / f"{base}_DoLP_RGB_u16.png", dolp01, rgb=True)

    if SAVE_PER_CHANNEL:
        for c, tag in enumerate(tags):
            d = dolp01[..., c]
            if USE_DENOISER_FOR_VIS and denoiser is not None and DENOISE_DOLP:
                d = denoise_vis_gray(d, denoiser)
            write_png16(out_dir / f"{base}_DoLP_{tag}_u16.png", d, rgb=False)

    # ---- Save AoLP scalar ----
    # Map [-pi/2, pi/2] -> [0,1]
    a01 = np.clip((aolp + 0.5*np.pi) / np.pi, 0.0, 1.0)

    if USE_DENOISER_FOR_VIS and denoiser is not None and DENOISE_AOLP:
        a01 = denoise_vis_rgb(a01, denoiser)

    if SAVE_PER_CHANNEL:
        for c, tag in enumerate(tags):
            write_png16(out_dir / f"{base}_AoLP_{tag}_scalar_u16.png", a01[..., c], rgb=False)
    write_png16(out_dir / f"{base}_AoLP_scalar_RGB_u16.png", a01, rgb=True)

    if SAVE_AOLP_HUE:
        for c, tag in enumerate(tags):
            hue_rgb = aolp_to_rgb(aolp[..., c])
            write_png16(out_dir / f"{base}_AoLP_{tag}_hueRgb_u16.png", hue_rgb, rgb=True)

    # ---- Normal map ----
    normal_exr = exr_path.with_name(exr_path.name.replace("_stokes", "_normal"))
    if normal_exr.exists():
        n01 = load_normal(normal_exr)
        write_png16(out_dir / f"{base}_normal_u16.png", n01, rgb=True)

    print(f"Done: {exr_path.name} -> {out_dir}")

if __name__ == "__main__":
    input_dir = Path(r"F:\polar_synthetic_dataset\data_polarized_30000")
    out_root = input_dir / "visualization"
    out_root.mkdir(parents=True, exist_ok=True)

    for exr_path in sorted(input_dir.glob("*_stokes.exr")):
        process_stokes_exr(exr_path, out_root)
