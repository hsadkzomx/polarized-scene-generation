# Polarized Scene Generation and Visualization

This repository contains tools for generating synthetic polarized scenes using Mitsuba 3 and visualizing the resulting Stokes parameters.

## Files

1.  `generate_polarized_scenes.py`: Generates synthetic scenes with random layouts of objects, lights, and polarizers, rendering them to EXR files containing Stokes parameters.
2.  `visualize_stokes.py`: Processes the rendered EXR files to generate visualizations for Intensity (RGB), Degree of Linear Polarization (DoLP), and Angle of Linear Polarization (AoLP).
3.  `requirements.txt`: Python package dependencies.

## Prerequisites

- [Mitsuba 3](https://mitsuba3.readthedocs.io/) (with `cuda_ad_spectral_polarized` variant enabled)
- DrJit
- Python 3.8+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quickstart

1. Install Mitsuba 3 with the `cuda_ad_spectral_polarized` variant (GPU required).
2. Install Python dependencies: `pip install -r requirements.txt`.
3. Edit `ASSET_DIR`, `MATERIAL_DIR`, and `OUTPUT_DIR` in `generate_polarized_scenes.py`.
4. Run scene generation: `python generate_polarized_scenes.py`.
5. Edit `input_dir` in `visualize_stokes.py` to point at your output folder.
6. Run visualization: `python visualize_stokes.py`.

## Usage

### 1. Scene Generation

Open `generate_polarized_scenes.py` and configure the following paths at the top of the file:

```python
ASSET_DIR = Path(r"path/to/models")          # Directory containing OBJ models
MATERIAL_DIR = Path(r"path/to/materials")    # Directory containing PBR textures
OUTPUT_DIR = Path(__file__).resolve().parent / "data_polarized_30000" # Output directory
```

Run the script:
```bash
python generate_polarized_scenes.py
```

#### Key Parameters in `generate_polarized_scenes.py`

**Paths and dataset**

*   **`NUM_SCENES`**: Total number of scenes to generate (default: 30000).
*   **`ASSET_DIR`**: Directory containing `.obj` models (must have at least 6).
*   **`MATERIAL_DIR`**: Directory containing PBR texture sets. The loader looks for `*_baseColor.*` names.
*   **`OUTPUT_DIR`**: Output folder for EXR files and `materials_log.json`.

**Render quality**

*   **`IMAGE_RES`**: Resolution of the rendered images (default: 512x512).
*   **`SAMPLES_PER_PIXEL`**: Samples per pixel for the path tracer (default: 128). Higher values reduce noise but increase render time.
*   **`RENDER_TIMEOUT_SEC`**: Maximum time allowed for rendering a single scene.

**Scene layout**

*   **`TARGET_EXTENT`**: Target size that each object is scaled to (largest bbox dimension).
*   **`TEXTURED_PROB`**: Probability of applying a texture when available (0.0–1.0).
*   **`SCENE_CENTER`**: World-space center of the scene layout.
*   **`ROTATION`**: Enable random per-object rotation around Y.
*   **`seed`**: Random seed for reproducibility.

**Camera and framing**

*   **`CAMERA_CFG.distance_range`**: Used only when no scene radius is provided.
*   **`CAMERA_CFG.distance_scale`**: Multiplier on scene radius for camera distance (default: 2.0 if not set).
*   **`CAMERA_CFG.min_scene_radius`**: Minimum radius used when scaling camera distance (default: 1.0).
*   **`CAMERA_CFG.elevation_deg_range`**: Camera elevation range in degrees.
*   **`CAMERA_CFG.azimuth_deg_range`**: Camera azimuth range in degrees.
*   **`CAMERA_CFG.base_scale`**: Base orthographic scale before fitting.
*   **`ORTHO_FRAME_CFG.enabled`**: Enable auto-fit orthographic framing.
*   **`ORTHO_FRAME_CFG.fill_fraction`**: Fraction of the frame filled by the object cluster.
*   **`ORTHO_FRAME_CFG.vertical_bias_frac`**: Vertical shift relative to the fitted scale.
*   **`ORTHO_FRAME_CFG.clamp_scale`**: Clamp range for the fitted orthographic scale.

**Lights and polarizers**

*   **`LIGHT_CFG.num_lights`**: Number of point lights (RGB-balanced in pairs).
*   **`LIGHT_CFG.distance_scale`**: Light distance multiplier from the scene.
*   **`LIGHT_CFG.cos_theta_min` / `cos_theta_max`**: Reserved (not used in current sampling).
*   **`BLIND_LIGHT_CFG.enabled`**: Add a directional “blind” light.
*   **`BLIND_LIGHT_CFG.irradiance`**: Irradiance spectrum and scale for the blind light.
*   **`BLIND_LIGHT_CFG.require_up`**: Constrain blind light direction relative to camera up.
*   **`POLARIZER_CFG.distance_fraction`**: Fractional distance from light to scene where the polarizer is placed.
*   **`POLARIZER_CFG.size_factor`**: Scale factor for polarizer disk size.

#### Example Configuration (Small Test)

To do a quick sanity check without generating the full dataset, use small values:

```python
ASSET_DIR = Path(r"/path/to/models")
MATERIAL_DIR = Path(r"/path/to/materials")
OUTPUT_DIR = Path(__file__).resolve().parent / "data_polarized_test"

NUM_SCENES = 3
IMAGE_RES = 256
SAMPLES_PER_PIXEL = 32
RENDER_TIMEOUT_SEC = 600
```

#### Sample Outputs

After running `generate_polarized_scenes.py`, you should see:

```
data_polarized_test/
  scene_00001_stokes.exr
  scene_00001_normal.exr
  scene_00002_stokes.exr
  scene_00002_normal.exr
  scene_00003_stokes.exr
  scene_00003_normal.exr
  materials_log.json
```

### 2. Visualization

Open `visualize_stokes.py` and configure the input directory at the bottom of the file:

```python
if __name__ == "__main__":
    input_dir = Path(r"path/to/data_polarized_30000") # Directory containing rendered *_stokes.exr files
    # ...
```

Run the script:
```bash
python visualize_stokes.py
```

This will create a `visualization` subdirectory within your input directory containing PNG images for:
*   **Intensity**: RGB and per-channel.
*   **DoLP**: Degree of Linear Polarization (RGB and per-channel).
*   **AoLP**: Angle of Linear Polarization (Scalar and Hue-mapped).

#### Visualization Settings
*   **`input_dir`**: Folder containing the `*_stokes.exr` files (and optional `*_normal.exr`).
*   **`USE_DENOISER_FOR_VIS`**: Enable/disable OptiX denoiser for the visualization images (requires compatible GPU).
*   **`SAVE_PER_CHANNEL`**: Save individual R, G, B channel images.
*   **`SAVE_AOLP_HUE`**: Save false-color hue visualizations for AoLP.
*   **`DENOISE_AOLP` / `DENOISE_DOLP`**: Apply denoiser to AoLP/DoLP visualizations.

## Troubleshooting

*   **Missing Assets**: Ensure `ASSET_DIR` and `MATERIAL_DIR` point to valid directories with `.obj` files and texture maps respectively.
*   **Mitsuba Variant**: Ensure your Mitsuba installation supports the `cuda_ad_spectral_polarized` variant.

## License

MIT License. See `LICENSE`.

## Citation

If you use this code in academic work, please cite it. See `CITATION.cff`.
