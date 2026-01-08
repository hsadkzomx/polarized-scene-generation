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

*   **`NUM_SCENES`**: Total number of scenes to generate (default: 30000).
*   **`IMAGE_RES`**: Resolution of the rendered images (default: 512x512).
*   **`SAMPLES_PER_PIXEL`**: Samples per pixel for the path tracer (default: 128). Higher values reduce noise but increase render time.
*   **`RENDER_TIMEOUT_SEC`**: Maximum time allowed for rendering a single scene.
*   **`CAMERA_CFG`**: configuration for camera placement (distance, elevation, azimuth).
*   **`LIGHT_CFG`**: Configuration for light sources (number, distance).
*   **`POLARIZER_CFG`**: Configuration for the polarizers placed in front of lights.

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
*   **`USE_DENOISER_FOR_VIS`**: Enable/disable OptiX denoiser for the visualization images (requires compatible GPU).
*   **`SAVE_PER_CHANNEL`**: Save individual R, G, B channel images.
*   **`SAVE_AOLP_HUE`**: Save false-color hue visualizations for AoLP.

## Troubleshooting

*   **Missing Assets**: Ensure `ASSET_DIR` and `MATERIAL_DIR` point to valid directories with `.obj` files and texture maps respectively.
*   **Mitsuba Variant**: Ensure your Mitsuba installation supports the `cuda_ad_spectral_polarized` variant.
