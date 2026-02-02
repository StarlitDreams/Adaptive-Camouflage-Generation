
<img width="1200" height="180" alt="Untitled-1" src="https://github.com/user-attachments/assets/33437db2-3480-4f7a-87d3-a81a8352e934" />

### Biome‑aware reaction–diffusion camouflage (Gierer–Meinhardt / Turing patterns)



<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue"/></a>
  <a href="#"><img alt="Platform" src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-success"/></a>
  <a href="#"><img alt="Dependencies" src="https://img.shields.io/badge/Deps-numpy%20%7C%20pillow-informational"/></a>
  <a href="#"><img alt="Output" src="https://img.shields.io/badge/Output-PNG%20%7C%20GIF-orange"/></a>
</p>

Generate seamless camouflage textures **from a biome image** by:
1) extracting dominant colors from the biome,  
2) simulating a fast FFT‑based **reaction–diffusion** pattern (macro + micro),  
3) blending scales for realism, and  
4) mapping the pattern back to the biome palette.

**Outputs**
- `extracted_palette.png` — dominant color strip  
- `camouflage_final.png` — high‑resolution camouflage texture  
- `camouflage_evolution.gif` — pattern evolution (macro+micro)

---

## Table of contents
- [Features](#features)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Outputs](#outputs-1)
- [Recommended presets](#recommended-presets)
- [Project structure](#project-structure)
- [Images (biomes + personal)](#images-biomes--personal)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features
- **Biome‑aware palettes**: pulls dominant colors directly from your input image
- **Fast simulation**: FFT/rFFT diffusion for speed
- **Two‑scale realism**: macro + micro patterns blended into a more natural texture
- **Seamless textures**: periodic boundary behavior suitable for tiling
- **PNG + GIF outputs**: final texture plus a nice evolution preview

---

## Quickstart

### 1) Create a virtual environment
**Windows (PowerShell)**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run
```bash
python main.py --biome path/to/biome.jpg
```

---

## Usage

### Basic
```bash
python main.py --biome path/to/biome.jpg
```

### High quality (recommended)
```bash
python main.py --biome path/to/biome.jpg --final_res 4096 --gif_res 720 --snapshot_every 30
```

### Fast test run (iterate quickly)
```bash
python main.py --biome path/to/biome.jpg --steps 4000 --macro_n 256 --micro_n 384 --final_res 1024
```

### CLI arguments
| Argument | Default | What it does |
|---|---:|---|
| `--biome` | required | Input biome image path (jpg/png/etc). |
| `--colors` | `4` | Number of dominant colors to extract. |
| `--sample_size` | `256` | Downsample size for palette extraction. |
| `--crop_fraction` | `0.90` | Center‑crop fraction before extraction (avoid sky/horizon). |
| `--merge_dist` | `0.08` | Merge threshold for similar colors. |
| `--keep_white` | off | Keep near‑white colors (good for snow/sand). |
| `--ignore_black` | off | Ignore near‑black colors (shadows). |
| `--final_res` | `2048` | Final PNG resolution (square). |
| `--gif_res` | `640` | GIF resolution. |
| `--fps` | `24` | GIF frames per second. |
| `--macro_n` | `384` | Macro simulation grid size (big blobs). |
| `--micro_n` | `640` | Micro simulation grid size (fine texture). |
| `--steps` | `14000` | Simulation steps per scale. |
| `--snapshot_every` | `60` | Snapshot interval (lower = more GIF frames). |
| `--micro_weight` | `0.38` | Micro blending weight (0..1). |
| `--macro_blur` | `3.0` | Blur macro before blending (px). |
| `--dt` | `1.0` | Time step. |
| `--dx` | `1.0` | Grid spacing. |
| `--log_every` | `600` | Progress log interval. |

---

## Outputs

After a run you’ll get:

### `extracted_palette.png`
A horizontal strip of the extracted dominant colors.

<p align="center">
 <img width="1024" height="128" alt="extracted_palette" src="https://github.com/user-attachments/assets/3d73e066-b212-47ae-9d73-1114924b3c59" />
</p>

### `camouflage_final.png`
The final **high‑resolution** camouflage texture.

<p align="center">
  <img width="2048" height="2048" alt="camouflage_final" src="https://github.com/user-attachments/assets/7e3c445a-d796-4ed5-833a-b995f18ec4b8" />

</p>

### `camouflage_evolution.gif`
Evolution of the blended pattern over time.

<p align="center">
  

https://github.com/user-attachments/assets/861f66a9-b627-49b3-aadb-bf265f28e4a7


</p>

> Tip: If the GIF is huge, reduce `--gif_res` (e.g., 480) or increase `--snapshot_every`.

---

## Recommended presets

### Forest / jungle (rich greens)
```bash
python main.py --biome forest.jpg --colors 5 --crop_fraction 0.85 --final_res 4096 --gif_res 720 --snapshot_every 30
```

### Desert / sand (keep whites)
```bash
python main.py --biome desert.jpg --colors 4 --keep_white --final_res 4096 --gif_res 640 --snapshot_every 40
```

### Snow / arctic (keep whites, reduce black shadows)
```bash
python main.py --biome snow.jpg --colors 4 --keep_white --ignore_black --final_res 4096 --gif_res 640 --snapshot_every 40
```

### Urban / concrete (more micro detail)
```bash
python main.py --biome urban.jpg --colors 4 --micro_weight 0.45 --final_res 4096 --gif_res 720 --snapshot_every 30
```

---

## Project structure
Suggested structure that scales nicely:

```
.
├─ main.py
├─ requirements.txt
├─ README.md
├─ images/
│  ├─ biomes/
│  ├─ personal/
│  └─ examples/
└─ outputs/            # (optional) keep generated outputs here
```

---

## Images (biomes + personal)

You said you want an **Images section** that can include **your own photos**.

### Suggested folders
- `images/biomes/` — biome reference images (forest, desert, snow, etc.)
- `images/personal/` — personal/reference images (including yourself)
- `images/examples/` — curated examples for the README (safe to share)

### Privacy best practice (important)
If this repo might become public:
- **Do not commit** personal images.
- Add this to `.gitignore` if you want to keep only local:
```gitignore
images/personal/
```

If you already ignore all images globally, that’s fine too—your local files still work for generation.

---

## Troubleshooting

### It looks “stuck” / very slow
FFT cost grows quickly with grid size. Try:
- smaller grids: `--macro_n 256 --micro_n 384`
- fewer steps: `--steps 6000`
- fewer frames: `--snapshot_every 90`

### Palette looks wrong (sky dominates / horizon issues)
- reduce crop: `--crop_fraction 0.75`
- increase sample: `--sample_size 384`
- keep whites for snow/sand: `--keep_white`
- ignore black shadows: `--ignore_black`

### GIF is too large
- lower GIF resolution: `--gif_res 480`
- take fewer snapshots: `--snapshot_every 80`
- lower FPS: `--fps 12`

---

## Roadmap
- [ ] Optional output tiling test (auto‑stitch preview)
- [ ] Multiple palette strategies (k‑means / median cut / LAB clustering)
- [ ] Parameter auto‑tuning per biome (more “spotty” vs “stripy” control)
- [ ] Output folder support (`--outdir outputs/`)

---

## License
Add a license if you plan to distribute:
- MIT (simple + permissive)
- Apache‑2.0 (patent grant)
- GPL‑3.0 (copyleft)

---

## Acknowledgements
- Reaction–diffusion / Turing pattern inspiration
- Gierer–Meinhardt activator–inhibitor dynamics
