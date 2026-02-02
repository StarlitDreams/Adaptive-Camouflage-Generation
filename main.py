

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Sequence

import numpy as np
from PIL import Image


# Pillow resampling compatibility (fixes Pylance "BILINEAR not found")
# Pillow>=10: Image.Resampling.BILINEAR
# Older: Image.BILINEAR
try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    # For very old Pillow versions, use numeric constant
    RESAMPLE_BILINEAR = 1  # BILINEAR constant value


# -------------------------- Color / palette extraction --------------------------

def _rgb_to_luma(rgb01: np.ndarray) -> float:
    r, g, b = float(rgb01[0]), float(rgb01[1]), float(rgb01[2])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def _hex_from_rgb01(rgb01: np.ndarray) -> str:
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    r = int(round(float(rgb01[0]) * 255))
    g = int(round(float(rgb01[1]) * 255))
    b = int(round(float(rgb01[2]) * 255))
    return f"#{r:02x}{g:02x}{b:02x}"

def _rgb01_from_hex(h: str) -> np.ndarray:
    h = h.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Hex color must be like #RRGGBB, got: {h!r}")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return np.array([r, g, b], dtype=np.float32)

def _merge_similar_colors(
    rgb_list: List[np.ndarray],
    counts: List[int],
    min_dist: float,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Greedy merge near-duplicate colors by Euclidean distance in RGB01.
    min_dist ~ 0.06..0.12 typical.
    """
    if not rgb_list:
        return [], []

    order = np.argsort(-np.array(counts))
    rgb_list = [rgb_list[i] for i in order]
    counts = [counts[i] for i in order]

    merged_rgb: List[np.ndarray] = []
    merged_cnt: List[int] = []

    for rgb, cnt in zip(rgb_list, counts):
        placed = False
        for j in range(len(merged_rgb)):
            d = float(np.linalg.norm(rgb - merged_rgb[j]))
            if d < min_dist:
                total = merged_cnt[j] + cnt
                merged_rgb[j] = (merged_rgb[j] * (merged_cnt[j] / total) + rgb * (cnt / total)).astype(np.float32)
                merged_cnt[j] = total
                placed = True
                break
        if not placed:
            merged_rgb.append(rgb.astype(np.float32))
            merged_cnt.append(int(cnt))

    order2 = np.argsort(-np.array(merged_cnt))
    merged_rgb = [merged_rgb[i] for i in order2]
    merged_cnt = [merged_cnt[i] for i in order2]
    return merged_rgb, merged_cnt

def extract_dominant_palette(
    image_path: str,
    k: int = 4,
    sample_size: int = 256,
    crop_fraction: float = 0.90,
    min_merge_dist: float = 0.08,
    ignore_near_white: bool = True,
    ignore_near_black: bool = False,
) -> List[str]:
    """
    Extract k-ish dominant colors using Pillow adaptive palette quantization.

    Pylance notes:
    - getcolors/getpalette are Optional in stubs => guarded
    - getcolors returns palette indices OR RGB tuples depending on mode => handled
    """
    img = Image.open(image_path).convert("RGBA")

    crop_fraction = float(np.clip(crop_fraction, 0.2, 1.0))
    if crop_fraction < 1.0:
        w, h = img.size
        cw, ch = int(w * crop_fraction), int(h * crop_fraction)
        left = (w - cw) // 2
        top = (h - ch) // 2
        img = img.crop((left, top, left + cw, top + ch))

    img = img.resize((sample_size, sample_size), resample=RESAMPLE_BILINEAR)

    # Composite alpha over black
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    img = Image.alpha_composite(bg, img).convert("RGB")

    target = max(k * 4, 16)
    pal_img = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=target)

    colors = pal_img.getcolors(maxcolors=sample_size * sample_size)
    if colors is None:
        raise RuntimeError("Could not extract colors from image (getcolors returned None).")

    pal = pal_img.getpalette()
    if pal is None:
        raise RuntimeError("Could not extract palette from image (getpalette returned None).")

    rgb_list: List[np.ndarray] = []
    cnt_list: List[int] = []

    # getcolors() can return:
    # - (count, index:int) for 'P' images
    # - (count, (r,g,b)) for some modes
    for cnt, col in sorted(colors, key=lambda x: -x[0]):
        rgb01: Optional[np.ndarray] = None

        if isinstance(col, int):
            idx = int(col)
            base = 3 * idx
            if base + 2 >= len(pal):
                continue
            r = pal[base + 0]
            g = pal[base + 1]
            b = pal[base + 2]
            rgb01 = np.array([r, g, b], dtype=np.float32) / 255.0

        elif isinstance(col, tuple) and len(col) >= 3:
            # Could be (r,g,b) or (r,g,b,a)
            r = int(col[0])
            g = int(col[1])
            b = int(col[2])
            rgb01 = np.array([r, g, b], dtype=np.float32) / 255.0

        if rgb01 is None:
            continue

        luma = _rgb_to_luma(rgb01)
        if ignore_near_white and luma > 0.92:
            continue
        if ignore_near_black and luma < 0.06:
            continue

        rgb_list.append(rgb01)
        cnt_list.append(int(cnt))

    if not rgb_list:
        # fallback: do not ignore anything
        for cnt, col in sorted(colors, key=lambda x: -x[0]):
            if isinstance(col, int):
                idx = int(col)
                base = 3 * idx
                if base + 2 >= len(pal):
                    continue
                r = pal[base + 0]
                g = pal[base + 1]
                b = pal[base + 2]
                rgb01 = np.array([r, g, b], dtype=np.float32) / 255.0
                rgb_list.append(rgb01)
                cnt_list.append(int(cnt))
            elif isinstance(col, tuple) and len(col) >= 3:
                r = int(col[0]); g = int(col[1]); b = int(col[2])
                rgb01 = np.array([r, g, b], dtype=np.float32) / 255.0
                rgb_list.append(rgb01)
                cnt_list.append(int(cnt))

    rgb_list, cnt_list = _merge_similar_colors(rgb_list, cnt_list, min_dist=min_merge_dist)

    rgb_list = rgb_list[: max(2, k)]
    rgb_list = sorted(rgb_list, key=_rgb_to_luma)

    return [_hex_from_rgb01(c) for c in rgb_list]

def save_palette_strip(
    hex_colors: List[str],
    path: str = "extracted_palette.png",
    width: int = 1024,
    height: int = 128,
) -> None:
    if len(hex_colors) < 2:
        return
    img = Image.new("RGB", (width, height), (0, 0, 0))
    px = img.load()
    if px is None:
        raise RuntimeError("Failed to load pixel access object from image")
    seg = width / len(hex_colors)

    for x in range(width):
        i = min(len(hex_colors) - 1, int(x / seg))
        rgb01 = _rgb01_from_hex(hex_colors[i])
        r = int(rgb01[0] * 255)
        g = int(rgb01[1] * 255)
        b = int(rgb01[2] * 255)
        for y in range(height):
            px[x, y] = (r, g, b)

    img.save(path)


# -------------------------- Palette mapping + rendering --------------------------

def make_lut(colors: List[str], n: int = 256) -> np.ndarray:
    if len(colors) < 2:
        raise ValueError("Provide at least 2 colors for a palette.")
    stops = np.stack([_rgb01_from_hex(c) for c in colors], axis=0)  # (k,3)
    k = stops.shape[0]
    xs = np.linspace(0.0, 1.0, k, dtype=np.float32)
    xq = np.linspace(0.0, 1.0, n, dtype=np.float32)
    lut = np.empty((n, 3), dtype=np.float32)
    for ch in range(3):
        lut[:, ch] = np.interp(xq, xs, stops[:, ch])
    return lut

def normalize_robust(x: np.ndarray, lo_q: float = 0.02, hi_q: float = 0.98) -> np.ndarray:
    lo = np.quantile(x, lo_q)
    hi = np.quantile(x, hi_q)
    if hi <= lo + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

def sigmoid_contrast(x01: np.ndarray, strength: float = 8.0) -> np.ndarray:
    x = np.clip(x01, 0.0, 1.0)
    return (1.0 / (1.0 + np.exp(-strength * (x - 0.5)))).astype(np.float32)

def field_to_rgb(field01: np.ndarray, lut: np.ndarray) -> np.ndarray:
    idx = np.clip((field01 * (lut.shape[0] - 1)).astype(np.int32), 0, lut.shape[0] - 1)
    rgb = lut[idx]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

def save_png(path: str, rgb_u8: np.ndarray) -> None:
    Image.fromarray(rgb_u8, mode="RGB").save(path)

def save_gif(path: str, frames_rgb_u8: List[np.ndarray], fps: int = 12) -> None:
    imgs = [Image.fromarray(f, mode="RGB") for f in frames_rgb_u8]
    duration_ms = int(1000 / max(1, fps))
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

def resize_field(field: np.ndarray, out_n: int) -> np.ndarray:
    img = Image.fromarray(field.astype(np.float32), mode="F")
    img = img.resize((out_n, out_n), resample=RESAMPLE_BILINEAR)
    return np.array(img, dtype=np.float32)

def make_frames_from_fields(
    fields: List[np.ndarray],
    palette: List[str],
    contrast_strength: float,
    gif_size: Optional[int] = None,
) -> List[np.ndarray]:
    lut = make_lut(palette, n=256)
    frames: List[np.ndarray] = []
    for f in fields:
        if gif_size is not None and f.shape[0] != gif_size:
            f = resize_field(f, gif_size)
        f01 = normalize_robust(f)
        f01 = sigmoid_contrast(f01, strength=contrast_strength)
        frames.append(field_to_rgb(f01, lut))
    return frames


# -------------------------- Gierer–Meinhardt reaction–diffusion (fast rFFT) --------------------------

@dataclass
class GMParams:
    Da: float = 0.02
    Di: float = 1.0
    rho: float = 0.001
    rho_a: float = 0.001
    mu_a: float = 0.02
    mu_i: float = 0.03
    kappa: float = 0.11
    eps: float = 1e-6

def _make_k2_rfft(n: int, dx: float) -> np.ndarray:
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx).astype(np.float32)
    ky = 2.0 * np.pi * np.fft.rfftfreq(n, d=dx).astype(np.float32)
    return ((kx * kx)[:, None] + (ky * ky)[None, :]).astype(np.float32)

def _precompute_diffusion_inv(n: int, dx: float, dt: float, Da: float, Di: float) -> Tuple[np.ndarray, np.ndarray]:
    k2 = _make_k2_rfft(n, dx)
    inv_a = (1.0 / (1.0 + dt * Da * k2)).astype(np.float32)
    inv_i = (1.0 / (1.0 + dt * Di * k2)).astype(np.float32)
    return inv_a, inv_i

def step_gm_fft_rfft(
    a: np.ndarray,
    i: np.ndarray,
    p: GMParams,
    dt: float,
    inv_a: np.ndarray,
    inv_i: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    a2 = a * a
    sat = a2 / (1.0 + p.kappa * a2)

    da = (p.rho / (i + p.eps)) * sat - p.mu_a * a + p.rho_a
    di = p.rho * sat - p.mu_i * i

    a_tmp = a + dt * da
    i_tmp = i + dt * di

    A = np.fft.rfft2(a_tmp)
    I = np.fft.rfft2(i_tmp)

    a_next = np.fft.irfft2(A * inv_a, s=a.shape).astype(np.float32)
    i_next = np.fft.irfft2(I * inv_i, s=i.shape).astype(np.float32)

    np.maximum(a_next, 0.0, out=a_next)
    np.maximum(i_next, 0.0, out=i_next)
    return a_next, i_next

def simulate_camouflage(
    n: int,
    steps: int,
    dt: float,
    dx: float,
    params: GMParams,
    seed: int,
    init_a: Tuple[float, float] = (0.5, 0.10),
    init_i: float = 0.10,
    snapshot_every: int = 60,
    log_every: int = 600,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    rng = np.random.default_rng(seed)

    a = rng.normal(loc=init_a[0], scale=init_a[1], size=(n, n)).astype(np.float32)
    np.maximum(a, 0.0, out=a)
    i = np.full((n, n), init_i, dtype=np.float32)

    inv_a, inv_i = _precompute_diffusion_inv(n=n, dx=dx, dt=dt, Da=params.Da, Di=params.Di)

    snaps: List[np.ndarray] = []
    t0 = time.time()

    for t in range(steps):
        a, i = step_gm_fft_rfft(a, i, params, dt, inv_a, inv_i)

        if snapshot_every and (t % snapshot_every == 0):
            snaps.append(a.copy())

        if log_every and (t % log_every == 0) and t > 0:
            elapsed = time.time() - t0
            it_s = t / max(elapsed, 1e-9)
            print(f"[n={n}] step {t}/{steps}  ({it_s:.1f} it/s)  frames={len(snaps)}")

    return a, i, snaps


# -------------------------- Two-scale blending --------------------------

def blur_fft_periodic(field: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return field.astype(np.float32, copy=False)

    n = field.shape[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0).astype(np.float32)
    ky = 2.0 * np.pi * np.fft.rfftfreq(n, d=1.0).astype(np.float32)
    k2 = (kx * kx)[:, None] + (ky * ky)[None, :]
    H = np.exp(-0.5 * (sigma_px ** 2) * k2).astype(np.float32)

    F = np.fft.rfft2(field.astype(np.float32))
    return np.fft.irfft2(F * H, s=field.shape).astype(np.float32)

def two_scale_blend(
    a_macro: np.ndarray,
    a_micro: np.ndarray,
    micro_weight: float = 0.38,
    macro_blur_px: float = 3.0,
) -> np.ndarray:
    m0 = normalize_robust(a_macro)
    m1 = normalize_robust(a_micro)
    m0b = blur_fft_periodic(m0, sigma_px=macro_blur_px)
    blended = (1.0 - micro_weight) * m0b + micro_weight * m1
    return blended.astype(np.float32)


# -------------------------- Heuristics --------------------------

def contrast_from_palette(hex_colors: List[str]) -> float:
    rgbs = [_rgb01_from_hex(h) for h in hex_colors]
    lumas = sorted(_rgb_to_luma(c) for c in rgbs)
    span = max(1e-6, lumas[-1] - lumas[0])
    t = float(np.clip((span - 0.2) / (0.9 - 0.2), 0.0, 1.0))
    return float(7.0 + 5.0 * t)


# -------------------------- CLI / main --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Turing-pattern camouflage from a biome image (dominant colors).")

    p.add_argument("--biome", required=True, help="Path to a biome image (jpg/png/etc).")
    p.add_argument("--colors", type=int, default=4, help="How many dominant colors to extract (2..8 typical).")
    p.add_argument("--sample_size", type=int, default=256, help="Downsample size for palette extraction.")
    p.add_argument("--crop_fraction", type=float, default=0.90, help="Center-crop fraction for extraction (0.2..1.0).")
    p.add_argument("--merge_dist", type=float, default=0.08, help="Merge threshold for similar colors (RGB01 distance).")
    p.add_argument("--keep_white", action="store_true", help="Do not ignore near-white colors during extraction.")
    p.add_argument("--ignore_black", action="store_true", help="Ignore near-black colors during extraction.")

    p.add_argument("--final_res", type=int, default=2048, help="Final PNG resolution (upscaled).")
    p.add_argument("--gif_res", type=int, default=640, help="GIF resolution (smaller = smaller file).")
    p.add_argument("--fps", type=int, default=24, help="GIF frames per second.")

    p.add_argument("--macro_n", type=int, default=384, help="Macro simulation grid size.")
    p.add_argument("--micro_n", type=int, default=640, help="Micro simulation grid size.")
    p.add_argument("--steps", type=int, default=14000, help="Simulation steps per scale.")
    p.add_argument("--snapshot_every", type=int, default=60, help="Snapshot every N steps (smaller => more GIF frames).")

    p.add_argument("--micro_weight", type=float, default=0.38, help="0..1, micro detail contribution.")
    p.add_argument("--macro_blur", type=float, default=3.0, help="Blur macro field before blending (pixels).")

    p.add_argument("--dt", type=float, default=1.0, help="Time step.")
    p.add_argument("--dx", type=float, default=1.0, help="Grid spacing.")
    p.add_argument("--log_every", type=int, default=600, help="Progress log interval (steps).")

    return p.parse_args()

def main() -> None:
    args = parse_args()

    palette = extract_dominant_palette(
        image_path=args.biome,
        k=max(2, int(args.colors)),
        sample_size=int(args.sample_size),
        crop_fraction=float(args.crop_fraction),
        min_merge_dist=float(args.merge_dist),
        ignore_near_white=(not args.keep_white),
        ignore_near_black=bool(args.ignore_black),
    )

    print("Extracted palette:", palette)
    save_palette_strip(palette, path="extracted_palette.png")

    contrast = contrast_from_palette(palette)
    print(f"Auto contrast strength: {contrast:.2f}")

    base = GMParams()

    p_macro = GMParams(**base.__dict__)
    p_micro = GMParams(**base.__dict__)
    p_macro.Di = base.Di * 1.35
    p_micro.Di = base.Di * 0.90

    n_macro = int(args.macro_n)
    n_micro = int(args.micro_n)

    if n_macro <= 32 or n_micro <= 32:
        raise ValueError("macro_n and micro_n must be > 32.")

    steps = int(args.steps)
    snapshot_every = int(args.snapshot_every)

    print("Running macro simulation...")
    _, _, snaps_macro = simulate_camouflage(
        n=n_macro,
        steps=steps,
        dt=float(args.dt),
        dx=float(args.dx),
        params=p_macro,
        seed=2,
        snapshot_every=snapshot_every,
        log_every=int(args.log_every),
    )

    print("Running micro simulation...")
    _, _, snaps_micro = simulate_camouflage(
        n=n_micro,
        steps=steps,
        dt=float(args.dt),
        dx=float(args.dx),
        params=p_micro,
        seed=7,
        snapshot_every=snapshot_every,
        log_every=int(args.log_every),
    )

    n_frames = min(len(snaps_macro), len(snaps_micro))
    snaps_macro = snaps_macro[:n_frames]
    snaps_micro = snaps_micro[:n_frames]

    micro_weight = float(args.micro_weight)
    macro_blur = float(args.macro_blur)

    print("Blending scales...")
    blended_snaps: List[np.ndarray] = []
    for am, ai in zip(snaps_macro, snaps_micro):
        am_up = resize_field(am, out_n=n_micro)
        blended = two_scale_blend(am_up, ai, micro_weight=micro_weight, macro_blur_px=macro_blur)
        blended_snaps.append(blended)

    final_res = int(args.final_res)
    final_field = blended_snaps[-1]
    if final_field.shape[0] != final_res:
        final_field = resize_field(final_field, out_n=final_res)

    lut = make_lut(palette, n=256)
    final01 = sigmoid_contrast(normalize_robust(final_field), strength=contrast)
    final_rgb = field_to_rgb(final01, lut)
    save_png("camouflage_final.png", final_rgb)

    gif_res = int(args.gif_res) if int(args.gif_res) > 0 else None
    fps = int(args.fps)

    print("Rendering GIF...")
    frames = make_frames_from_fields(blended_snaps, palette=palette, contrast_strength=contrast, gif_size=gif_res)
    save_gif("camouflage_evolution.gif", frames, fps=fps)

    print("Wrote: extracted_palette.png")
    print(f"Wrote: camouflage_final.png ({final_res}x{final_res})")
    if gif_res is None:
        print(f"Wrote: camouflage_evolution.gif ({len(frames)} frames @ {fps} fps, gif_res=full)")
    else:
        print(f"Wrote: camouflage_evolution.gif ({len(frames)} frames @ {fps} fps, gif_res={gif_res})")

if __name__ == "__main__":
    main()
