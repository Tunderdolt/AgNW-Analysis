"""
clean_images.py
===============
Produces clean versions of SEM and VLM images with:
  - SEM: instrument annotation bar removed, clean scale bar overlaid
  - VLM: existing scale bar replaced with a clean one

Images are auto-detected by extension (.tif/.tiff = SEM, .png/.jpg = VLM).
VLM magnification is read from the filename (e.g. "50x", "100x").

Usage
-----
    # Single image
    python clean_images.py image.tif

    # Multiple images
    python clean_images.py *.tif *.png

    # Whole folder
    python clean_images.py --input_dir ./images/ --output_dir ./clean/

    # Override scale if filename has no magnification
    python clean_images.py mystery.png --vlm_scale_um 20.0

    # Customise scale bar appearance
    python clean_images.py image.tif --bar_color white --bar_height 8 --font_scale 1.5
"""

import argparse
import sys
import re
from pathlib import Path

import numpy as np
import cv2


# ---- Scale bar detection (adapted from nanowire_analysis.py) ----------------

def detect_scale_bar_sem(img, scale_nm=10_000.0, fallback_nm_per_px=25.0):
    """Return (nm_per_px, bar_start_row) for a SEM image."""
    h = img.shape[0]
    region = img[int(h * 0.80):, :].astype(np.uint8)
    bin_r  = (region > 180).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(bin_r, connectivity=8)
    candidates = []
    for i in range(1, len(stats)):
        w  = stats[i, cv2.CC_STAT_WIDTH]
        ht = stats[i, cv2.CC_STAT_HEIGHT]
        if w / max(ht, 1) > 8 and w > 20:
            implied = scale_nm / max(w, 1)
            score   = abs(implied - fallback_nm_per_px) / max(fallback_nm_per_px, 1)
            candidates.append((score, w))
    if not candidates:
        raise RuntimeError("SEM scale bar not found.")
    best_w = min(candidates, key=lambda x: x[0])[1]

    # Find topmost black row in bottom 30% -> that is the annotation bar top
    row_means     = img.mean(axis=1)
    BLACK_THRESH  = 8
    in_black      = False
    best_black_top = h
    for r in range(h - 1, int(h * 0.70), -1):
        if row_means[r] < BLACK_THRESH:
            if not in_black:
                in_black = True
        else:
            if in_black:
                best_black_top = min(best_black_top, r + 2)
                in_black = False
    if in_black:
        best_black_top = min(best_black_top, int(h * 0.70) + 2)

    bar_start = best_black_top if best_black_top < h else int(h * 0.80)
    return scale_nm / best_w, bar_start


VLM_SCALE_TABLE = {
    20:  (100.0, 366.0),   # (scale_bar_um, nm_per_px)
    50:  (50.0,  146.4),
    100: (20.0,   73.2),
}

def detect_scale_vlm(path: Path, override_um: float = None):
    """Return nm_per_px for a VLM image from filename magnification."""
    stem = path.stem
    # Extract largest numeric token followed by x
    stem2 = re.sub(r'(\d+)([xX])(?=\D|$)', r'\1x ', stem)
    mags  = []
    for tok in re.split(r'[\s_\-]+', stem2):
        if re.fullmatch(r'\d{2,5}x?', tok.strip(), re.IGNORECASE):
            mags.append(int(re.sub(r'x$', '', tok.strip(), flags=re.IGNORECASE)))
    mag = max(mags) if mags else None

    if override_um is not None and mag is not None:
        # Use filename mag to get nm_per_px from table, ignore override_um
        if mag in VLM_SCALE_TABLE:
            _, nm_per_px = VLM_SCALE_TABLE[mag]
            bar_um = VLM_SCALE_TABLE[mag][0]
            return nm_per_px, bar_um
    if mag in (VLM_SCALE_TABLE or {}):
        nm_per_px, bar_um = VLM_SCALE_TABLE[mag][1], VLM_SCALE_TABLE[mag][0]
        return nm_per_px, bar_um
    if override_um is not None:
        # User provided scale bar length in um; estimate nm_per_px from image width
        # (assume scale bar is ~15% of image width at 100x as a rough prior)
        return None, override_um   # nm_per_px determined later
    raise RuntimeError(
        f"Cannot determine VLM scale for {path.name}. "
        "Add magnification (e.g. '100x') to the filename or use --vlm_scale_um."
    )


# ---- Scale bar drawing -------------------------------------------------------

def draw_scale_bar(img_rgb: np.ndarray,
                   nm_per_px: float,
                   bar_nm: float,
                   color: tuple = (255, 255, 255),
                   position: str = "bottom-left",
                   bar_height: int = 6,
                   font_scale: float = 1.2,
                   margin_frac: float = 0.03) -> np.ndarray:
    """
    Draw a clean scale bar with label onto img_rgb (modified in place copy).

    Parameters
    ----------
    nm_per_px  : physical scale
    bar_nm     : length of scale bar in nm
    color      : RGB tuple
    position   : 'bottom-left' or 'bottom-right'
    bar_height : thickness of bar in pixels
    font_scale : cv2 font scale
    margin_frac: margin from edge as fraction of image width
    """
    out = img_rgb.copy()
    h, w = out.shape[:2]
    margin = max(20, int(w * margin_frac))

    bar_px = int(round(bar_nm / nm_per_px))

    # Label text
    if bar_nm >= 1000:
        label = f"{bar_nm/1000:.0f} um"
    else:
        label = f"{bar_nm:.0f} nm"

    font      = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale * 1.5))
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Vertical positions
    bar_y2 = h - margin
    bar_y1 = bar_y2 - bar_height
    text_y = bar_y1 - 8

    # Horizontal positions
    if position == "bottom-right":
        bar_x1 = w - margin - bar_px
    else:
        bar_x1 = margin
    bar_x2 = bar_x1 + bar_px
    text_x = bar_x1 + (bar_px - tw) // 2

    # Shadow for visibility on any background
    shadow = (0, 0, 0) if color != (0, 0, 0) else (200, 200, 200)
    for dx, dy in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
        cv2.rectangle(out, (bar_x1+dx, bar_y1+dy), (bar_x2+dx, bar_y2+dy),
                      shadow, -1)
        cv2.putText(out, label, (text_x+dx, text_y+dy),
                    font, font_scale, shadow, thickness + 1, cv2.LINE_AA)

    # Main bar and label
    cv2.rectangle(out, (bar_x1, bar_y1), (bar_x2, bar_y2), color, -1)
    cv2.putText(out, label, (text_x, text_y),
                font, font_scale, color, thickness, cv2.LINE_AA)

    return out


def choose_bar_length(nm_per_px: float, img_w: int,
                      target_frac: float = 0.20) -> float:
    """
    Pick a round scale bar length (in nm) that is approximately
    target_frac of the image width.
    """
    target_nm = nm_per_px * img_w * target_frac
    # Round to nearest nice number
    for nice in [100, 200, 500, 1000, 2000, 5000, 10000, 20000,
                 50000, 100000, 200000, 500000]:
        if nice >= target_nm * 0.5:
            return float(nice)
    return target_nm


# ---- Per-image processing ---------------------------------------------------

def process_sem(path: Path, out_dir: Path, args) -> Path:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Try tifffile for 16-bit TIFs
        import tifffile
        raw = tifffile.imread(str(path))
        if raw.dtype != np.uint8:
            raw = ((raw - raw.min()) / max(raw.max() - raw.min(), 1) * 255).astype(np.uint8)
        img = raw

    nm_per_px, bar_start = detect_scale_bar_sem(
        img,
        scale_nm=args.sem_scale_nm,
        fallback_nm_per_px=args.fallback_nm_per_px,
    )
    print(f"  SEM: {nm_per_px:.2f} nm/px  |  annotation bar at row {bar_start}")

    # Crop out the annotation bar
    img_crop = img[:bar_start, :]

    # Convert to RGB for drawing
    rgb = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2RGB)

    # Choose and draw scale bar
    bar_nm = choose_bar_length(nm_per_px, img_crop.shape[1])
    color  = (255, 255, 255) if args.bar_color == "white" else (0, 0, 0)
    rgb    = draw_scale_bar(rgb, nm_per_px, bar_nm,
                            color=color,
                            position=args.bar_position,
                            bar_height=args.bar_height,
                            font_scale=args.font_scale)

    out_path = out_dir / f"{path.stem}_clean.png"
    cv2.imwrite(str(out_path), rgb)
    print(f"  Saved: {out_path.name}  ({img_crop.shape[1]}x{img_crop.shape[0]} px)")
    return out_path


def process_vlm(path: Path, out_dir: Path, args) -> Path:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot read {path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]

    nm_per_px, bar_um = detect_scale_vlm(path, override_um=args.vlm_scale_um)

    if nm_per_px is None:
        # Override mode without table lookup — estimate nm_per_px from bar_um
        # and assume the existing scale bar is ~15% of image width
        bar_px_est = w * 0.15
        nm_per_px  = (bar_um * 1000.0) / bar_px_est
    bar_nm = bar_um * 1000.0

    print(f"  VLM: {nm_per_px:.2f} nm/px  |  scale bar = {bar_um:.0f} um")

    # Blank out the existing scale bar region (bottom 10% of image)
    # by filling with the median colour of a safe area
    safe_region   = img_rgb[int(h*0.40):int(h*0.80), int(w*0.10):int(w*0.90)]
    median_colour = np.median(safe_region.reshape(-1, 3), axis=0).astype(np.uint8)
    img_rgb[int(h*0.88):, :] = median_colour

    # Draw new scale bar
    color = (255, 255, 255) if args.bar_color == "white" else (0, 0, 0)
    img_rgb = draw_scale_bar(img_rgb, nm_per_px, bar_nm,
                             color=color,
                             position=args.bar_position,
                             bar_height=args.bar_height,
                             font_scale=args.font_scale)

    out_path = out_dir / f"{path.stem}_clean.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {out_path.name}  ({w}x{h} px)")
    return out_path


# ---- CLI --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Clean SEM/VLM images: remove instrument bar, add readable scale bar.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("images", nargs="*", type=Path,
                    help="Image file(s) to process")
    ap.add_argument("--input_dir", type=Path, default=None,
                    help="Process all images in this folder")
    ap.add_argument("--output_dir", type=Path, default=None,
                    help="Output folder (default: 'clean/' next to each image)")
    ap.add_argument("--sem_scale_nm", type=float, default=10000.0,
                    help="Physical length of SEM scale bar in nm")
    ap.add_argument("--fallback_nm_per_px", type=float, default=25.0,
                    help="Expected SEM nm/px (used to score scale bar candidates)")
    ap.add_argument("--vlm_scale_um", type=float, default=None,
                    help="VLM scale bar length in um (overrides filename detection)")
    ap.add_argument("--bar_color", choices=["white","black"], default="white",
                    help="Scale bar colour")
    ap.add_argument("--bar_position", choices=["bottom-left","bottom-right"],
                    default="bottom-left", help="Scale bar position")
    ap.add_argument("--bar_height", type=int, default=6,
                    help="Scale bar thickness in pixels")
    ap.add_argument("--font_scale", type=float, default=1.2,
                    help="Scale bar label font size")
    args = ap.parse_args()

    # Collect input files
    paths = list(args.images)
    if args.input_dir:
        for ext in ["*.tif","*.tiff","*.TIF","*.TIFF","*.png","*.jpg","*.jpeg"]:
            paths.extend(args.input_dir.glob(ext))
    if not paths:
        ap.print_help()
        sys.exit(1)

    n_ok = n_fail = 0
    for path in sorted(set(paths)):
        if not path.exists():
            print(f"[SKIP] Not found: {path}")
            continue

        out_dir = args.output_dir or (path.parent / "clean")
        out_dir.mkdir(parents=True, exist_ok=True)

        ext = path.suffix.lower()
        print(f"\n{path.name}")

        try:
            if ext in (".tif", ".tiff"):
                process_sem(path, out_dir, args)
            elif ext in (".png", ".jpg", ".jpeg"):
                process_vlm(path, out_dir, args)
            else:
                print(f"  [SKIP] Unrecognised extension: {ext}")
                continue
            n_ok += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            n_fail += 1

    print(f"\nDone: {n_ok} saved, {n_fail} failed.")
    if args.output_dir:
        print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()
