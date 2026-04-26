"""
nanowire_analysis.py  v14 — Nano1D hybrid
==========================================
Measures AgNW length (and diameter for SEM) from SEM TIF and VLM PNG images.

Architecture
------------
  Preprocessing : Frangi vesselness filter (β=0.3, σ=2–4) at half-res (SEM) or full-res
                  (VLM), giving a cleaner binary than CLAHE+Otsu alone.
  Skeleton      : Zhang algorithm → Nano1D neighbour-count image (Eqs 1-4).
                  N=1 tails, N=2 sections, N≥3 intersections. N=4 centres removed.
  Gap bridging  : Two-pass KD-tree bridging of facing tails within max_gap_um.
  Tracing       : Nano1D pixel-by-pixel recall-vector algorithm.
                  At intersections the last `search_distance` pixels form a rolling
                  tangent estimate (recall vector); the neighbour most aligned with it
                  is chosen.  This is far more robust than single-segment PCA directions.
  Filters       : Chord-deviation ratio > max_chord_ratio rejects V-kinks.
  Outputs       : Per-wire CSV (length, diameter [SEM], sinuosity, orientation, truncated),
                  length distribution + orientation rose plots, false-colour overlay, and
                  a sinuosity field for downstream post-filtering.

Reference: Moradpur-Tari et al., Ultramicroscopy 261 (2024) 113949.

SEM pipeline  : length + per-wire diameter (FWHM of greyscale perpendicular profile).
VLM pipeline  : length only (optical resolution too coarse for reliable diameter).
Coverage mode : binary fill fraction for 20× images where wires are sub-pixel.

Usage
-----
    python nanowire_analysis.py *.tif *.png --output_dir results/
    python nanowire_analysis.py --input_dir /path/to/files --output_dir results/
    python nanowire_analysis.py img.tif --max_chord_ratio 0.20   # tighter kink filter
    python nanowire_analysis.py img.tif --search_distance 20     # shorter recall vector
    python nanowire_analysis.py img.tif --no_frangi              # raw CLAHE+Otsu (faster)

VLM scale is auto-calibrated from the filename magnification (50x, 100x etc.).
Use --vlm_scale_um only to override for images without magnification in their name.

Requirements
------------
    pip install tifffile pillow numpy scipy scikit-image matplotlib opencv-python-headless
"""

import argparse, csv, math, random, sys, warnings
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from skimage import filters, measure
from skimage.filters import frangi
from skimage.measure import label as sk_label
from skimage.morphology import remove_small_objects, skeletonize

warnings.filterwarnings("ignore")

DOWNSAMPLE_SEM = 2
DOWNSAMPLE_VLM = 1

CSV_FIELDS_SEM = ["image","mode","length_nm","length_um","diameter_nm","aspect_ratio",
                   "n_segments","ep_to_ep","truncated","sinuosity","wire_angle_deg","centroid_row","centroid_col"]
CSV_FIELDS_VLM = ["image","mode","length_nm","length_um",
                   "n_segments","ep_to_ep","truncated","sinuosity","wire_angle_deg","centroid_row","centroid_col"]




# ─── ORIENTATION ANALYSIS ────────────────────────────────────────────────────

def orientation_stats(angles_deg: np.ndarray) -> dict:
    """Test whether a set of orientations (undirected, in [0, 180°)) is
    significantly non-uniform using the Rayleigh test and a chi-squared
    goodness-of-fit against a uniform distribution.

    The Rayleigh test (on doubled angles) is the standard test for a single
    preferred direction in undirected circular data.  The chi-squared test
    is complementary — it is sensitive to multimodal or any non-uniform
    pattern even when the Rayleigh test has low power (e.g. bimodal).

    Returns a dict with:
      n_angles         : number of orientations tested
      mean_direction_deg : Rayleigh mean direction [0, 180°)
      mean_resultant_R   : mean resultant length [0, 1]; 0=isotropic, 1=fully aligned
      r_strength         : verbal description of R ('negligible'/'weak'/'moderate'/'strong')
      rayleigh_p         : p-value; < 0.05 => reject isotropy
      chi2_stat          : chi-squared statistic (12 equal bins)
      chi2_p             : chi-squared p-value
      significant        : True if either test is significant at α=0.05
      summary_line       : one-line human-readable summary
    """
    from scipy import stats as _stats

    angles_deg = np.asarray(angles_deg, dtype=float)
    n = len(angles_deg)
    if n < 10:
        return {"n_angles": n, "significant": False,
                "summary_line": f"n={n} (too few for reliable test)"}

    # ── Rayleigh test (doubled angles for undirected data) ────────────────
    phi = 2.0 * np.radians(angles_deg)
    C, S = np.mean(np.cos(phi)), np.mean(np.sin(phi))
    R = float(np.sqrt(C**2 + S**2))
    mean_dir = float(np.degrees(np.arctan2(S, C)) / 2.0 % 180.0)
    Z = n * R**2
    # Zar (1999) approximation, accurate for all n
    p_ray = float(np.clip(
        np.exp(-Z) * (1 + (2*Z - Z**2) / (4*n)
                       - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*n**2)),
        0.0, 1.0))

    # ── Chi-squared vs uniform (12 equal bins) ─────────────────────────────
    N_BINS = 12
    counts, _ = np.histogram(angles_deg, bins=N_BINS, range=(0, 180))
    expected = np.full(N_BINS, n / N_BINS)
    chi2, p_chi = _stats.chisquare(counts, f_exp=expected)

    # Verbal effect-size label for R
    if R < 0.10:   r_str = "negligible"
    elif R < 0.30: r_str = "weak"
    elif R < 0.60: r_str = "moderate"
    else:          r_str = "strong"

    # Circular std (radians -> degrees, halved for undirected)
    circ_std = float(np.degrees(np.sqrt(max(-2.0 * np.log(max(R, 1e-12)), 0.0))) / 2.0)

    sig = (p_ray < 0.05) or (p_chi < 0.05)

    if sig:
        note = (f"Preferred direction {mean_dir:.0f}° "
                f"({r_str} anisotropy, R={R:.3f}); "
                f"Rayleigh p={p_ray:.4f}, χ²={chi2:.1f} p={p_chi:.4f}")
    else:
        note = (f"No significant preferred direction "
                f"(R={R:.3f} [{r_str}]); "
                f"Rayleigh p={p_ray:.4f}, χ²={chi2:.1f} p={p_chi:.4f}")

    return {
        "n_angles":            n,
        "mean_direction_deg":  round(mean_dir, 1),
        "mean_resultant_R":    round(R, 4),
        "circ_std_deg":        round(circ_std, 1),
        "r_strength":          r_str,
        "rayleigh_p":          round(p_ray, 4),
        "chi2_stat":           round(float(chi2), 2),
        "chi2_p":              round(float(p_chi), 4),
        "significant":         sig,
        "summary_line":        note,
    }

def wire_sinuosity(coords_raw: np.ndarray, downsample: int,
                    nm_per_px: float) -> float:
    """Compute the sinuosity (arc/chord ratio) of a traced wire.

    Sinuosity = 1.0 for a perfectly straight wire and increases with curvature
    or kinking.  Physical upper bound for a smooth arc: a semicircle gives
    sinuosity = π/2 ≈ 1.57.  Any value above ~1.5 indicates a non-physical
    V- or Z-shaped kink rather than genuine wire curvature.

    Implementation notes
    --------------------
    1. Coordinates are sorted by their projection onto the wire's principal axis
       (SVD). This corrects for the segment-concatenation zig-zag that occurs
       when adjacent segments have opposing PCA directions.
    2. Coords are sampled every 5 skeleton pixels to remove the pixel-grid
       staircase artefact from the binary skeleton.

    Parameters
    ----------
    coords_raw : (N, 2) array of skeleton pixel coordinates (row, col)
    downsample : skeleton downsampling factor relative to original image
    nm_per_px  : physical scale in nm per original pixel

    Returns
    -------
    float : sinuosity ≥ 1.0; returns 1.0 for very short wires (<6 points)
    """
    c = coords_raw.astype(float) * downsample * nm_per_px / 1000.0  # µm
    if len(c) < 6:
        return 1.0
    # Sort by projection onto principal axis to fix multi-segment ordering
    centered = c - c.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        axis = Vt[0]
    except (np.linalg.LinAlgError, ValueError):
        return 1.0
    proj = centered @ axis
    c_sorted = c[np.argsort(proj)]
    # Sample every 5 pixels to remove pixel-grid zig-zag
    stride = max(1, len(c_sorted) // 60)
    cs = c_sorted[::stride]
    chord = cs[-1] - cs[0]
    chord_len = float(np.sqrt((chord ** 2).sum()))
    if chord_len < 0.1:
        return 999.0  # degenerate (circular path)
    arc = float(np.sqrt((np.diff(cs, axis=0) ** 2).sum(axis=1)).sum())
    return arc / chord_len



# ─── VLM SCALE BAR CALIBRATION ───────────────────────────────────────────────
# Reference nm/px values from confirmed images at each magnification.
# Used to detect and correct scale bar label errors caused by microscope and
# software magnification settings being out of sync.
VLM_CALIBRATION = {
    20:  366.3,   # A_20x_13_1.png  : 100µm bar / 273px interior
    50:  144.9,   # 50x_C_2-2_clean : 20µm bar  / 138px interior
    100:  72.46,  # 100x_E_2-2_clean: 10µm bar  / 138px interior
}
VLM_CALIB_TOLERANCE = 0.20  # warn + correct if >20% from calibrated value


def parse_mag_from_filename(path) -> int | None:
    """Extract magnification integer from filename (e.g. '50x' -> 50)."""
    import re
    m = re.search(r"(\d+)x", path.stem, re.IGNORECASE)
    return int(m.group(1)) if m else None


def validate_vlm_scale(measured_nm_per_px: float, path,
                        calib: dict = VLM_CALIBRATION,
                        tol: float = VLM_CALIB_TOLERANCE) -> tuple:
    """Cross-check measured nm/px against filename magnification calibration.

    If the filename contains a known magnification (e.g. '50x') and the
    scale bar label gives a nm/px that differs from the calibrated value
    by more than `tol`, the calibrated value is used instead.

    Returns (nm_per_px, status) where status is 'ok', 'corrected', or 'unknown'.
    """
    mag = parse_mag_from_filename(path)
    if mag is None or mag not in calib:
        return measured_nm_per_px, "unknown"
    expected = calib[mag]
    ratio = measured_nm_per_px / expected
    if abs(ratio - 1.0) > tol:
        return expected, "corrected"
    return measured_nm_per_px, "ok"

# ─── SCALE BAR ───────────────────────────────────────────────────────────────

def detect_scale_bar_sem(img, scale_nm=10_000.0,
                          fallback_nm_per_px=25.0):
    """Detect the SEM scale bar and return (nm_per_px, instrument_bar_row).

    Candidate components are scored by how closely their implied nm/px
    matches `fallback_nm_per_px` (the expected physical scale).  This makes
    the detection robust against bright wire bundles or particles that happen
    to be wider than the scale bar.
    """
    h = img.shape[0]
    region = img[int(h*0.80):, :].astype(np.uint8)
    bin_r = (region > 180).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(bin_r, connectivity=8)
    # Collect high-aspect candidates
    candidates = []
    for i in range(1, len(stats)):
        w = stats[i, cv2.CC_STAT_WIDTH]; ht = stats[i, cv2.CC_STAT_HEIGHT]
        if w / max(ht, 1) > 8 and w > 20:
            implied_nm_per_px = scale_nm / max(w, 1)
            score = abs(implied_nm_per_px - fallback_nm_per_px) / max(fallback_nm_per_px, 1)
            candidates.append((score, w))
    if not candidates:
        raise RuntimeError("SEM scale bar not found.")
    # Best candidate = closest implied nm/px to the fallback expectation
    best_w = min(candidates, key=lambda x: x[0])[1]
    row_means = img.mean(axis=1)
    # SEM info bars often contain a black annotation box (mean ≈ 0) sitting
    # above the scale-bar graphics.  Find the topmost black zone in the
    # bottom 30% of the image — that is the true start of the info region.
    BLACK_THRESH = 8
    in_black = False
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

    if best_black_top < h:
        bar_start = best_black_top          # crop just above the black box
    else:
        bar_start = int(h * 0.80)           # fallback: simple bottom crop
        for r in range(h - 1, int(h * 0.80), -1):
            if row_means[r] > 30:
                bar_start = r + 1
                break
    return scale_nm / best_w, bar_start


def detect_scale_bar_vlm(img_grey, label_um):
    h, w = img_grey.shape
    search = img_grey[int(h*0.90):, int(w*0.70):]
    white = (search > 180).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    best_w = 0
    for i in range(1, len(stats)):
        sw=stats[i,cv2.CC_STAT_WIDTH]; sh=stats[i,cv2.CC_STAT_HEIGHT]
        if stats[i,cv2.CC_STAT_AREA]>300 and sw/max(sh,1)>1.5 and sw>best_w:
            best_w = sw
    if best_w == 0:
        raise RuntimeError("VLM scale bar box not found.")
    return (label_um*1000)/max(best_w-8,1), int(h*0.92)


# ─── PREPROCESSING ───────────────────────────────────────────────────────────

def make_binary_sem(img, clahe_clip, thresh_frac,
                     use_frangi=True, frangi_sigmas=(2,4), frangi_thresh=0.01,
                     downsample=2):
    """Binary mask for SEM images.

    With use_frangi=True (default): applies Frangi vesselness filter at `downsample`
    resolution, which dramatically reduces junction blobs and improves skeleton quality.
    The Frangi binary is returned at the downsampled resolution.

    With use_frangi=False: falls back to CLAHE+Otsu at full resolution.

    In both cases a full-res CLAHE+Otsu binary is also computed and returned
    for use in diameter measurement (dist transform).
    """
    # 3×3 median filter: removes single-pixel hot/dead detector pixels common in
    # SEM images without blurring wire edges (unlike Gaussian).  Negligible cost.
    img_clean = cv2.medianBlur(img, 3)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(64,64))
    eq = clahe.apply(img_clean)
    binary_clahe = (eq > filters.threshold_otsu(eq)*thresh_frac).astype(np.uint8)

    if use_frangi:
        # Bilateral filter at full resolution before downsampling:
        # edge-preserving smoothing that reduces SEM speckle without blurring wires.
        img_bilateral = cv2.bilateralFilter(
            img_clean, d=7, sigmaColor=20, sigmaSpace=5)
        img_ds = img_bilateral[::downsample, ::downsample].astype(float) / 255.0
        sig_range = range(int(frangi_sigmas[0]), int(frangi_sigmas[1])+1)
        f_out = frangi(img_ds, sigmas=sig_range, black_ridges=False,
                       alpha=0.5, beta=0.3, gamma=None)
        binary_skel = (f_out > frangi_thresh).astype(np.uint8)
        return binary_skel, binary_clahe   # (half-res skeleton binary, full-res clahe binary)
    else:
        # Downsample CLAHE binary for skeleton
        binary_ds = binary_clahe[::downsample, ::downsample]
        return binary_ds, binary_clahe


def make_binary_vlm(img, clahe_clip, thresh_val,
                     use_frangi=True, frangi_sigmas=(1,3), frangi_thresh=0.005):
    """Binary mask for VLM images (full resolution, no downsampling).

    With use_frangi=True: Frangi vesselness filter; much cleaner wire segmentation.
    With use_frangi=False: CLAHE + fixed threshold (original behaviour).
    """
    if use_frangi:
        img_float = img.astype(float) / 255.0
        sig_range = range(int(frangi_sigmas[0]), int(frangi_sigmas[1])+1)
        f_out = frangi(img_float, sigmas=sig_range, black_ridges=False,
                       alpha=0.5, beta=0.3, gamma=None)
        return (f_out > frangi_thresh).astype(np.uint8)
    else:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(32,32))
        eq = clahe.apply(img)
        return (eq > thresh_val).astype(np.uint8)


# ─── SKELETON ────────────────────────────────────────────────────────────────

def build_skeleton(binary_full, downsample):
    if downsample > 1:
        bh = binary_full[::downsample, ::downsample].astype(bool)
    else:
        bh = binary_full.astype(bool)
    try:    bh = remove_small_objects(bh, max_size=8)
    except TypeError: bh = remove_small_objects(bh, min_size=9)
    skel = skeletonize(bh)
    kernel = np.ones((3,3), dtype=np.uint8)
    nc = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
    # degree 1 = endpoint, degree 3+ = junction
    endpoint_mask_raw = skel & (nc == 2)
    junction_mask     = skel & (nc >= 4)
    junc_labeled, _   = measure.label(junction_mask, return_num=True, connectivity=2)
    # Distance from every pixel to nearest junction (for endpoint quality filter)
    junc_dist = distance_transform_edt(~junction_mask)
    # Also expose the raw frame-edge pixels so callers can flag truncated wires
    H_sk, W_sk = skel.shape
    frame_mask = np.zeros(skel.shape, dtype=bool)
    frame_mask[0, :]   |= skel[0, :]
    frame_mask[-1, :]  |= skel[-1, :]
    frame_mask[:, 0]   |= skel[:, 0]
    frame_mask[:, -1]  |= skel[:, -1]
    return skel, endpoint_mask_raw, junction_mask, junc_labeled, junc_dist, frame_mask


def wire_length_heatmap(wire_data, skel_shape, downsample, nm_per_px,
                         sigma_um=5.0):
    """Gaussian-smoothed heatmap of wire length density across the image field.

    Each traced wire contributes its length_um to the pixel at its centroid.
    The result is smoothed so that sparse regions read near zero and dense
    wire clusters read high — a scientifically informative spatial map.
    Returns a 2-D float array in the skeleton coordinate frame.
    """
    H, W = skel_shape
    length_map = np.zeros((H, W), dtype=np.float32)
    for wd in wire_data:
        cen = wd.get("centroid")
        if cen is None: continue
        r, c = int(np.clip(cen[0], 0, H-1)), int(np.clip(cen[1], 0, W-1))
        length_map[r, c] += wd.get("length_um", 0.0)
    sigma_px = sigma_um * 1000.0 / (nm_per_px * downsample)
    return gaussian_filter(length_map, sigma=sigma_px)

# ─── POST-TRACING WIRE MERGE ─────────────────────────────────────────────────

def merge_wire_fragments(wires: list, nm_per_px: float, downsample: int,
                          max_gap_um: float = 1.0,
                          cone_angle_deg: float = 45.0,
                          max_join_angle_deg: float = 35.0,
                          max_chord_ratio: float = 0.25,
                          max_sinuosity: float = 1.4,
                          search_distance: int = 30,
                          border_um: float = 0.5,
                          skel_shape: tuple = (0, 0),
                          junc_dist: np.ndarray = None,
                          n_passes: int = 2) -> list:
    """Greedy post-tracing wire merge.

    At every junction the Nano1D tracer picks one arm and discards the others.
    If the recall vector chose incorrectly, the true wire is split into two
    fragments: one ending inside the junction blob and another starting from
    the other side.  This pass finds such pairs and reassembles them.

    Algorithm (per pass):
      1. EDGE EXCLUSION: endpoints within `border_um` of the image frame are
         excluded from the merge endpoint pool.  This prevents wires from being
         stitched together along the frame boundary.
      2. PROXIMITY CONE: for each endpoint pair within `max_gap_um`, the gap
         vector must fall within `cone_angle_deg` of *both* endpoints' outgoing
         recall directions.  This is the directional cone — the other endpoint
         must lie roughly in the direction the wire is heading.
      3. JOIN CURVATURE: the angle between the two wires' outgoing directions
         at the join point must be ≤ `max_join_angle_deg`.  Concretely this is
         angle(ei.dir, −ej.dir): a valid smooth continuation has these nearly
         parallel (≈ 0°); a kink at the join gives a large angle.  This is the
         key constraint that prevents over-connectivity.
      4. SHAPE VALIDITY: the merged coordinate array must satisfy
         chord_ratio ≤ max_chord_ratio AND sinuosity ≤ max_sinuosity.
      Greedy accept from longest to shortest; repeat for `n_passes` iterations.
    """
    from scipy.spatial import cKDTree

    max_gap_px    = max_gap_um  * 1000.0 / (nm_per_px * downsample)
    border_px     = border_um   * 1000.0 / (nm_per_px * downsample)
    H_sk, W_sk    = skel_shape

    # Pre-compute endpoint junction distance threshold in skeleton pixels.
    # Endpoints within this distance of a junction stopped because of it
    # (junction-split fragment). Genuine wire tips are much farther away.
    # 300nm physical ≈ 6px at SEM half-res, ≈ 4px at VLM 100x.
    _ep_junc_thresh = 300.0 / (nm_per_px * downsample)  # px

    def _rv(coords: np.ndarray, n: int = 30) -> np.ndarray:
        """Recall vector: unit direction of last n pixels."""
        pts = coords.astype(float)
        n   = min(n, len(pts))
        v   = pts[-1] - pts[max(0, len(pts) - n)]
        m   = float(np.linalg.norm(v))
        return v / m if m > 1e-10 else np.array([0.0, 1.0])

    def _near_border(coord: np.ndarray) -> bool:
        """True if this endpoint is within border_px of the image frame."""
        if H_sk == 0 or W_sk == 0:
            return False
        r, c = float(coord[0]), float(coord[1])
        return (r < border_px or r > H_sk - border_px or
                c < border_px or c > W_sk - border_px)

    def _chord(coords: np.ndarray) -> float:
        pts = coords.astype(float)
        p0, p1 = pts[0], pts[-1]
        cl = float(np.linalg.norm(p1 - p0))
        if cl < 1.0: return 0.0
        cd = (p1 - p0) / cl
        cp = np.array([-cd[1], cd[0]])
        return float(np.max(np.abs((pts - p0) @ cp)) / cl)

    def _sin(coords: np.ndarray) -> float:
        pts = coords.astype(float)
        arc = sum(float(np.linalg.norm(pts[i+1] - pts[i]))
                  for i in range(len(pts) - 1))
        return arc / max(float(np.linalg.norm(pts[-1] - pts[0])), 1.0)

    def _arc_um(coords: np.ndarray) -> float:
        pts = coords.astype(float)
        return (sum(float(np.linalg.norm(pts[i+1] - pts[i]))
                    for i in range(len(pts) - 1))
                * nm_per_px * downsample / 1000.0)

    def _make_wire(mc: np.ndarray, wa: dict, wb: dict) -> dict:
        mc32 = mc.astype(np.int32)
        pts  = mc32.astype(float)
        cen  = pts.mean(axis=0)
        cent = pts - cen
        try:
            _, _, Vt = np.linalg.svd(cent, full_matrices=False)
            d = Vt[0]
        except (np.linalg.LinAlgError, ValueError):
            d = np.array([0.0, 1.0])
        ang = float(np.degrees(np.arctan2(d[0], d[1])) % 180.0)
        return {
            "length_um":      round(_arc_um(mc32), 3),
            "length_nm":      round(_arc_um(mc32) * 1000.0, 1),
            "n_segments":     wa.get("n_segments", 1) + wb.get("n_segments", 1),
            "ep_to_ep":       wa.get("ep_to_ep", False) and wb.get("ep_to_ep", False),
            "wire_angle_deg": round(ang, 1),
            "truncated":      wa.get("truncated", False) or wb.get("truncated", False),
            "sinuosity":      round(_sin(mc32), 4),
            "_coords":        mc32,
            "centroid_row":   round(float(cen[0]), 1),
            "centroid_col":   round(float(cen[1]), 1),
            "diameter_nm":    float("nan"),
        }

    current = list(wires)

    for pass_idx in range(n_passes):
        # Build endpoint index — exclude border-proximate endpoints
        eps: list[dict] = []
        for wi, w in enumerate(current):
            c = w["_coords"].astype(float)
            if len(c) < 3:
                continue
            n  = min(search_distance, len(c))
            vs = c[n - 1] - c[0];               ms = float(np.linalg.norm(vs))
            ve = c[-1] - c[max(0, len(c) - n)]; me = float(np.linalg.norm(ve))
            vs = vs / ms if ms > 1e-10 else np.array([0.0, 1.0])
            ve = ve / me if me > 1e-10 else np.array([0.0, 1.0])
            # Only add endpoints not close to the image border
            if not _near_border(c[0]):
                eps.append({"wi": wi, "end": "start", "coord": c[0],  "dir": -vs})
            if not _near_border(c[-1]):
                eps.append({"wi": wi, "end": "end",   "coord": c[-1], "dir":  ve})

        if not eps:
            break

        ep_arr = np.array([e["coord"] for e in eps])
        tree   = cKDTree(ep_arr)
        pairs  = list(tree.query_pairs(max_gap_px))

        # Evaluate and score each pair
        candidates: list[dict] = []
        for i, j in pairs:
            ei, ej = eps[i], eps[j]
            if ei["wi"] == ej["wi"]:
                continue
            gap = ej["coord"] - ei["coord"]
            gl  = float(np.linalg.norm(gap))
            if gl < 0.1:
                continue
            gd = gap / gl

            # ── PROXIMITY CONE: both endpoints must face toward each other ──
            # The gap vector must lie within cone_angle_deg of each endpoint's
            # outgoing recall direction.
            ai = float(np.degrees(np.arccos(np.clip(np.dot(ei["dir"],  gd), -1.0, 1.0))))
            aj = float(np.degrees(np.arccos(np.clip(np.dot(ej["dir"], -gd), -1.0, 1.0))))
            if ai > cone_angle_deg or aj > cone_angle_deg:
                continue

            # ── JOIN CURVATURE GATE ──────────────────────────────────────────
            # At the join point wire A exits via ei.dir and wire B continues
            # in direction -ej.dir (reversed).  A valid smooth continuation
            # has these nearly parallel; a kink gives a large angle.
            # This directly prevents over-connectivity by enforcing that the
            # merged wire cannot bend back at the point of joining.
            join_angle = float(np.degrees(np.arccos(
                np.clip(np.dot(ei["dir"], -ej["dir"]), -1.0, 1.0))))
            if join_angle > max_join_angle_deg:
                continue

            # ── ENDPOINT JUNCTION GATE ──────────────────────────────────────
            # Both endpoints must be within _ep_junc_thresh of a junction pixel.
            # A fragment split at junction J has its endpoint AT J (dist ≈ 0).
            # A genuine wire tip is far from any junction.
            # This is the key gate that prevents over-connectivity: two random
            # wires whose endpoints happen to be close will NOT both have their
            # endpoints adjacent to a junction — only true junction-split pairs do.
            if junc_dist is not None:
                r_i = int(np.clip(ei["coord"][0], 0, H_sk - 1))
                c_i = int(np.clip(ei["coord"][1], 0, W_sk - 1))
                r_j = int(np.clip(ej["coord"][0], 0, H_sk - 1))
                c_j = int(np.clip(ej["coord"][1], 0, W_sk - 1))
                if (junc_dist[r_i, c_i] >= _ep_junc_thresh or
                        junc_dist[r_j, c_j] >= _ep_junc_thresh):
                    continue   # at least one endpoint is a genuine tip, not junction-split

            ci = current[ei["wi"]]["_coords"].astype(float)
            cj = current[ej["wi"]]["_coords"].astype(float)
            if ei["end"] == "start": ci = ci[::-1]   # flip so join is at end of ci
            if ej["end"] == "end":   cj = cj[::-1]   # flip so join is at start of cj

            mc = np.vstack([ci, cj])
            cr = _chord(mc)
            si = _sin(mc)
            if cr > max_chord_ratio or si > max_sinuosity:
                continue

            candidates.append({
                "wi": ei["wi"], "wj": ej["wi"],
                "mc": mc, "len": _arc_um(mc),
                "chord": cr,
            })

        candidates.sort(key=lambda x: x["len"], reverse=True)

        # Greedy accept: longest valid merge first, no wire used twice
        used: set = set()
        new_wires: list[dict] = []
        for m in candidates:
            if m["wi"] in used or m["wj"] in used:
                continue
            used.add(m["wi"])
            used.add(m["wj"])
            new_wires.append(_make_wire(m["mc"], current[m["wi"]], current[m["wj"]]))

        n_merged = len(new_wires)
        kept = [w for i, w in enumerate(current) if i not in used]
        current = kept + new_wires

        if n_merged == 0:
            break   # converged

        print(f"  Merge pass {pass_idx + 1}: {n_merged} fragment pairs joined "
              f"→ {len(current)} wires total")

    return current


# ─── DIAMETER (SEM) ──────────────────────────────────────────────────────────

def measure_diameter_fwhm(coords_half, img_orig, nm_per_px,
                           n_samples=12, profile_half=40):
    coords = coords_half * 2
    n = len(coords)
    if n < 5: return None
    step = max(1, n // n_samples)
    widths = []
    for i in range(1, min(n_samples, n//step)-1):
        idx = i*step
        if idx+step >= n: break
        r0,c0 = coords[idx]; r1,c1 = coords[min(idx+step,n-1)]
        dr,dc = r1-r0, c1-c0; mag = math.hypot(dr,dc)
        if mag < 1e-6: continue
        pr,pc = -dc/mag, dr/mag
        ts = np.arange(-profile_half, profile_half+1)
        rs = np.clip(np.round(r0+ts*pr).astype(int), 0, img_orig.shape[0]-1)
        cs_i = np.clip(np.round(c0+ts*pc).astype(int), 0, img_orig.shape[1]-1)
        prof = img_orig[rs, cs_i].astype(float)
        pk_i = int(np.argmax(prof)); pk_v = prof[pk_i]
        bg = float(np.percentile(prof, 10))
        if pk_v - bg < 15: continue
        hm = bg + (pk_v-bg)*0.5
        left = right = 0
        for t in range(pk_i, 0, -1):
            if prof[t] < hm: left = pk_i-t; break
        for t in range(pk_i, len(prof)-1):
            if prof[t] < hm: right = t-pk_i; break
        if left > 0 and right > 0:
            widths.append((left+right)*nm_per_px)
    if not widths: return None
    w = np.array(widths)
    good = w[w < np.median(w)*3.0]
    return float(np.median(good)) if len(good) else None


def attach_diameter(wire_data, img_crop, nm_per_px, min_diam_nm, max_diam_nm):
    out = []
    for wd in wire_data:
        diam = measure_diameter_fwhm(wd["_coords"], img_crop, nm_per_px)
        if diam is None or not (min_diam_nm <= diam <= max_diam_nm):
            continue
        wd["diameter_nm"] = round(diam, 1)
        wd["aspect_ratio"] = round(wd["length_nm"] / diam, 2) if diam > 0 else float("nan")
        out.append(wd)
    return out

# ─── OUTPUTS ─────────────────────────────────────────────────────────────────

def save_raw_skeleton(img_crop, skel, endpoint_mask, junction_mask,
                       downsample, image_name, out_dir):
    """Save raw skeleton: grey=wire, GREEN=endpoint, RED=junction."""
    S = max(downsample*2, 2)
    if img_crop.ndim == 3:
        bg = img_crop[::S, ::S].copy()
    else:
        bg = np.stack([img_crop[::S, ::S]]*3, axis=-1).copy()
    h_out, w_out = bg.shape[:2]

    step = S // downsample
    def small(mask):
        m = mask[::step, ::step]
        return m[:h_out, :w_out]

    sk_s  = small(skel)
    ep_s  = small(endpoint_mask)
    jn_s  = small(junction_mask)

    bg[sk_s] = [200, 200, 200]
    bg[jn_s] = [255,  50,  50]   # RED  = junction
    bg[ep_s] = [ 50, 255,  50]   # GREEN = endpoint

    out = out_dir / f"{image_name}_raw_skeleton.png"
    PILImage.fromarray(bg.astype(np.uint8)).save(out)
    print(f"    Skeleton-> {out.name}")


def save_overlay(img_crop, all_wire_data, sampled,
                  downsample, image_name, out_dir, nm_per_px):
    S = max(downsample*2, 2)
    if img_crop.ndim == 3:
        rgb = img_crop[::S, ::S].copy()
    else:
        rgb = np.stack([img_crop[::S, ::S]]*3, axis=-1).copy()
    h_out, w_out = rgb.shape[:2]

    # Wire-length density heatmap: warm orange tint where wires are dense
    if all_wire_data:
        lmap = wire_length_heatmap(all_wire_data, skel_shape=(img_crop.shape[0]//downsample,
                                                               img_crop.shape[1]//downsample),
                                   downsample=downsample, nm_per_px=nm_per_px)
        lmap_norm = lmap / max(lmap.max(), 1e-6)          # [0, 1]
        step = S // downsample
        lm_s = lmap_norm[::step, ::step]
        h_lm = min(lm_s.shape[0], h_out)
        w_lm = min(lm_s.shape[1], w_out)
        lm_s = lm_s[:h_lm, :w_lm]
        # Add orange tint proportional to local wire-length density
        alpha = (lm_s * 0.35).astype(np.float32)
        rgb[:h_lm,:w_lm,0] = np.clip(rgb[:h_lm,:w_lm,0].astype(np.float32)
                                        + alpha * 120, 0, 255).astype(np.uint8)
        rgb[:h_lm,:w_lm,1] = np.clip(rgb[:h_lm,:w_lm,1].astype(np.float32)
                                        - alpha *  30, 0, 255).astype(np.uint8)
        rgb[:h_lm,:w_lm,2] = np.clip(rgb[:h_lm,:w_lm,2].astype(np.float32)
                                        - alpha *  60, 0, 255).astype(np.uint8)

    sampled_set = {id(d) for d in sampled}
    max_len = max((d["length_um"] for d in all_wire_data), default=20.0)
    cmap = plt.colormaps["plasma"]

    for wd in all_wire_data:
        if id(wd) in sampled_set: continue
        col_bg = [120, 100, 80] if wd.get("truncated") else [160, 160, 160]
        for ry, rx in wd["_coords"]:
            oy = int(ry*downsample)//S; ox = int(rx*downsample)//S
            if 0<=oy<h_out and 0<=ox<w_out: rgb[oy,ox] = col_bg

    for wd in sorted(sampled, key=lambda d: d["length_um"]):
        t = min(wd["length_um"]/max(max_len,1.0), 1.0)
        col = (np.array(cmap(t)[:3])*255).astype(np.uint8)
        if wd.get("truncated"):  # desaturate truncated (frame-exit) wires
            col = ((col.astype(int) + 180) // 2).astype(np.uint8)
        for ry, rx in wd["_coords"]:
            oy = int(ry*downsample)//S; ox = int(rx*downsample)//S
            for ddy in range(-1,2):
                for ddx in range(-1,2):
                    ny,nx = oy+ddy, ox+ddx
                    if 0<=ny<h_out and 0<=nx<w_out: rgb[ny,nx] = col

    sb_um = 10.0 if nm_per_px < 100 else 20.0
    sb_px = max(int(sb_um*1000/nm_per_px)//S, 4)
    rgb[h_out-30:h_out-22, 30:30+sb_px] = 255

    ep2ep_n = sum(1 for d in sampled if d.get("ep_to_ep"))
    fig, ax = plt.subplots(figsize=(w_out/80, h_out/80), dpi=80)
    ax.imshow(rgb, interpolation="nearest")
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, max_len))
    cb = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("Wire length (um)", fontsize=10)
    ax.set_title(
        f"{image_name}  |  {len(all_wire_data)} wires total ({len(sampled)} shown)  "
        f"({ep2ep_n} tip-to-tip)  |  {nm_per_px:.1f} nm/px",
        fontsize=8)
    ax.text(30+sb_px//2, h_out-38, f"{sb_um:.0f} um",
            ha="center", va="bottom", color="white", fontsize=8)
    ax.axis("off"); plt.tight_layout()
    out = out_dir / f"{image_name}_overlay.png"
    plt.savefig(out, dpi=80, bbox_inches="tight"); plt.close()
    print(f"    Overlay -> {out.name}")


def save_plots(sampled, image_name, out_dir, nm_per_px, mode, img_shape=None):
    lengths_um = np.array([d["length_um"] for d in sampled])
    n = len(sampled)
    # Always 3 panels: length, (diameter for SEM), orientation
    has_diam = mode == "SEM"
    has_ar   = has_diam  # aspect ratio only when diameter is available
    n_data_cols = (1 + int(has_diam) + int(has_ar))  # length [+ diam + AR]
    n_cols = n_data_cols + 1                           # +1 for orientation
    fig, axes = plt.subplots(1, n_cols, figsize=(7*n_cols, 5))
    if n_cols == 1: axes = [axes]
    fig.suptitle(f"{image_name}  [{mode}]  -  {n} wires  |  {nm_per_px:.1f} nm/px",
                 fontsize=12, y=1.01)

    def _panel(ax, data, ch, ck, cm, xlabel, title, unit):
        p99 = np.percentile(data, 99)
        ax.hist(data, bins=30, color=ch, edgecolor="white", linewidth=0.4,
                alpha=0.85, density=True)
        if len(data) > 3:
            kde = gaussian_kde(data, bw_method="silverman")
            x = np.linspace(data.min()*0.8, p99*1.15, 300)
            ax.plot(x, kde(x), color=ck, linewidth=2, label="KDE")
        med = np.median(data)
        ax.axvline(med, color=cm, lw=1.5, ls="--", label=f"median {med:.1f} {unit}")
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{title}  (mu={data.mean():.1f}, sigma={data.std():.1f} {unit})")
        ax.set_xlim(left=0); ax.legend(fontsize=9)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    _panel(axes[0], lengths_um, "#4A90D9","#1A5FA8","#E05C2A",
           "Wire length (um)", "Length", "um")
    if has_diam:
        ds = np.array([d["diameter_nm"] for d in sampled
                       if not math.isnan(d.get("diameter_nm", float("nan")))])
        if len(ds) > 0:
            _panel(axes[1], ds, "#E8994A","#A05A10","#1A5FA8",
                   "Diameter (nm)", "Diameter", "nm")

    # Aspect ratio panel (SEM only, after diameter)
    if has_ar:
        ar_data = np.array([d["aspect_ratio"] for d in sampled
                            if not math.isnan(d.get("aspect_ratio", float("nan")))
                            and d.get("aspect_ratio", 0) > 0])
        if len(ar_data) > 0:
            _panel(axes[2], ar_data, "#7B5EA7","#4A2080","#E05C2A",
                   "Aspect ratio (length / diameter)", "Aspect ratio", "")
            axes[2].set_title(
                f"Aspect ratio  (mu={ar_data.mean():.1f}, "
                f"sigma={ar_data.std():.1f})")

    # Orientation rose plot
    angles = np.array([d["wire_angle_deg"] for d in sampled
                        if d.get("wire_angle_deg") is not None
                        and not d.get("truncated", False)
                        and not (isinstance(d.get("wire_angle_deg"), float) and
                                 __import__("math").isnan(d["wire_angle_deg"]))])
    if len(angles) >= 10:
        ori_ax = axes[n_data_cols]
        ori_ax.hist(angles, bins=18, range=(0, 180),
                    color="#4A90D9", edgecolor="white", linewidth=0.4, density=True)
        # Uniform reference line
        ori_ax.axhline(1.0 / 180, color="#E05C2A", ls="--", lw=1.2,
                       label="Uniform reference")
        ori_ax.set_xlabel("Wire orientation (°)", fontsize=11)
        ori_ax.set_ylabel("Density", fontsize=11)
        stats_d = orientation_stats(angles)
        sig_marker = "★" if stats_d["significant"] else ""
        ori_ax.set_title(
            f"Orientation {sig_marker}\n"
            f"R={stats_d['mean_resultant_R']:.3f} ({stats_d['r_strength']}), "
            f"µ={stats_d['mean_direction_deg']:.0f}°\n"
            f"Rayleigh p={stats_d['rayleigh_p']:.4f}  "
            f"χ² p={stats_d['chi2_p']:.4f}",
            fontsize=9
        )
        ori_ax.set_xlim(0, 180)
        ori_ax.legend(fontsize=8)
    plt.tight_layout()
    out = out_dir / f"{image_name}_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Plots   -> {out.name}")

# ─── SHARED SAVE ─────────────────────────────────────────────────────────────

def save_results(wire_data, img_crop, args, out_dir,
                  name, nm_per_px, downsample, mode, csv_fields):
    """Save all outputs for one image.

    Plots and CSV statistics are computed on ALL qualified wires so that the
    tail of the distribution (long/rare wires) is never lost to sampling.
    The overlay is rendered from a sample for visual clarity when wire counts
    are large (controlled by --n_sample).
    """
    # ── Tag every wire with the image name (in-place) ─────────────────────
    for d in wire_data:
        d["image"] = name
        d.setdefault("truncated", False)

    # Split into complete (not truncated) and truncated subsets.
    # Length/orientation statistics use only complete wires.
    complete = [d for d in wire_data if not d.get("truncated", False)]
    n_trunc  = len(wire_data) - len(complete)
    if n_trunc:
        print(f"  Truncated   : {n_trunc} wires touch frame edge — excluded from statistics")

    # ── Write complete wire CSV ────────────────────────────────────────────
    all_csv = out_dir / f"{name}_all_wires.csv"
    with open(all_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows([{k:v for k,v in d.items() if k in csv_fields}
                          for d in wire_data])
    print(f"    Wires  -> {all_csv.name}  ({len(wire_data)} wires, seed={args.seed})")

    # ── Summary statistics on ALL wires ───────────────────────────────────
    complete = [d for d in wire_data if not d.get("truncated", False)]
    n_trunc  = len(wire_data) - len(complete)
    ls = np.array([d["length_um"] for d in complete]) if complete else np.array([0.0])
    ep2ep = sum(1 for d in complete if d.get("ep_to_ep"))
    print(f"  Wires       : {len(complete)} complete, {n_trunc} truncated (frame-edge)")
    print(f"  Tip-to-tip  : {ep2ep} ({ep2ep/max(len(complete),1):.0%} of complete wires)")
    print(f"  Length      : median={np.median(ls):.1f}  "
          f"p90={np.percentile(ls,90):.1f}  max={ls.max():.1f} um")
    if mode == "SEM":
        ds = np.array([d["diameter_nm"] for d in complete
                       if not math.isnan(d.get("diameter_nm", float("nan")))])
        if len(ds):
            print(f"  Diameter    : median={np.median(ds):.0f}  mean={ds.mean():.0f} nm"
                  f"  (apparent — PSF not deconvolved)")

    # ── Plots and orientation on complete (non-truncated) wires only ─────────
    _img_shape = img_crop.shape[:2] if hasattr(img_crop, 'shape') else None
    save_plots(complete, name, out_dir, nm_per_px, mode, img_shape=_img_shape)

    # ── Overlay sampled for rendering speed ───────────────────────────────
    n_overlay = min(args.n_sample, len(wire_data))
    overlay_wires = random.sample(wire_data, n_overlay)
    save_overlay(img_crop, wire_data, overlay_wires, downsample, name, out_dir, nm_per_px)
    return wire_data

# ─── COVERAGE ANALYSIS (for coarse VLM: 20x etc.) ───────────────────────────

def compute_coverage(img_crop: np.ndarray, img_path, nm_per_px: float,
                     out_dir, name: str, vlm_thresh: float = 60.0) -> dict:
    """Compute wire network coverage metrics for images where individual wire
    tracing is not possible (wire diameter << pixel size, e.g. 20x VLM).

    Outputs:
        _coverage.png  : binary overlay showing detected wire network
        Coverage stats : fill fraction, orientation distribution
    Returns dict with summary metrics.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if img_crop.ndim == 3:
        grey = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
    else:
        grey = img_crop

    # CLAHE + threshold to isolate wire network
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32))
    eq = clahe.apply(grey)
    thresh = filters.threshold_otsu(eq)
    # At 20x wires appear bright against grey substrate
    binary = (eq > thresh).astype(np.uint8)
    try:
        binary_bool = remove_small_objects(binary.astype(bool), max_size=8)
    except TypeError:
        binary_bool = remove_small_objects(binary.astype(bool), min_size=9)

    coverage_pct = binary_bool.mean() * 100.0
    h, w = grey.shape
    field_um_w = w * nm_per_px / 1000
    field_um_h = h * nm_per_px / 1000

    # Skeleton for orientation distribution
    skel = skeletonize(binary_bool)
    kern = np.ones((3, 3), dtype=np.uint8)
    nc = cv2.filter2D(skel.astype(np.uint8), -1, kern)
    normal_mask = skel & (nc == 3)  # degree-2 pixels for orientation

    # Orientation via segment PCA (one angle per skeleton segment).
    # Per-pixel gradient approaches suffer from a staircase artefact:
    # diagonal wires are represented as horizontal+vertical steps on the
    # grid, creating a systematic deficit at 45° and 135°.  Computing PCA
    # over all pixels of each connected segment recovers the true direction.
    from skimage.measure import label as _sk_label
    junction_mask_cov = skel & (nc >= 4)
    segs_only = skel & ~junction_mask_cov
    seg_lab = _sk_label(segs_only, connectivity=2)
    raw_angles = []
    MIN_SEG_PX = 5  # skip tiny noise fragments
    for region in measure.regionprops(seg_lab):
        if region.area < MIN_SEG_PX:
            continue
        coords = region.coords.astype(float)
        centered = coords - coords.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            d = Vt[0]
            raw_angles.append(np.degrees(np.arctan2(d[0], d[1])) % 180.0)
        except (np.linalg.LinAlgError, ValueError):
            continue
    angles_deg = np.array(raw_angles) if raw_angles else np.zeros(1)

    # Save coverage overlay
    S = 2
    if img_crop.ndim == 3:
        bg = img_crop[::S, ::S].copy()
    else:
        bg = np.stack([img_crop[::S, ::S]] * 3, axis=-1).copy()

    h_out, w_out = bg.shape[:2]
    bin_small = binary_bool[::S, ::S][:h_out, :w_out]
    bg[bin_small, 0] = np.clip(bg[bin_small, 0].astype(int) + 80, 0, 255)
    bg[bin_small, 1] = np.clip(bg[bin_small, 1].astype(int) - 20, 0, 255)
    bg[bin_small, 2] = np.clip(bg[bin_small, 2].astype(int) - 20, 0, 255)

    # Scale bar
    sb_um = 100.0
    sb_px = max(int(sb_um * 1000 / nm_per_px) // S, 4)
    bg[h_out - 30: h_out - 22, 20: 20 + sb_px] = 255

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(bg, interpolation="nearest")
    axes[0].set_title(
        f"{name}  |  Coverage = {coverage_pct:.1f}%  |  "
        f"{field_um_w:.0f}×{field_um_h:.0f} µm  |  {nm_per_px:.0f} nm/px",
        fontsize=10,
    )
    axes[0].text(20 + sb_px // 2, h_out - 38, f"{sb_um:.0f} µm",
                 ha="center", va="bottom", color="white", fontsize=9)
    axes[0].axis("off")

    # Orientation rose diagram (weighted by vesselness response)
    axes[1].hist(angles_deg, bins=36, range=(0, 180), color="#4A90D9",
                 edgecolor="white", linewidth=0.4)
    axes[1].set_xlabel("Wire orientation (°)", fontsize=11)
    axes[1].set_ylabel("Skeleton pixel count", fontsize=11)
    axes[1].set_title(f"Wire orientation distribution\n(0° = horizontal)", fontsize=10)
    axes[1].set_xlim(0, 180)

    plt.suptitle(
        f"{name}  [Coverage mode — wires sub-pixel at {nm_per_px:.0f} nm/px]\n"
        f"Orientation via segment PCA (no grid staircase artefact)",
        fontsize=11)
    plt.tight_layout()
    out = out_dir / f"{name}_coverage.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    # Orientation statistics on coverage skeleton
    ori_stats = orientation_stats(angles_deg)
    print(f"    Coverage -> {out.name}  ({coverage_pct:.1f}% wire coverage)")
    print(f"    Orient.  : {ori_stats['summary_line']}")

    metrics = {
        "image": name,
        "mode": "coverage",
        "nm_per_px": round(nm_per_px, 2),
        "field_um_w": round(field_um_w, 1),
        "field_um_h": round(field_um_h, 1),
        "coverage_pct": round(coverage_pct, 2),
        "skeleton_px": int(skel.sum()),
        "median_orientation_deg":  round(float(np.median(angles_deg)), 1),
        "rayleigh_p":              ori_stats["rayleigh_p"],
        "mean_resultant_R":        ori_stats["mean_resultant_R"],
        "preferred_direction_deg": ori_stats["mean_direction_deg"],
        "orientation_significant": ori_stats["significant"],
    }
    return metrics



# ─── NANO1D-STYLE PIXEL TRACING ──────────────────────────────────────────────
#
# This replaces the previous segment-based pipeline (get_segment_info,
# precompute_pairs, find_best_continuation, trace_wires, gap_bridging,
# spur_pruning) with the algorithm from:
#
#   Moradpur-Tari et al., "Nano1D: An accurate Computer Vision software for
#   analysis and segmentation of low-dimensional nanostructures",
#   Ultramicroscopy 261 (2024) 113949.
#
# The core innovation is pixel-by-pixel tracing with a ROLLING RECALL VECTOR
# at intersections: the algorithm uses the actual pixel coordinates of the
# last `search_distance` pixels as a tangent estimate, rather than a single
# noisy segment direction.  This is dramatically more robust at crossings.
#
# Our additions on top of Nano1D:
#   - Frangi vesselness binary (far cleaner than raw threshold)
#   - Gap bridging (reconnects small breaks in the skeleton)
#   - Chord deviation post-filter (rejects V-kinks)
#   - Truncated wire detection (wires touching image frame)
#   - Per-wire PCA angle for orientation statistics
#   - Diameter FWHM (SEM only, via attach_diameter)


def neighbour_count_image(skel: np.ndarray) -> np.ndarray:
    """Compute the Nano1D nearest-neighbour count N(x,y) for every skeleton pixel.

    Uses the connectivity kernel ω_c (centre=6, others=1):
        g      = ω_c ∗ skel
        N(x,y) = (g·b − 6)·b′·skel
    where b=1 if g>6, b'=1 if g·b>0.

    Result values:
        N = 0  : background
        N = 1  : tail (endpoint)
        N = 2  : section (normal pixel)
        N ≥ 3  : intersection (junction)
    """
    wc = np.ones((3, 3), dtype=np.float32)
    wc[1, 1] = 6.0
    g  = cv2.filter2D(skel.astype(np.float32), -1, wc,
                       borderType=cv2.BORDER_CONSTANT)
    b  = (g > 6).astype(np.float32)
    bp = ((g * b) > 0).astype(np.float32)
    N  = ((g * b - 6) * bp * skel.astype(np.float32)).astype(np.int32)
    return N


def remove_nn4_centres(skel: np.ndarray, N: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove centre pixels of 4-way intersections (N=4).

    Per Nano1D section 3.1: these pixels create ambiguous branching points
    that confuse the orienteering; removing them forces the junction into
    cleaner 3-way (trisection) topology which the recall vector can handle.
    The neighbour counts are recomputed after removal.
    """
    skel = skel.copy()
    ys4, xs4 = np.where(N == 4)
    for y, x in zip(ys4, xs4):
        skel[y, x] = 0
    return skel, neighbour_count_image(skel)


def bridge_gaps_nano(binary: np.ndarray, skel: np.ndarray, N: np.ndarray,
                     nm_per_px: float, downsample: int,
                     max_gap_um: float = 2.0,
                     max_angle_deg: float = 25.0,
                     max_cross_pct: float = 0.15) -> tuple[np.ndarray, int]:
    """Connect pairs of facing tails (N=1) across small gaps.

    Uses a KD-tree over tail positions.  For each candidate pair, the
    gap-closing line is drawn only if:
      • both tails point toward each other (angle ≤ max_angle_deg), and
      • the bridge path does not cross existing skeleton (cross_frac ≤ max_cross_pct).

    Returns (updated_binary, n_bridges_added).
    """
    from scipy.spatial import cKDTree

    H, W = skel.shape
    max_gap_px = max_gap_um * 1000.0 / (nm_per_px * downsample)

    tail_ys, tail_xs = np.where(N == 1)
    if len(tail_ys) < 2:
        return binary, 0

    # Estimate tail direction from the single neighbouring skeleton pixel
    skel_arr = skel.astype(bool)

    def tail_dir(ty, tx):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                ny, nx = ty + dy, tx + dx
                if 0 <= ny < H and 0 <= nx < W and skel_arr[ny, nx]:
                    return np.array([ty - ny, tx - nx], dtype=float)
        return np.array([0.0, 1.0])

    tail_pts = np.column_stack([tail_ys, tail_xs]).astype(float)
    tree = cKDTree(tail_pts)
    pairs = tree.query_pairs(max_gap_px)

    candidates = []
    for i, j in pairs:
        r1, c1 = int(tail_ys[i]), int(tail_xs[i])
        r2, c2 = int(tail_ys[j]), int(tail_xs[j])
        dr, dc = r2 - r1, c2 - c1
        dist = np.hypot(dr, dc)
        if dist < 1: continue
        gv = np.array([dr, dc]) / dist
        d1 = tail_dir(r1, c1); d2 = tail_dir(r2, c2)
        mag1 = np.linalg.norm(d1); mag2 = np.linalg.norm(d2)
        if mag1 < 1e-10 or mag2 < 1e-10: continue
        d1 /= mag1; d2 /= mag2
        a1 = np.degrees(np.arccos(np.clip(np.dot(d1,  gv), -1.0, 1.0)))
        a2 = np.degrees(np.arccos(np.clip(np.dot(d2, -gv), -1.0, 1.0)))
        if a1 <= max_angle_deg and a2 <= max_angle_deg:
            candidates.append((dist, i, j, (r1, c1), (r2, c2)))

    candidates.sort(key=lambda x: x[0])
    binary_new = binary.copy()
    used: set = set()
    n_bridges = 0

    for dist, i, j, ep1, ep2, *_ in candidates:
        if ep1 in used or ep2 in used:
            continue
        r1, c1 = ep1; r2, c2 = ep2
        n_pts = max(int(dist) + 1, 2)
        ts = np.linspace(0.0, 1.0, n_pts)
        brs = np.round(r1 + ts * (r2 - r1)).astype(int)
        bcs = np.round(c1 + ts * (c2 - c1)).astype(int)
        vm = (brs >= 0) & (brs < H) & (bcs >= 0) & (bcs < W)
        brs, bcs = brs[vm], bcs[vm]
        mid_r, mid_c = brs[2:-2], bcs[2:-2]
        cross_frac = float(skel_arr[mid_r, mid_c].mean()) if len(mid_r) > 0 else 0.0
        if cross_frac > max_cross_pct:
            continue
        binary_new[brs, bcs] = True
        used.add(ep1); used.add(ep2)
        n_bridges += 1

    return binary_new, n_bridges


def trace_wires_nano1d(skel: np.ndarray, N: np.ndarray,
                        nm_per_px: float, downsample: int,
                        min_len_nm: float = 5000.0,
                        search_distance: int = 30,
                        max_chord_dev_ratio: float = 0.40,
                        frame_h: int = 0, frame_w: int = 0) -> list[dict]:
    """Trace wires using the Nano1D pixel-by-pixel recall-vector algorithm.

    Starting from every un-visited tail pixel (N=1), the tracer follows the
    skeleton pixel by pixel:

      • SECTION pixels (N=2): pick the neighbour most aligned with the
        last step direction (avoids backtracking).

      • INTERSECTION pixels (N≥3): compute a RECALL VECTOR from the last
        `search_distance` pixels of the current trace, then pick the neighbour
        most aligned with that vector.  This is the Nano1D core innovation:
        the rolling recall vector is far more robust than a single segment
        direction because it integrates the actual pixel path.

    Post-acceptance filters:
      • min_len_nm: minimum arc length.
      • max_chord_dev_ratio: maximum (max perpendicular deviation from the
        endpoint chord) / chord_length.  Rejects V-kinks (which have large
        ratios) while passing physically valid parabolic or S-curve bows.

    Returns a list of wire dicts compatible with the rest of the pipeline.
    """
    H, W = skel.shape
    frame_h = frame_h or H
    frame_w = frame_w or W

    # O(1) skeleton lookup
    skel_set: set[tuple] = set(zip(map(int, np.where(skel)[0]),
                                   map(int, np.where(skel)[1])))

    tail_ys, tail_xs = np.where(N == 1)
    scan_count: dict[tuple, int] = {}   # how many times a tail has been used
    wires: list[dict] = []

    min_len_px = min_len_nm / (nm_per_px * downsample)

    for ty, tx in zip(map(int, tail_ys), map(int, tail_xs)):
        start = (ty, tx)
        if scan_count.get(start, 0) > 0:
            continue

        # ── First step from tail ────────────────────────────────────────────
        init_nbrs = [
            (ty + dy, tx + dx)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dy, dx) != (0, 0) and (ty + dy, tx + dx) in skel_set
        ]
        if not init_nbrs:
            continue

        scan_count[start] = 1
        cur = init_nbrs[0]
        line: list[tuple] = [start, cur]
        visited: set[tuple] = {start, cur}
        step = np.array([cur[0] - ty, cur[1] - tx], dtype=float)

        # ── Trace pixel by pixel ────────────────────────────────────────────
        reached_tail = False
        for _ in range(4000):  # safety limit ~200 µm at 50 nm/px
            y, x = cur
            nn = N[y, x]

            # Neighbours not yet visited in this trace
            nbrs = [
                (y + dy, x + dx)
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy, dx) != (0, 0)
                and (y + dy, x + dx) in skel_set
                and (y + dy, x + dx) not in visited
            ]

            if nn == 1:
                # Reached another tail → complete trace
                scan_count[cur] = scan_count.get(cur, 0) + 1
                reached_tail = True
                break

            if not nbrs:
                break

            if nn <= 2:
                # ── Section: avoid backtracking via last step ─────────────
                if len(nbrs) == 1:
                    nxt = nbrs[0]
                else:
                    best_s = -999.0; nxt = nbrs[0]
                    for n in nbrs:
                        cand = np.array([n[0] - y, n[1] - x], dtype=float)
                        sc = float(np.dot(step, cand))
                        if sc > best_s:
                            best_s = sc; nxt = n
            else:
                # ── Intersection: recall vector from last search_distance px ──
                n_recall = min(search_distance, len(line))
                p0 = np.array(line[-n_recall], dtype=float)
                p1 = np.array(line[-1],        dtype=float)
                recall = p1 - p0
                mag = np.linalg.norm(recall)
                if mag > 1e-10:
                    recall /= mag
                else:
                    mag_s = np.linalg.norm(step)
                    recall = step / mag_s if mag_s > 1e-10 else np.array([0.0, 1.0])

                best_s = -999.0; nxt = nbrs[0]
                for n in nbrs:
                    arm = np.array([n[0] - y, n[1] - x], dtype=float)
                    arm_mag = np.linalg.norm(arm)
                    if arm_mag > 1e-10:
                        arm /= arm_mag
                    sc = float(np.dot(recall, arm))
                    if sc > best_s:
                        best_s = sc; nxt = n

            step = np.array([nxt[0] - y, nxt[1] - x], dtype=float)
            visited.add(nxt)
            line.append(nxt)
            cur = nxt

        # ── Accept if long enough ───────────────────────────────────────────
        if len(line) < min_len_px:
            continue

        # ── Chord deviation filter: reject V-kinks ─────────────────────────
        if max_chord_dev_ratio > 0 and len(line) >= 4:
            pts = np.array(line, dtype=float)
            p0, p1 = pts[0], pts[-1]
            chord = p1 - p0
            cl = np.linalg.norm(chord)
            if cl > 1.0:
                cd = chord / cl
                cp = np.array([-cd[1], cd[0]])
                devs = np.abs((pts - p0) @ cp)
                if devs.max() / cl > max_chord_dev_ratio:
                    continue   # V-kink — discard

        # ── Wire angle from global PCA of pixel coords ──────────────────────
        coords_arr = np.array(line, dtype=float)
        centered   = coords_arr - coords_arr.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            d = Vt[0]
        except (np.linalg.LinAlgError, ValueError):
            d = np.array([0.0, 1.0])
        wire_angle = float(np.degrees(np.arctan2(d[0], d[1])) % 180.0)

        # ── Arc length (pixel-hop, √2 for diagonals) ───────────────────────
        arc = sum(
            math.sqrt(2.0) if (abs(line[i+1][0] - line[i][0]) == 1
                               and abs(line[i+1][1] - line[i][1]) == 1) else 1.0
            for i in range(len(line) - 1)
        )
        ln_nm = arc * nm_per_px * downsample
        ln_um = ln_nm / 1000.0

        # ── Sinuosity: arc / chord (1.0 = straight, >1 = curved) ───────────
        _chord_len = float(np.linalg.norm(
            np.array(line[-1], float) - np.array(line[0], float)))
        sinuosity_val = round(arc / max(_chord_len, 1.0), 4)

        # ── Truncation flag ─────────────────────────────────────────────────
        def _on_frame(r, c):
            return r <= 1 or c <= 1 or r >= H - 2 or c >= W - 2

        truncated = _on_frame(*line[0]) or _on_frame(*line[-1])

        centroid = tuple(coords_arr.mean(axis=0).tolist())

        wires.append({
            "length_nm":      round(ln_nm, 1),
            "length_um":      round(ln_um, 3),
            "n_segments":     1,          # pixel-level: one continuous path
            "ep_to_ep":       reached_tail,
            "wire_angle_deg": round(wire_angle, 1),
            "truncated":      truncated,
            "sinuosity":      sinuosity_val,
            "_coords":        coords_arr.astype(np.int32),
            "centroid_row":   round(float(centroid[0]), 1),
            "centroid_col":   round(float(centroid[1]), 1),
        })

    return wires


# ─── PER-IMAGE PIPELINES ─────────────────────────────────────────────────────

def _wire_chord(coords: np.ndarray) -> float:
    """Chord deviation ratio for a traced wire."""
    pts = coords.astype(float)
    p0, p1 = pts[0], pts[-1]
    cl = float(np.linalg.norm(p1 - p0))
    if cl < 1.0: return 0.0
    cd = (p1 - p0) / cl
    cp = np.array([-cd[1], cd[0]])
    return float(np.max(np.abs((pts - p0) @ cp)) / cl)


def process_sem(tif_path, args, out_dir):
    import tifffile
    name = tif_path.stem
    print(f"\n{'─'*66}\n  [SEM] {tif_path.name}")
    try:
        with tifffile.TiffFile(tif_path) as tif:
            img = tif.pages[0].asarray()
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(np.uint8)
    except Exception as e:
        print(f"  ERROR: {e}"); return None

    h, w = img.shape
    print(f"  Size        : {w} x {h} px")
    try:
        nm_per_px, bar_start = detect_scale_bar_sem(img, args.scale_nm,
                                                     args.fallback_nm_per_px)
        print(f"  Scale       : {nm_per_px:.2f} nm/px  |  bar at row {bar_start}")
    except RuntimeError as e:
        print(f"  WARNING: {e}  Fallback={args.fallback_nm_per_px}")
        nm_per_px = args.fallback_nm_per_px; bar_start = int(h*0.88)

    img_crop = img[:bar_start, :]
    DS = DOWNSAMPLE_SEM

    # ── Frangi binary at half-res; CLAHE+Otsu full-res for diameter ──────────
    binary_skel, binary_full = make_binary_sem(
        img_crop, args.clahe_clip, args.thresh_frac,
        use_frangi=not args.no_frangi,
        frangi_sigmas=args.frangi_sigmas, frangi_thresh=args.frangi_thresh,
        downsample=DS,
    )
    dist_full = distance_transform_edt(binary_full)  # full-res for diameter

    # ── Skeleton + Nano1D neighbour counts ───────────────────────────────────
    skel, _, junc_mask, _, _, _ = build_skeleton(binary_skel, 1)
    N = neighbour_count_image(skel)
    skel, N = remove_nn4_centres(skel, N)

    n_tails = int((N == 1).sum())
    n_junc  = int((N >= 3).sum())
    junc_rate = n_junc / max(int(skel.sum()), 1)
    print(f"  Skeleton    : {skel.sum():,} px  |  {n_tails:,} tails  |  "
          f"Junction rate: {junc_rate:.0%}")

    # ── Gap bridging (two passes) ─────────────────────────────────────────────
    bin_b, n1 = bridge_gaps_nano(binary_skel, skel, N, nm_per_px, DS,
                                  max_gap_um=args.max_gap_um,
                                  max_angle_deg=args.max_bridge_angle)
    if n1:
        skel, _, junc_mask, _, _, _ = build_skeleton(bin_b, 1)
        N = neighbour_count_image(skel)
        skel, N = remove_nn4_centres(skel, N)
        bin_b2, n2 = bridge_gaps_nano(bin_b, skel, N, nm_per_px, DS,
                                       max_gap_um=args.max_gap_um,
                                       max_angle_deg=args.max_bridge_angle)
        if n2:
            skel, _, junc_mask, _, _, _ = build_skeleton(bin_b2, 1)
            N = neighbour_count_image(skel)
            skel, N = remove_nn4_centres(skel, N)
        print(f"  Bridges     : {n1}+{n2 if n1 else 0}")

    H_sk, W_sk = skel.shape
    save_raw_skeleton(img_crop, skel,
                      (N == 1).astype(bool), (N >= 3).astype(bool),
                      DS, name, out_dir)

    # ── Nano1D pixel-by-pixel tracing with recall vector ─────────────────────
    # Trace ALL fragments with no chord filter and a very low min_len,
    # so the merge pass has the complete set of wire pieces to work with.
    wire_frags_raw = trace_wires_nano1d(
        skel, N,
        nm_per_px=nm_per_px, downsample=DS,
        min_len_nm=500.0,
        search_distance=args.search_distance,
        max_chord_dev_ratio=0.0,
    )
    # Pre-filter the merge pool: remove medium-length traces that are
    # already confirmed kinks (chord > max_chord_ratio), but keep all
    # short stubs (<3µm) which are valid junction fragments.
    POOL_THRESH_NM = 3000.
    wire_frags = [w for w in wire_frags_raw
                  if w['length_nm'] < POOL_THRESH_NM
                  or args.max_chord_ratio <= 0
                  or _wire_chord(w['_coords']) <= args.max_chord_ratio]

    # ── Post-tracing merge: reassemble wires split at junctions ──────────────
    wire_merged = (wire_frags if args.no_merge else
                   merge_wire_fragments(
                       wire_frags, nm_per_px, DS,
                       max_gap_um=args.merge_gap_um,
                       cone_angle_deg=args.merge_angle,
                       max_join_angle_deg=args.merge_join_angle,
                       max_chord_ratio=args.merge_chord,
                       border_um=args.border_um,
                       skel_shape=skel.shape,
                       junc_dist=distance_transform_edt(~(N >= 3).astype(bool)),
                       search_distance=args.search_distance,
                       n_passes=2,
                   ))

    # Apply final chord, sinuosity, minimum-length, and border filters
    _border_px = args.border_um * 1000. / (nm_per_px * DS)
    _H, _W = skel.shape
    def _in_border(coords):
        """True if any coordinate falls inside the border exclusion zone."""
        r, c = coords[:, 0], coords[:, 1]
        return bool(np.any(r < _border_px) or np.any(r > _H - _border_px) or
                    np.any(c < _border_px) or np.any(c > _W - _border_px))
    wire_data = [w for w in wire_merged
                 if w['length_nm'] >= args.min_len_um * 1000
                 and (args.max_chord_ratio <= 0
                      or _wire_chord(w['_coords']) <= args.max_chord_ratio)
                 and w.get('sinuosity', 1.0) <= args.max_sinuosity
                 and not _in_border(w['_coords'])]

    # ── Diameter (SEM only) ───────────────────────────────────────────────────
    for wd in wire_data:
        wd.setdefault("diameter_nm", float("nan"))
    wire_data = attach_diameter(wire_data, img_crop, nm_per_px,
                                 args.min_diam_nm, args.max_diam_nm)
    for wd in wire_data:
        wd.setdefault("diameter_nm", float("nan"))

    n_qual = len(wire_data)
    if n_qual == 0:
        print("  No wires qualified."); return []

    print(f"  Qualified   : {n_qual} wires (>= {args.min_len_um} um)")

    # Orientation stats on complete (non-truncated) wires
    complete_angles = [d["wire_angle_deg"] for d in wire_data
                       if not d.get("truncated", False)
                       and d.get("wire_angle_deg") is not None]
    if len(complete_angles) >= 10:
        _ori = orientation_stats(np.array(complete_angles, dtype=float))
        print(f"  Orientation : {_ori['summary_line']}")

    for d in wire_data:
        d["image"] = name; d["mode"] = "SEM"

    return save_results(wire_data, img_crop, args, out_dir,
                         name, nm_per_px, DS, "SEM", CSV_FIELDS_SEM)


def process_vlm(img_path, args, out_dir, label_um=None):
    name = img_path.stem
    print(f"\n{'─'*66}\n  [VLM] {img_path.name}")
    try:
        img_rgb = np.array(PILImage.open(img_path).convert("RGB"))
    except Exception as e:
        print(f"  ERROR: {e}"); return None

    h, w = img_rgb.shape[:2]
    grey = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    print(f"  Size        : {w} x {h} px")

    mag_from_name = parse_mag_from_filename(img_path)
    if mag_from_name and mag_from_name in VLM_CALIBRATION:
        _detect_label = label_um if label_um is not None else (
            args.vlm_scale_um if args.vlm_scale_um is not None else 10.0)
    else:
        _detect_label = label_um if label_um is not None else (
            args.vlm_scale_um if args.vlm_scale_um is not None else 10.0)

    try:
        nm_per_px, crop_row = detect_scale_bar_vlm(grey, _detect_label)
        nm_per_px_calib, calib_status = validate_vlm_scale(nm_per_px, img_path)
        if args.vlm_scale_um is not None and calib_status != "unknown":
            ratio = nm_per_px / max(nm_per_px_calib, 1e-6)
            if abs(ratio - 1.0) > 0.20:
                print(f"  WARNING: --vlm_scale_um {args.vlm_scale_um} µm implies "
                      f"{nm_per_px:.1f} nm/px but calibration expects "
                      f"{nm_per_px_calib:.1f} nm/px — using calibrated value.")
                nm_per_px = nm_per_px_calib; calib_status = "corrected"
        else:
            nm_per_px = nm_per_px_calib
        calib_note = ("" if calib_status in ("ok", "unknown")
                      else "  [CORRECTED — calibrated from filename mag]")
        print(f"  Scale       : {nm_per_px:.2f} nm/px  |  "
              f"field {w*nm_per_px/1000:.0f}x{crop_row*nm_per_px/1000:.0f} um{calib_note}")
    except RuntimeError as e:
        print(f"  WARNING: {e}  Fallback={args.fallback_nm_per_px}")
        nm_per_px = args.fallback_nm_per_px; crop_row = int(h*0.92)

    img_crop_rgb  = img_rgb[:crop_row, :]
    img_crop_grey = grey[:crop_row, :]
    DS = DOWNSAMPLE_VLM

    # Coverage mode for very coarse images (wires sub-pixel)
    COVERAGE_THRESHOLD_NM_PX = 300.0
    if nm_per_px > COVERAGE_THRESHOLD_NM_PX:
        print(f"  Mode        : COVERAGE (wires sub-pixel at {nm_per_px:.0f} nm/px)")
        metrics = compute_coverage(img_crop_rgb, img_path, nm_per_px,
                                    out_dir, name, vlm_thresh=args.vlm_thresh)
        return [metrics] if metrics else None

    # ── Frangi binary at full VLM resolution ─────────────────────────────────
    binary_full = make_binary_vlm(
        img_crop_grey, clahe_clip=3.0, thresh_val=args.vlm_thresh,
        use_frangi=not args.no_frangi,
        frangi_sigmas=args.vlm_frangi_sigmas, frangi_thresh=args.vlm_frangi_thresh,
    )

    # ── Skeleton + Nano1D neighbour counts ───────────────────────────────────
    skel, _, junc_mask, _, _, _ = build_skeleton(binary_full, DS)
    N = neighbour_count_image(skel)
    skel, N = remove_nn4_centres(skel, N)

    n_tails = int((N == 1).sum())
    n_junc  = int((N >= 3).sum())
    junc_rate = n_junc / max(int(skel.sum()), 1)
    print(f"  Skeleton    : {skel.sum():,} px  |  {n_tails:,} tails  |  "
          f"Junction rate: {junc_rate:.0%}")

    # ── Gap bridging (two passes) ─────────────────────────────────────────────
    bin_b, n1 = bridge_gaps_nano(binary_full, skel, N, nm_per_px, DS,
                                  max_gap_um=args.max_gap_um,
                                  max_angle_deg=args.max_bridge_angle)
    if n1:
        skel, _, junc_mask, _, _, _ = build_skeleton(bin_b, 1)
        N = neighbour_count_image(skel)
        skel, N = remove_nn4_centres(skel, N)
        bin_b2, n2 = bridge_gaps_nano(bin_b, skel, N, nm_per_px, DS,
                                       max_gap_um=args.max_gap_um,
                                       max_angle_deg=args.max_bridge_angle)
        if n2:
            skel, _, junc_mask, _, _, _ = build_skeleton(bin_b2, 1)
            N = neighbour_count_image(skel)
            skel, N = remove_nn4_centres(skel, N)
        print(f"  Bridges     : {n1}+{n2 if n1 else 0}")

    save_raw_skeleton(img_crop_rgb, skel,
                      (N == 1).astype(bool), (N >= 3).astype(bool),
                      DS, name, out_dir)

    # ── Nano1D pixel-by-pixel tracing with recall vector ─────────────────────
    wire_frags_raw = trace_wires_nano1d(
        skel, N,
        nm_per_px=nm_per_px, downsample=DS,
        min_len_nm=500.0,
        search_distance=args.search_distance,
        max_chord_dev_ratio=0.0,
    )
    POOL_THRESH_NM = 3000.
    wire_frags = [w for w in wire_frags_raw
                  if w['length_nm'] < POOL_THRESH_NM
                  or args.max_chord_ratio <= 0
                  or _wire_chord(w['_coords']) <= args.max_chord_ratio]

    # ── Post-tracing merge ────────────────────────────────────────────────────
    wire_merged = (wire_frags if args.no_merge else
                   merge_wire_fragments(
                       wire_frags, nm_per_px, DS,
                       max_gap_um=args.merge_gap_um,
                       cone_angle_deg=args.merge_angle,
                       max_join_angle_deg=args.merge_join_angle,
                       max_chord_ratio=args.merge_chord,
                       border_um=args.border_um,
                       skel_shape=skel.shape,
                       junc_dist=distance_transform_edt(~(N >= 3).astype(bool)),
                       search_distance=args.search_distance,
                       n_passes=2,
                   ))
    _border_px = args.border_um * 1000. / (nm_per_px * DS)
    _H, _W = skel.shape
    def _in_border(coords):
        r, c = coords[:, 0], coords[:, 1]
        return bool(np.any(r < _border_px) or np.any(r > _H - _border_px) or
                    np.any(c < _border_px) or np.any(c > _W - _border_px))
    wire_data = [w for w in wire_merged
                 if w['length_nm'] >= args.vlm_min_len_um * 1000
                 and (args.max_chord_ratio <= 0
                      or _wire_chord(w['_coords']) <= args.max_chord_ratio)
                 and w.get('sinuosity', 1.0) <= args.max_sinuosity
                 and not _in_border(w['_coords'])]

    n_qual = len(wire_data)
    if n_qual == 0:
        print("  No wires qualified."); return []

    print(f"  Qualified   : {n_qual} wires (>= {args.vlm_min_len_um} um)")

    complete_angles = [d["wire_angle_deg"] for d in wire_data
                       if not d.get("truncated", False)
                       and d.get("wire_angle_deg") is not None]
    if len(complete_angles) >= 10:
        _ori = orientation_stats(np.array(complete_angles, dtype=float))
        print(f"  Orientation : {_ori['summary_line']}")

    for d in wire_data:
        d["image"] = name; d["mode"] = "VLM"

    return save_results(wire_data, img_crop_rgb, args, out_dir,
                         name, nm_per_px, DS, "VLM", CSV_FIELDS_VLM)

# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Measure AgNW length (+diameter for SEM) from SEM TIF and VLM PNG.\n"
                    "Uses Frangi vesselness preprocessing with Nano1D pixel-by-pixel "
                    "recall-vector tracing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("images", nargs="*")
    p.add_argument("--input_dir",             type=Path,  default=None)
    p.add_argument("--output_dir",            type=Path,  default=Path("results"))
    p.add_argument("--n_sample",              type=int,   default=300,
                   help="Max wires for overlay rendering (analysis uses all wires)")
    # Tracing
    p.add_argument("--search_distance",       type=int,   default=30,
                   help="Nano1D recall vector length in pixels (default 30 ≈ 1.5 µm)")
    p.add_argument("--max_chord_ratio",       type=float, default=0.40,
                   help="Max chord-deviation ratio; rejects V-kinks (0 = disable). "
                        "Ratio = max perpendicular distance from endpoint chord / chord length.")
    p.add_argument("--max_sinuosity",          type=float, default=1.5,
                   help="Max sinuosity (arc/chord) for a qualified wire. "
                        "Removes near-loop traces (sinuosity > 1.5). Physical wires "
                        "with S-curves typically have sinuosity < 1.3.")
    # Wire merge pass
    p.add_argument("--merge_gap_um",           type=float, default=1.0,
                   help="Max gap between wire endpoints to attempt merge (µm).")
    p.add_argument("--merge_angle",            type=float, default=30.0,
                   help="Proximity cone half-angle (°): the other endpoint must "
                        "lie within this angle of each wire's outgoing direction.")
    p.add_argument("--merge_join_angle",       type=float, default=25.0,
                   help="Max join curvature (°): angle between the two wires' "
                        "outgoing directions at the join point. Enforces smooth "
                        "continuation — prevents back-bending merges.")
    p.add_argument("--merge_chord",            type=float, default=0.25,
                   help="Max chord-deviation ratio for a merged wire.")
    p.add_argument("--border_um",              type=float, default=0.5,
                   help="Width of the edge exclusion border (µm). Wire endpoints "
                        "within this distance of the image frame are excluded from "
                        "merging, and wires with ANY coordinate inside this border "
                        "are removed from the final output.")
    p.add_argument("--no_merge",    action="store_true", default=False,
                   help="Disable the post-tracing wire merge pass.")
    # Shared
    p.add_argument("--max_gap_um",            type=float, default=2.0)
    p.add_argument("--max_bridge_angle",      type=float, default=25.0)
    p.add_argument("--no_frangi",    action="store_true", default=False)
    p.add_argument("--frangi_thresh",         type=float, default=0.01)
    p.add_argument("--frangi_sigmas",         type=float, nargs=2, default=[2,4])
    p.add_argument("--vlm_frangi_sigmas",     type=float, nargs=2, default=[1,3])
    p.add_argument("--seed",                  type=int,   default=42)
    # SEM
    p.add_argument("--min_len_um",            type=float, default=5.0)
    p.add_argument("--min_diam_nm",           type=float, default=10.0)
    p.add_argument("--max_diam_nm",           type=float, default=400.0)
    p.add_argument("--length_penalty_um",     type=float, default=20.0)
    p.add_argument("--thresh_frac",           type=float, default=0.85)
    p.add_argument("--clahe_clip",            type=float, default=2.0)
    p.add_argument("--scale_nm",              type=float, default=10_000.0)
    p.add_argument("--fallback_nm_per_px",    type=float, default=25.0)
    # VLM
    p.add_argument("--vlm_min_len_um",        type=float, default=10.0)
    p.add_argument("--vlm_scale_um",          type=float, default=None,
                   help="Override VLM scale bar in µm; leave unset for auto-calibration.")
    p.add_argument("--vlm_thresh",            type=float, default=60.0)
    p.add_argument("--vlm_frangi_thresh",     type=float, default=0.005)
    return p.parse_args()


def is_sem(p): return p.suffix.lower() in (".tif",".tiff")
def is_vlm(p): return p.suffix.lower() in (".png",".jpg",".jpeg")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    paths = []
    if args.input_dir:
        for ext in ("*.tif","*.tiff","*.TIF","*.TIFF","*.png","*.PNG","*.jpg","*.JPG"):
            paths.extend(sorted(args.input_dir.rglob(ext)))
    for p in args.images: paths.append(Path(p))

    if not paths:
        print("No image files found."); sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_sem = sum(is_sem(p) for p in paths)
    n_vlm = sum(is_vlm(p) for p in paths)
    print(f"Output : {args.output_dir.resolve()}")
    print(f"Images : {len(paths)}  ({n_sem} SEM, {n_vlm} VLM)")
    print(f"Seed   : {args.seed}  |  search_distance={args.search_distance}  "
          f"|  chord={args.max_chord_ratio}  si={args.max_sinuosity}")
    print(f"Tracer : Nano1D pixel recall-vector  |  "
          f"Binary: {'Frangi' if not args.no_frangi else 'CLAHE+Otsu'}")

    all_results = []
    for path in paths:
        if is_sem(path):   result = process_sem(path, args, args.output_dir)
        elif is_vlm(path): result = process_vlm(path, args, args.output_dir)
        else: print(f"  Skipping: {path.name}"); continue
        if result: all_results.extend(result)

    if all_results:
        wire_results = [d for d in all_results if d.get("mode") != "coverage"]
        cov_results  = [d for d in all_results if d.get("mode") == "coverage"]
        n_imgs = len(set(d.get("image", "") for d in all_results))

        summary = args.output_dir / "all_images_summary.csv"
        fields = CSV_FIELDS_SEM
        rows_to_write = wire_results if wire_results else all_results
        with open(summary, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows_to_write)

        print(f"\n{'='*66}")
        if wire_results:
            al = np.array([d["length_um"] for d in wire_results])
            complete = [d for d in wire_results if not d.get("truncated", False)]
            ep2ep = sum(1 for d in complete if d.get("ep_to_ep"))
            print(f"  Done. {len(complete):,} complete + "
                  f"{len(wire_results)-len(complete)} truncated wires "
                  f"across {n_imgs} image(s).")
            print(f"  Tip-to-tip  : {ep2ep} ({ep2ep/max(len(complete),1):.0%} of complete)")
            print(f"  Length      : median={np.median(al):.1f}  "
                  f"p90={np.percentile(al,90):.1f}  max={al.max():.1f} um")
        if cov_results:
            covs = [d["coverage_pct"] for d in cov_results]
            print(f"  Coverage    : {len(cov_results)} image(s), "
                  f"mean {np.mean(covs):.1f}% network fill")
            for d in cov_results:
                print(f"    {d['image']}: {d['coverage_pct']:.1f}% "
                      f"({d['field_um_w']:.0f}x{d['field_um_h']:.0f} µm field)")
        print(f"  Summary   -> {summary}")
    else:
        print("\nNo wires or coverage data measured.")


if __name__ == "__main__":
    main()