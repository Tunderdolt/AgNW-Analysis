"""
run_batch_analysis.py
=====================
Batch runner for the AgNW nanowire analysis pipeline.

Usage
-----
    python run_batch_analysis.py /path/to/database/folder
    python run_batch_analysis.py /path/to/database --dry_run   # preview only
    python run_batch_analysis.py /path/to/database --jobs 4    # 4 parallel CPUs

The script will:
  1. Walk the database folder, skipping "Absorbance data" and
     "Initial Report image analysis" (and any hidden folders).
  2. Parse each subfolder name of the form:
         [imager] SEM|VLM taken [date] of samples from [date]
     extracting modality (SEM/VLM) and synthesis date.
  3. Parse each image filename to extract sample (A-G, Alpha, Beta, Gamma)
     and magnification, in any order and with or without separators.
  4. Run nanowire_analysis.py on every qualifying image.
  5. Package outputs into a timestamped zip:

       AgNW_Results_<YYYYMMDD_HHMMSS>.zip
         ├-- SEM/
         |   +-- <synthesis_date>/
         |       +-- <sample>/
         |           +-- <magnification>/
         |               ├-- <image>_overlay.png
         |               ├-- <image>_distributions.png
         |               ├-- <image>_raw_skeleton.png
         |               +-- <image>_all_wires.csv
         ├-- VLM/
         |   +-- ...
         ├-- all_samples_summary.csv   ← all wires, all images, one file
         +-- AgNW_Analysis_Guide.html

Requirements
------------
    pip install tifffile pillow numpy scipy scikit-image matplotlib opencv-python-headless

    nanowire_analysis.py must be in the same directory as this script, or:
        export NANOWIRE_SCRIPT=/full/path/to/nanowire_analysis.py
"""

import os, re, sys, csv, zipfile, logging, argparse, tempfile, traceback
from pathlib import Path
from datetime import datetime

# --- Configuration ------------------------------------------------------------

SKIP_FOLDER_SUBSTRINGS = ["absorbance", "initial report"]
SAMPLE_NAMES           = {"a","b","c","d","e","f","g","alpha","beta","gamma"}
SEM_EXTENSIONS         = {".tif",".tiff"}
VLM_EXTENSIONS         = {".png",".jpg",".jpeg"}

DEFAULT_SCRIPT  = Path(__file__).parent / "nanowire_analysis.py"
NANOWIRE_SCRIPT = Path(os.environ.get("NANOWIRE_SCRIPT", str(DEFAULT_SCRIPT)))
GUIDE_HTML      = Path(__file__).parent / "AgNW_Analysis_Guide.html"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("batch")

# --- Date normalisation -------------------------------------------------------

# Date patterns tried in order — most specific first
_DATE_PATTERNS = [
    (r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})\b',
     ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"]),
    (r'\b(\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})\b',
     ["%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d"]),
    (r'\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
     r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
     r'Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b',
     ["%d %B %Y", "%d %b %Y"]),
    (r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
     r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|'
     r'Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b',
     ["%B %Y", "%b %Y"]),
    (r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})\b',
     ["%d/%m/%y", "%d-%m-%y"]),
]

def _extract_date(text: str) -> str | None:
    """Find and parse any recognisable date anywhere in text. Returns YYYY-MM-DD or None."""
    for pattern, fmts in _DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            for fmt in fmts:
                try:
                    return datetime.strptime(m.group(1), fmt).strftime("%Y-%m-%d")
                except ValueError:
                    pass
    return None

# --- Folder name parser -------------------------------------------------------

def parse_folder_name(name: str) -> dict | None:
    """Return {modality, synthesis_date} or None if folder should be skipped."""
    nl = name.lower()
    for sub in SKIP_FOLDER_SUBSTRINGS:
        if sub in nl: return None

    modality = ("SEM" if re.search(r'\bsem\b', nl)
                else "VLM" if re.search(r'\bvlm\b', nl)
                else None)

    m = re.search(r"samples?\s+from\s+(.+?)(?:\s*$)", name, re.IGNORECASE)
    if m:
        raw = m.group(1).strip().rstrip(") ")
        synthesis_date = _extract_date(raw) or re.sub(r'[<>:"/\\|?*]', '-', raw)
    else:
        synthesis_date = re.sub(r'[<>:"/\\|?*]', '-', name.strip())
        log.warning("No 'samples from' phrase in %r — using full folder name", name)

    return {"modality": modality, "synthesis_date": synthesis_date}

# --- Image filename parser ----------------------------------------------------

def parse_image_name(filename: str) -> dict:
    """Extract sample name and magnification; order- and separator-independent."""
    stem = re.sub(r'\.(tif|tiff|png|jpg|jpeg)$', '', filename, flags=re.IGNORECASE)
    # Keep "1000x" as one token
    stem = re.sub(r'(\d+)([xX])(?=\D|$)', r'\1x ', stem)
    # Split at letter↔digit boundaries (but not x↔digit)
    s = re.sub(r'([A-Ww])(\d)', r'\1 \2', stem)
    s = re.sub(r'(\d)([A-Wa-wY-Zy-z])', r'\1 \2', s)
    tokens = [t.strip() for t in re.split(r'[\s_\-]+', s)
              if t.strip() and t.strip().lower() != 'x']

    samples = [t for t in tokens if t.lower() in SAMPLE_NAMES]
    mags    = [t for t in tokens if re.fullmatch(r'\d{2,5}x?', t, re.IGNORECASE)]

    sample = samples[0].capitalize() if samples else "Unknown"
    mag    = re.sub(r'x$','',mags[0],flags=re.IGNORECASE)+"x" if mags else "Unknown"
    return {"sample": sample, "magnification": mag}

# --- Directory discovery ------------------------------------------------------

def _find_folder_info(dirpath: Path, root: Path) -> dict | None:
    cur = dirpath
    while cur != root and cur != cur.parent:
        fi = parse_folder_name(cur.name)
        if fi is not None: return fi
        cur = cur.parent
    return None

def _safe(s: str) -> str:
    return re.sub(r'[^\w\-]', '_', s)

def discover_jobs(root: Path) -> list[dict]:
    jobs = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        dirpath = Path(dirpath)
        for fname in sorted(filenames):
            ext = Path(fname).suffix.lower()
            if   ext in SEM_EXTENSIONS: file_mod = "SEM"
            elif ext in VLM_EXTENSIONS: file_mod = "VLM"
            else: continue

            fi = _find_folder_info(dirpath, root)
            if fi is None: continue
            if fi.get("modality") and fi["modality"] != file_mod: continue

            jobs.append({
                "folder_info": fi,
                "image_path":  dirpath / fname,
                "image_info":  parse_image_name(fname),
                "modality":    fi.get("modality") or file_mod,
            })
    log.info("Discovered %d images", len(jobs))
    return jobs

# --- Pipeline runner ----------------------------------------------------------

def run_job(job: dict, tmp_dir: Path) -> list[Path]:
    """
    Run the nanowire analysis on a single image by importing and calling
    nanowire_analysis.py directly -- no subprocess, so this works inside
    a PyInstaller exe where sys.executable points back to the exe itself.
    """
    fi, ii = job["folder_info"], job["image_info"]
    out = (tmp_dir / job["modality"]
           / _safe(fi["synthesis_date"])
           / _safe(ii["sample"])
           / _safe(ii["magnification"]))
    out.mkdir(parents=True, exist_ok=True)

    try:
        import importlib.util as _ilu, argparse, traceback
        import numpy as _np

        _spec = _ilu.spec_from_file_location("nanowire_analysis", str(NANOWIRE_SCRIPT))
        _na   = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_na)

        # Minimal args namespace with all defaults
        args = argparse.Namespace(
            output_dir=out, n_sample=300,
            search_distance=30, max_chord_ratio=0.40, max_sinuosity=1.5,
            merge_gap_um=1.0, merge_angle=30.0, merge_join_angle=25.0,
            merge_chord=0.25, border_um=0.5, no_merge=False,
            max_gap_um=2.0, max_bridge_angle=25.0,
            no_frangi=False, frangi_thresh=0.01,
            frangi_sigmas=[2.0, 4.0], vlm_frangi_sigmas=[1.0, 3.0],
            seed=42,
            min_len_um=5.0, min_diam_nm=10.0, max_diam_nm=400.0,
            length_penalty_um=20.0, thresh_frac=0.85, clahe_clip=2.0,
            scale_nm=10000.0, fallback_nm_per_px=25.0,
            vlm_min_len_um=10.0, vlm_scale_um=None,
            vlm_thresh=60.0, vlm_frangi_thresh=0.005,
        )

        _np.random.seed(args.seed)
        img_path = job["image_path"]

        if job["modality"] == "SEM":
            result = _na.process_sem(img_path, args, out)
        else:
            result = _na.process_vlm(img_path, args, out)

        if result is None:
            raise RuntimeError("process returned None")

        produced = [p for p in out.iterdir() if p.is_file()]
        log.info("  -> %d files  [%s]", len(produced), out.relative_to(tmp_dir))
        return produced

    except Exception:
        msg = traceback.format_exc()
        log.error("FAILED: %s", job["image_path"].name)
        log.error(msg[-600:])
        err = out / f"{job['image_path'].stem}_ERROR.txt"
        err.write_text(msg)
        return [err]
# --- Summary CSV --------------------------------------------------------------

def build_summary(tmp_dir: Path, out_path: Path) -> int:
    rows, fields = [], None
    extra = ["modality","synthesis_date","sample","magnification"]
    for csv_path in sorted(tmp_dir.rglob("*_all_wires.csv")):
        parts = csv_path.relative_to(tmp_dir).parts
        ctx = {
            "modality":       parts[0]                    if len(parts)>1 else "?",
            "synthesis_date": parts[1].replace("_","-")   if len(parts)>2 else "?",
            "sample":         parts[2]                    if len(parts)>3 else "?",
            "magnification":  parts[3].replace("_","")    if len(parts)>4 else "?",
        }
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fields is None:
                fields = extra + [c for c in (reader.fieldnames or []) if c not in extra]
            for row in reader:
                row.update(ctx); rows.append(row)
    if not rows:
        out_path.write_text("No data\n"); return 0
    with open(out_path,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    return len(rows)

# --- Combined sample statistics -----------------------------------------------

def combine_sample_results(tmp_dir: Path) -> int:
    """
    Find groups of images sharing (modality, date, sample, mag) and produce
    combined CSVs, distribution plots, and provenance notes for each group.
    Returns the number of groups combined.
    """
    import math
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from scipy.stats import gaussian_kde

    groups: dict = {}
    for csv_path in sorted(tmp_dir.rglob("*_all_wires.csv")):
        parts = csv_path.relative_to(tmp_dir).parts
        if len(parts) < 5 or "_combined_" in csv_path.name:
            continue
        key = (parts[0], parts[1], parts[2], parts[3])
        groups.setdefault(key, []).append(csv_path)

    n_combined = 0
    for (modality, date, sample, mag), csv_paths in groups.items():
        if len(csv_paths) < 2:
            continue
        log.info("Combining %d images -> %s/%s/%s/%s",
                 len(csv_paths), modality, date, sample, mag)

        all_rows, fieldnames, source_images = [], None, []
        for p in sorted(csv_paths):
            with open(p, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames or [])
                for row in reader:
                    all_rows.append(row)
            source_images.append(p.stem.replace("_all_wires", ""))

        if not all_rows or not fieldnames:
            continue

        out_dir = tmp_dir / modality / date / sample / mag
        label   = f"{sample}_{mag}_combined"

        with open(out_dir / f"{label}_all_wires.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader(); w.writerows(all_rows)

        complete = [r for r in all_rows if r.get("truncated", "False") == "False"]
        if not complete:
            continue

        def _fl(rows, col):
            out = []
            for r in rows:
                try:
                    v = float(r[col])
                    if not math.isnan(v) and v > 0: out.append(v)
                except (ValueError, KeyError): pass
            return np.array(out)

        lengths   = _fl(complete, "length_um")
        diameters = _fl(complete, "diameter_nm") if modality == "SEM" else np.array([])
        aspects   = _fl(complete, "aspect_ratio") if modality == "SEM" else np.array([])
        angles    = _fl(complete, "wire_angle_deg")
        has_diam  = modality == "SEM" and len(diameters) > 0
        has_ar    = has_diam and len(aspects) > 0
        n_cols    = 1 + int(has_diam) + int(has_ar) + 1

        fig, axes = plt.subplots(1, n_cols, figsize=(7*n_cols, 5))
        if n_cols == 1: axes = [axes]
        clean_date = date.replace("_", "-")
        fig.suptitle(f"{sample} {mag} [{modality}] -- {len(csv_paths)} FOVs  "
                     f"N={len(complete)} wires  |  {clean_date}", fontsize=11, y=1.01)

        def _panel(ax, data, ch, ck, cm, xlabel, unit):
            if len(data) < 2:
                ax.text(0.5, 0.5, "Insufficient data", ha="center",
                        va="center", transform=ax.transAxes); return
            p99 = np.percentile(data, 99)
            ax.hist(data, bins=30, color=ch, edgecolor="white",
                    linewidth=0.4, alpha=0.85, density=True)
            try:
                kde = gaussian_kde(data, bw_method="silverman")
                x = np.linspace(data.min()*0.8, p99*1.15, 300)
                ax.plot(x, kde(x), color=ck, linewidth=2, label="KDE")
            except Exception: pass
            med = np.median(data)
            ax.axvline(med, color=cm, lw=1.5, ls="--",
                       label=f"median {med:.1f} {unit}")
            ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel("Density", fontsize=11)
            ax.set_title(f"mu={data.mean():.1f}, sigma={data.std():.1f} {unit}")
            ax.set_xlim(left=0); ax.legend(fontsize=9)
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        _panel(axes[0], lengths, "#4A90D9","#1A5FA8","#E05C2A","Wire length (um)","um")
        ax_i = 1
        if has_diam:
            _panel(axes[ax_i], diameters,"#E8994A","#A05A10","#1A5FA8","Diameter (nm)","nm")
            ax_i += 1
        if has_ar:
            _panel(axes[ax_i], aspects,"#7B5EA7","#4A2080","#E05C2A","Aspect ratio","")
            ax_i += 1
        if len(angles) >= 10:
            ori = axes[ax_i]
            ori.hist(angles, bins=18, range=(0,180), color="#4A90D9",
                     edgecolor="white", linewidth=0.4, density=True)
            ori.axhline(1.0/180, color="#E05C2A", ls="--", lw=1.2, label="Uniform")
            phi = 2.0*np.radians(angles)
            R   = float(np.sqrt(np.mean(np.cos(phi))**2+np.mean(np.sin(phi))**2))
            mu  = float(np.degrees(np.arctan2(np.mean(np.sin(phi)),
                                               np.mean(np.cos(phi))))/2 % 180)
            ori.set_title(f"Orientation  R={R:.3f}  mu={mu:.0f} deg")
            ori.set_xlabel("Wire orientation (deg)", fontsize=11)
            ori.set_ylabel("Density", fontsize=11)
            ori.set_xlim(0, 180); ori.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(out_dir / f"{label}_distributions.png", dpi=150, bbox_inches="tight")
        plt.close()

        (out_dir / f"{label}_sources.txt").write_text(
            f"Combined analysis\nSample: {sample}\nMagnification: {mag}\n"
            f"Modality: {modality}\nSynthesis date: {clean_date}\n"
            f"Fields of view: {len(csv_paths)}\nTotal wires: {len(complete)}\n"
            f"\nSource images:\n"
            + "\n".join(f"  {n}" for n in source_images) + "\n",
            encoding="utf-8")

        log.info("  -> %d wires from %d FOVs  [%s]",
                 len(complete), len(csv_paths), label)
        n_combined += 1

    return n_combined


# --- Zip packaging ------------------------------------------------------------

def package_zip(tmp_dir: Path, summary_csv: Path, zip_path: Path) -> None:
    arc_root = Path("AgNW_Results")
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for p in sorted(tmp_dir.rglob("*")):
            if p.is_file(): zf.write(p, arc_root / p.relative_to(tmp_dir))
        zf.write(summary_csv, arc_root/"all_samples_summary.csv")
        if GUIDE_HTML.exists():
            zf.write(GUIDE_HTML, arc_root/"AgNW_Analysis_Guide.html")
    log.info("Zip: %s  (%.1f MB)", zip_path.name, zip_path.stat().st_size/1e6)

# --- Main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Batch AgNW analysis — point at the database folder, get a zip.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("database_dir", type=Path, help="Root database folder")
    ap.add_argument("--output_dir", type=Path, default=None,
                    help="Where to save the zip (default: database_dir parent)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Preview discovered jobs without running the pipeline")
    ap.add_argument("--jobs", type=int, default=1, metavar="N",
                    help="Parallel workers (default 1). Set to CPU core count for speed.")
    args = ap.parse_args()

    root = args.database_dir.resolve()
    if not root.is_dir():
        log.error("Not a directory: %s", root); sys.exit(1)
    if not NANOWIRE_SCRIPT.exists():
        log.error("nanowire_analysis.py not found at %s\n"
                  "Set NANOWIRE_SCRIPT env var to its full path.", NANOWIRE_SCRIPT)
        sys.exit(1)

    out_dir = (args.output_dir or root.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path    = out_dir / f"AgNW_Results_{ts}.zip"
    summary_csv = out_dir / f"_summary_tmp_{ts}.csv"

    log.info("Database : %s", root)
    log.info("Output   : %s", zip_path)

    jobs = discover_jobs(root)
    if not jobs:
        log.error("No qualifying images found."); sys.exit(1)

    # Print discovery table
    hdr = f"\n  {'Modality':<10} {'Synth date':<16} {'Sample':<10} {'Mag':<10} Filename"
    bar = "-" * 76
    print(bar); print(hdr); print(bar)
    for j in jobs:
        fi, ii = j["folder_info"], j["image_info"]
        print(f"  {j['modality']:<10} {fi['synthesis_date']:<16} "
              f"{ii['sample']:<10} {ii['magnification']:<10} {j['image_path'].name}")
    print(bar)
    print(f"  Total: {len(jobs)} images\n")

    if args.dry_run:
        log.info("Dry run — stopping here."); sys.exit(0)

    with tempfile.TemporaryDirectory(prefix="agnw_") as tmp_str:
        tmp_dir = Path(tmp_str)
        n_ok = n_fail = 0

        if args.jobs > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                fmap = {pool.submit(run_job, j, tmp_dir): j for j in jobs}
                done = 0
                for fut in as_completed(fmap):
                    done += 1
                    j = fmap[fut]
                    try:
                        prod = fut.result()
                        ok = prod and not all(str(p).endswith("_ERROR.txt") for p in prod)
                        if ok: n_ok += 1
                        else:  n_fail += 1
                        log.info("[%d/%d] %s", done, len(jobs), j["image_path"].name)
                    except Exception as e:
                        log.error("[%d/%d] %s crashed: %s", done, len(jobs),
                                  j["image_path"].name, e)
                        n_fail += 1
        else:
            for i, j in enumerate(jobs, 1):
                fi, ii = j["folder_info"], j["image_info"]
                log.info("[%d/%d] %s  %s | %s | %s",
                         i, len(jobs), j["modality"],
                         fi["synthesis_date"], ii["sample"], ii["magnification"])
                prod = run_job(j, tmp_dir)
                if prod and not all(str(p).endswith("_ERROR.txt") for p in prod):
                    n_ok += 1
                else:
                    n_fail += 1

        log.info("Done: %d OK, %d failed", n_ok, n_fail)

        n_combined = combine_sample_results(tmp_dir)
        if n_combined:
            log.info("Combined %d sample groups with multiple FOVs", n_combined)
        n_rows = build_summary(tmp_dir, summary_csv)
        log.info("Summary: %d wire rows", n_rows)
        package_zip(tmp_dir, summary_csv, zip_path)

    summary_csv.unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print(f"  Results zip : {zip_path}")
    if n_fail: print(f"  ⚠  {n_fail} image(s) failed — see *_ERROR.txt in zip")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
