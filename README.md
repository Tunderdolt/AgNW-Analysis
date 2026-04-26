# AgNW Analysis Pipeline

Automated measurement of silver nanowire (AgNW) length, diameter, aspect ratio, and orientation from SEM and VLM micrographs. Built around a hybrid implementation of the **Nano1D** skeleton-tracing algorithm.

---

## Algorithm

The pipeline is a hybrid approach combining Frangi vesselness filtering with the Nano1D pixel-level recall-vector tracing algorithm described in:

> Moradpur-Tari, E. et al. *Nano1D: An open-source automated image analysis tool to quantify dimensions of 1D nanostructures.* **Ultramicroscopy** 261, 113949 (2024). https://doi.org/10.1016/j.ultramic.2024.113949

The key steps are:

1. **Frangi vesselness filtering** (β = 0.3, σ = 2–4 px) enhances tubular structures and produces a cleaner binary than thresholding alone, especially at wire crossings.
2. **Zhang skeletonisation** reduces the binary to single-pixel paths. Each pixel is labelled by its Nano1D neighbour count (N=1 tail, N=2 section, N≥3 junction).
3. **Two-pass gap bridging** reconnects small breaks caused by the Frangi filter.
4. **Nano1D recall-vector tracing** follows the skeleton from each tail pixel. At junctions the last `search_distance` pixels form a rolling tangent estimate; the arm most aligned with that direction is chosen. This handles wire crossings far more robustly than single-segment PCA methods.
5. **Post-tracing merge pass** detects fragments split at junctions (both endpoints within 300 nm of a junction pixel) and reassembles them.
6. **Quality filters**: chord-deviation ratio, sinuosity, minimum length, and a 0.5 µm border exclusion zone.
7. **Diameter measurement** (SEM only): FWHM of greyscale profiles taken perpendicular to the wire at 12 evenly-spaced points.

---

## Files

| File | Description |
|---|---|
| `nanowire_analysis.py` | Core analysis pipeline — processes individual SEM/VLM images |
| `run_batch_analysis.py` | Batch runner — walks a database folder, processes all images, packages results into a zip |
| `summarise_statistics.py` | Computes per-sample statistics (mean ± SD, median, percentiles) from a summary CSV |
| `AgNW_Analyser_GUI.py` | Tkinter GUI for point-and-click batch analysis |
| `AgNW_Analyser.bat` | Windows double-click launcher — checks/installs dependencies, then runs the batch analysis |
| `AgNW_Analyser.spec` | PyInstaller spec file for building a standalone Windows exe |
| `hook_tcl.py` | PyInstaller runtime hook — sets TCL/TK paths inside the frozen exe |
| `AgNW_Analyser_Guide.html` | Full user guide |

---

## Requirements

Python 3.9 or newer.

```bash
pip install tifffile pillow numpy scipy scikit-image matplotlib opencv-python
```

For Excel output from `summarise_statistics.py`:
```bash
pip install openpyxl
```

For building the standalone exe:
```bash
pip install pyinstaller
```

---

## Quick start

### Single image
```bash
python nanowire_analysis.py F_1000x.tif --output_dir results/
```

### Folder of images
```bash
python nanowire_analysis.py --input_dir ./images/ --output_dir results/
```

### Full database batch run
```bash
python run_batch_analysis.py /path/to/database --output_dir /path/to/output
```

Preview what will be found without running the analysis:
```bash
python run_batch_analysis.py /path/to/database --dry_run
```

### Per-sample statistics from a batch summary CSV
```bash
python summarise_statistics.py all_samples_summary.csv --output stats.xlsx
```

---

## Folder naming convention (batch runner)

The batch runner parses database subfolders named in the format:

```
[Imager] SEM|VLM taken [date] of samples from [date of synthesis]
```

It extracts the **synthesis date** (not the imaging date) as the primary sample identifier. An alternate format with just the synthesis date at the end is also supported:

```
[Imager] SEM|VLM [date of synthesis]
```

Images within each folder should follow:

```
[sample] [magnification] [anything else].tif / .png
```

where `sample` is one of A–G, Alpha, Beta, Gamma and `magnification` is e.g. `1000x`, `50x`, `100x`. Order and separators are flexible.

---

## Output structure

Each image produces:

| File | Contents |
|---|---|
| `{name}_all_wires.csv` | Per-wire measurements (length, diameter, aspect ratio, sinuosity, orientation, …) |
| `{name}_overlay.png` | False-colour wire overlay on the original image (colour = length) |
| `{name}_distributions.png` | Length, diameter, aspect ratio histograms + orientation rose plot |
| `{name}_raw_skeleton.png` | Debug skeleton image (endpoints in green, junctions in red) |

The batch runner additionally produces:

| File | Contents |
|---|---|
| `all_samples_summary.csv` | All wires from all images in one file |
| `{sample}_{mag}_combined_*` | Combined outputs for samples with multiple fields of view |
| `AgNW_Results_YYYYMMDD_HHMMSS.zip` | Everything packaged for distribution |

---

## CSV fields

| Column | Unit | Description |
|---|---|---|
| `length_nm` / `length_um` | nm / µm | Arc length along skeleton |
| `diameter_nm` | nm | FWHM apparent diameter (SEM only, not PSF-corrected) |
| `aspect_ratio` | — | length / diameter (SEM only) |
| `sinuosity` | — | Arc length / chord length. 1.0 = straight |
| `wire_angle_deg` | 0–180° | Orientation of wire principal axis |
| `ep_to_ep` | bool | Both endpoints are genuine wire tips |
| `truncated` | bool | Wire touches the 0.5 µm border exclusion zone |
| `n_segments` | int | Number of fragments joined by the merge pass |

---

## Building the standalone exe (Windows)

```powershell
# Set TCL/TK paths for conda Python (adjust to your environment)
$env:TCL_LIBRARY = "C:\Users\<user>\.julia\conda\3\x86_64\Library\lib\tcl8.6"
$env:TK_LIBRARY  = "C:\Users\<user>\.julia\conda\3\x86_64\Library\lib\tk8.6"

& "C:\path\to\python.exe" -m PyInstaller AgNW_Analyser.spec
```

The exe appears in `dist/AgNW_Analyser.exe`. Note the `CONDA` path and TCL/TK paths in `AgNW_Analyser.spec` are hardcoded to the build machine and will need updating for a different environment.

---

## Notes on diameter measurements

Diameter values are apparent widths measured by FWHM of the greyscale intensity profile perpendicular to the wire. They include the electron-beam point-spread function and are systematically larger than the true wire diameter. Use them for within-instrument comparisons and aspect ratio calculations only — do not compare raw values between instruments without PSF deconvolution.

---

## Orientation statistics

Wire orientation R (mean resultant length) is computed using circular statistics on doubled angles to account for axial symmetry:

**R = √( mean(cos 2θ)² + mean(sin 2θ)² )**

R = 0 is fully isotropic; R = 1 is perfectly aligned. The orientation angle µ is the mean preferred direction (0–180°).
