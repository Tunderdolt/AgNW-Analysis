"""
AgNW_Analyser_GUI.py
====================
Double-click launcher for the AgNW batch analysis pipeline.

Runs the entire analysis IN-PROCESS (no subprocess calls) so it works
correctly when packaged as a PyInstaller exe on Windows.

Build:
    & "C:\\path\\to\\python.exe" -m PyInstaller AgNW_Analyser.spec
"""

# freeze_support must be first executable line
import multiprocessing
multiprocessing.freeze_support()

import sys, os, threading, logging, queue, traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime

# Locate sibling scripts - works both as .py and as frozen .exe.
# When frozen by PyInstaller, scripts bundled as datas land in sys._MEIPASS
# (the temporary extraction folder), NOT next to the exe.
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = Path(sys._MEIPASS)
else:
    SCRIPT_DIR = Path(__file__).parent

sys.path.insert(0, str(SCRIPT_DIR))

BG=      "#1e1e1e"
PANEL=   "#2d2d2d"
ACCENT=  "#4ec9b0"
TEXT=    "#d4d4d4"
MUTED=   "#808080"
SUCCESS= "#4ec9b0"
ERROR=   "#f48771"
WARN=    "#ce9178"


class QueueHandler(logging.Handler):
    """Sends log records into a queue for the GUI to display."""
    def __init__(self, q):
        super().__init__()
        self._q = q

    def emit(self, record):
        msg = self.format(record)
        lvl = record.levelno
        if lvl >= logging.ERROR:
            tag = "err"
        elif lvl >= logging.WARNING:
            tag = "warn"
        elif any(w in msg for w in ("Done","complete","saved","Zip","Results","Combined")):
            tag = "ok"
        else:
            tag = "info"
        self._q.put((msg, tag))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AgNW Batch Analyser")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(680, 520)
        w, h = 740, 600
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        self._q         = queue.Queue()
        self._running   = False
        self._stop_flag = threading.Event()
        self._dry_var   = tk.BooleanVar(value=False)

        self._build_ui()
        self.after(100, self._poll)
        self.after(300, self._check_imports)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        tk.Frame(self, bg=ACCENT, height=4).pack(fill="x")

        hdr = tk.Frame(self, bg=BG, pady=16)
        hdr.pack(fill="x", padx=24)
        tk.Label(hdr, text="AgNW Batch Analyser",
                 font=("Segoe UI", 18, "bold"), bg=BG, fg=ACCENT).pack(anchor="w")
        tk.Label(hdr, text="Automated silver nanowire measurement from SEM and VLM images",
                 font=("Segoe UI", 10), bg=BG, fg=MUTED).pack(anchor="w")

        frm = tk.Frame(self, bg=BG, padx=24)
        frm.pack(fill="x")
        self._db_var  = tk.StringVar()
        self._out_var = tk.StringVar()
        self._row(frm, "Database folder",
                  "Root folder containing SEM and VLM subfolders",
                  self._db_var, self._browse_db, 0)
        self._row(frm, "Output folder",
                  "Where the results zip will be saved",
                  self._out_var, self._browse_out, 1)

        opts = tk.Frame(self, bg=BG, padx=24, pady=6)
        opts.pack(fill="x")
        tk.Checkbutton(opts, text="Dry run  (discover images only, no analysis)",
                       variable=self._dry_var,
                       font=("Segoe UI", 10), bg=BG, fg=TEXT,
                       selectcolor=PANEL, activebackground=BG,
                       activeforeground=TEXT).pack(anchor="w")

        btns = tk.Frame(self, bg=BG, padx=24, pady=6)
        btns.pack(fill="x")
        self._run_btn = tk.Button(btns, text="  Run Analysis  ",
                                   font=("Segoe UI", 11, "bold"),
                                   bg=ACCENT, fg=BG, relief="flat",
                                   cursor="hand2", padx=16, pady=8,
                                   command=self._start)
        self._run_btn.pack(side="left")
        self._stop_btn = tk.Button(btns, text="  Stop  ",
                                    font=("Segoe UI", 11), bg=PANEL, fg=ERROR,
                                    relief="flat", cursor="hand2", padx=16, pady=8,
                                    state="disabled", command=self._stop)
        self._stop_btn.pack(side="left", padx=(10, 0))
        tk.Button(btns, text="Clear log", font=("Segoe UI", 10),
                  bg=PANEL, fg=MUTED, relief="flat", cursor="hand2",
                  padx=12, pady=8, command=self._clear).pack(side="right")

        self._bar = ttk.Progressbar(self, mode="indeterminate")
        self._bar.pack(fill="x", padx=24, pady=4)
        s = ttk.Style(); s.theme_use("default")
        s.configure("TProgressbar", troughcolor=PANEL, background=ACCENT, thickness=6)

        lf = tk.Frame(self, bg=BG, padx=24, pady=8)
        lf.pack(fill="both", expand=True)
        tk.Label(lf, text="Log", font=("Segoe UI", 9), bg=BG, fg=MUTED).pack(anchor="w")
        try:
            import tkinter.font as tkf
            fam = "Cascadia Code" if "Cascadia Code" in tkf.families() else "Consolas"
        except Exception:
            fam = "Consolas"
        self._box = scrolledtext.ScrolledText(
            lf, font=(fam, 9), bg=PANEL, fg=TEXT, insertbackground=TEXT,
            relief="flat", wrap="word", state="disabled")
        self._box.pack(fill="both", expand=True, pady=4)
        for t, c in [("ok", SUCCESS), ("err", ERROR), ("warn", WARN), ("info", TEXT)]:
            self._box.tag_config(t, foreground=c)
        self._log("AgNW Analyser ready.", "ok")

    def _row(self, p, label, hint, var, cmd, row):
        tk.Label(p, text=label, font=("Segoe UI", 10, "bold"),
                 bg=BG, fg=TEXT).grid(row=row*3, column=0, columnspan=2,
                                       sticky="w", pady=(10, 0))
        tk.Label(p, text=hint, font=("Segoe UI", 9),
                 bg=BG, fg=MUTED).grid(row=row*3+1, column=0, columnspan=2, sticky="w")
        tk.Entry(p, textvariable=var, font=("Segoe UI", 10), bg=PANEL, fg=TEXT,
                 insertbackground=TEXT, relief="flat", width=55
                 ).grid(row=row*3+2, column=0, sticky="ew", pady=2, ipady=6)
        tk.Button(p, text="Browse...", font=("Segoe UI", 10), bg=PANEL, fg=ACCENT,
                  relief="flat", cursor="hand2", padx=10, command=cmd
                  ).grid(row=row*3+2, column=1, sticky="w", padx=(6, 0))
        p.columnconfigure(0, weight=1)

    def _browse_db(self):
        p = filedialog.askdirectory(title="Select database root folder")
        if p:
            self._db_var.set(p)
            if not self._out_var.get():
                self._out_var.set(str(Path(p).parent))

    def _browse_out(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self._out_var.set(p)

    # ── Import check (in-process, no subprocess) ──────────────────────────────

    def _check_imports(self):
        missing = []
        for pkg, pip_name in [
            ("tifffile","tifffile"), ("PIL","pillow"), ("numpy","numpy"),
            ("scipy","scipy"), ("skimage","scikit-image"),
            ("matplotlib","matplotlib"), ("cv2","opencv-python"),
        ]:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pip_name)

        if missing:
            self._log(f"Missing packages: {', '.join(missing)}", "warn")
            self._log("pip install " + " ".join(missing), "warn")
            messagebox.showwarning("Missing packages",
                "These packages are missing:\n\n" +
                "\n".join(f"  - {m}" for m in missing) +
                "\n\nInstall them then restart.")
        else:
            self._log("All packages found.", "ok")

        for name in ("nanowire_analysis.py", "run_batch_analysis.py"):
            if not (SCRIPT_DIR / name).exists():
                self._log(f"ERROR: {name} not found in {SCRIPT_DIR}", "err")

    # ── Run ───────────────────────────────────────────────────────────────────

    def _start(self):
        db  = self._db_var.get().strip()
        out = self._out_var.get().strip() or None
        if not db:
            messagebox.showerror("No folder", "Please select a database folder.")
            return
        if not Path(db).is_dir():
            messagebox.showerror("Not found", f"Folder not found:\n{db}")
            return
        if not (SCRIPT_DIR / "run_batch_analysis.py").exists():
            messagebox.showerror("Missing", "run_batch_analysis.py not found.")
            return

        self._running = True
        self._stop_flag.clear()
        self._run_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._bar.start(12)
        self._clear()
        self._log(f"Database : {db}", "info")
        if out:
            self._log(f"Output   : {out}", "info")

        threading.Thread(target=self._worker, args=(db, out), daemon=True).start()

    def _worker(self, db, out):
        q_handler = QueueHandler(self._q)
        q_handler.setFormatter(logging.Formatter("%(message)s"))
        root_log = logging.getLogger()
        root_log.addHandler(q_handler)
        prev_level = root_log.level
        root_log.setLevel(logging.INFO)

        try:
            # Import batch module in-process
            import importlib.util as ilu
            spec = ilu.spec_from_file_location(
                "batch", str(SCRIPT_DIR / "run_batch_analysis.py"))
            batch = ilu.module_from_spec(spec)
            spec.loader.exec_module(batch)
            batch.NANOWIRE_SCRIPT = SCRIPT_DIR / "nanowire_analysis.py"

            root = Path(db)
            jobs = batch.discover_jobs(root)

            if not jobs:
                self._q.put(("No images found.", "warn"))
                self._q.put(("__ERR__", "err"))
                return

            self._q.put((f"Found {len(jobs)} images.", "ok"))

            if self._dry_var.get():
                for j in jobs:
                    fi, ii = j["folder_info"], j["image_info"]
                    self._q.put((
                        f"  {j['modality']:<5} "
                        f"{fi['synthesis_date']:<13} "
                        f"{ii['sample']:<7} "
                        f"{ii['magnification']:<7} "
                        f"{j['image_path'].name}", "info"))
                self._q.put(("Dry run done.", "ok"))
                self._q.put(("__OK__", "ok"))
                return

            import tempfile
            with tempfile.TemporaryDirectory(prefix="agnw_") as tmp_str:
                tmp = Path(tmp_str)
                n_ok = n_fail = 0

                for i, job in enumerate(jobs, 1):
                    if self._stop_flag.is_set():
                        self._q.put(("Stopped by user.", "warn"))
                        break
                    fi, ii = job["folder_info"], job["image_info"]
                    self._q.put((
                        f"[{i}/{len(jobs)}] {job['modality']} | "
                        f"{fi['synthesis_date']} | "
                        f"{ii['sample']} | {ii['magnification']} | "
                        f"{job['image_path'].name}", "info"))
                    try:
                        prod = batch.run_job(job, tmp)
                        if prod and not all(str(p).endswith("_ERROR.txt") for p in prod):
                            n_ok += 1
                        else:
                            n_fail += 1
                    except Exception as e:
                        self._q.put((f"  ERROR: {e}", "err"))
                        n_fail += 1

                self._q.put((f"Done: {n_ok} OK, {n_fail} failed.", "ok"))

                n_comb = batch.combine_sample_results(tmp)
                if n_comb:
                    self._q.put((f"Combined {n_comb} multi-FOV group(s).", "ok"))

                out_dir = Path(out) if out else root.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv = out_dir / f"_tmp_{ts}.csv"
                zip_path = out_dir / f"AgNW_Results_{ts}.zip"
                batch.build_summary(tmp, csv)
                batch.package_zip(tmp, csv, zip_path)
                csv.unlink(missing_ok=True)
                self._q.put((f"Results zip: {zip_path}", "ok"))

            self._q.put(("__OK__", "ok"))

        except Exception:
            self._q.put((traceback.format_exc(), "err"))
            self._q.put(("__ERR__", "err"))
        finally:
            root_log.removeHandler(q_handler)
            root_log.setLevel(prev_level)

    def _stop(self):
        self._stop_flag.set()
        self._log("Stop requested...", "warn")
        self._stop_btn.config(state="disabled")

    def _done(self):
        self._running = False
        self._bar.stop()
        self._run_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

    # ── Log ───────────────────────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                msg, tag = self._q.get_nowait()
                if msg == "__OK__":
                    self._done()
                    self._log("Analysis complete!", "ok")
                    messagebox.showinfo("Done",
                        "Analysis complete!\n\nCheck the output folder for your results zip.")
                elif msg == "__ERR__":
                    self._done()
                    self._log("Finished with errors. See log above.", "err")
                else:
                    self._log(msg, tag)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _log(self, text, tag="info"):
        self._box.config(state="normal")
        self._box.insert("end", text + "\n", tag)
        self._box.see("end")
        self._box.config(state="disabled")

    def _clear(self):
        self._box.config(state="normal")
        self._box.delete("1.0", "end")
        self._box.config(state="disabled")


if __name__ == "__main__":
    app = App()
    app.mainloop()
