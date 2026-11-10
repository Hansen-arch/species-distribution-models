#!/usr/bin/env python3
"""Bootstrap and run the panel inside a Colab VM.

Uploaded with the project bundle and executed through ``colab exec -f``, so
arguments arrive in the SDM_ARGS environment variable rather than argv.

    colab exec -s sdm --timeout 3600 -f colab/remote_run.py --env SDM_ARGS="--species all"
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

BUNDLE = Path("/content/sdm_bundle.tar.gz")
ROOT = Path("/content/sdm")
RESULTS_TAR = Path("/content/sdm_results.tar.gz")
MAXENT_URL = (
    "https://raw.githubusercontent.com/mrmaxent/Maxent/master/"
    "ArchivedReleases/3.4.4/maxent.jar"
)
PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "rasterio": "rasterio",
    "matplotlib": "matplotlib",
    "requests": "requests",
    "xgboost": "xgboost",
    "tabulate": "tabulate",
}


def step(message: str) -> None:
    print(f"[remote] {message}", flush=True)


def shell(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    if not (ROOT / "scripts" / "run_sdm.py").exists():
        step("unpacking the project bundle")
        ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(BUNDLE) as bundle:
            bundle.extractall(ROOT)

    # A Colab VM has no X display and MaxEnt opens a Swing window even in batch
    # mode, so xvfb is needed. Java is often preinstalled while xvfb is not, so
    # the two are checked separately.
    if not (shutil.which("java") and shutil.which("xvfb-run")):
        step("installing a JRE and xvfb for MaxEnt")
        shell(["apt-get", "update", "-qq"])
        shell(["apt-get", "install", "-y", "-qq", "default-jre-headless", "xvfb", "xauth"])
    step(f"java: {shutil.which('java')}, xvfb: {shutil.which('xvfb-run')}")

    jar = ROOT / "vendor" / "maxent.jar"
    if not jar.is_file():
        raise SystemExit(
            "vendor/maxent.jar is missing from the uploaded bundle. "
            "MaxEnt cannot run without it; check the excludes in run_on_colab.sh."
        )
    step(f"maxent.jar present ({jar.stat().st_size / 1e6:.1f} MB)")

    # xvfb-run silently needs xauth to start its server; if it fails, Java
    # falls back to a headless JVM and MaxEnt cannot build its window.
    try:
        probe = __import__("subprocess").run(
            ["xvfb-run", "-a", "java", "-version"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as error:  # a probe must never stop the run
        step(f"xvfb probe failed: {type(error).__name__}")
        probe = None
    if probe is not None:
        first = ((probe.stderr or probe.stdout).strip().splitlines() or [""])[0]
        step(f"xvfb probe rc={probe.returncode}: {first}")

    jar = ROOT / "vendor" / "maxent.jar"
    if not jar.exists():
        step("downloading maxent.jar 3.4.4")
        import urllib.request

        jar.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MAXENT_URL, jar)

    missing = []
    for module, package in PACKAGES.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        step(f"installing {missing}")
        shell([sys.executable, "-m", "pip", "install", "-q", *missing])
    else:
        step("python packages already present")

    import os

    args = os.environ.get("SDM_ARGS", "--species all").split()
    step(f"running the panel with {args}")
    # Capture rather than inherit: a child writing to the inherited descriptor
    # does not reliably reach the notebook stream that colab exec forwards.
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_sdm.py"), *args],
        capture_output=True,
        text=True,
    )
    print(result.stdout[-20000:], flush=True)
    if result.returncode != 0:
        print(result.stderr[-8000:], file=sys.stderr, flush=True)
        raise SystemExit(f"pipeline exited with {result.returncode}")

    table = ROOT / "results" / "comparison_table.csv"
    if table.exists():
        step("comparison table")
        print(table.read_text(), flush=True)

    step("packing artefacts")
    with tarfile.open(RESULTS_TAR, "w:gz") as bundle:
        bundle.add(ROOT / "results", arcname="results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
