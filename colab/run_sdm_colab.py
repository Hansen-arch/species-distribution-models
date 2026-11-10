#!/usr/bin/env python3
"""Run the SDM panel on a Google Colab VM.

    colab run --gpu T4 colab/run_sdm_colab.py --species all

The module provisions what the VM does not already have (a JRE for MaxEnt, the
MaxEnt jar, the Python packages), runs the pipeline, and prints the comparison
table so the local terminal receives it through the CLI's stdout stream. With a
Drive mount the artefacts are also copied there, because ``colab run`` tears the
VM down when the script exits.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAXENT_URL = (
    "https://raw.githubusercontent.com/mrmaxent/Maxent/master/"
    "ArchivedReleases/3.4.4/maxent.jar"
)
RESULTS = ROOT / "results"


def step(message: str) -> None:
    print(f"[colab] {message}", flush=True)


def shell(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:], file=sys.stderr)
        print(result.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"command failed: {' '.join(command)}")


def ensure_java() -> None:
    if shutil.which("java"):
        step("java present")
        return
    step("installing a JRE for MaxEnt")
    shell(["apt-get", "update", "-qq"])
    shell(["apt-get", "install", "-y", "-qq", "default-jre-headless"])


def ensure_maxent() -> Path:
    jar = ROOT / "vendor" / "maxent.jar"
    jar.parent.mkdir(parents=True, exist_ok=True)
    if jar.exists():
        return jar
    step("downloading maxent.jar 3.4.4")
    import requests

    response = requests.get(MAXENT_URL, timeout=300)
    response.raise_for_status()
    jar.write_bytes(response.content)
    return jar


def ensure_packages() -> None:
    """Install only what the Colab image does not ship."""
    import importlib

    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "rasterio": "rasterio",
        "matplotlib": "matplotlib",
        "requests": "requests",
        "xgboost": "xgboost",
        "tabulate": "tabulate",
    }
    missing = []
    for module, package in required.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        step(f"installing {missing}")
        shell([sys.executable, "-m", "pip", "install", "-q", *missing])
    else:
        step("python packages already present")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", default="all")
    parser.add_argument("--models", default="svm,rf,maxent,xgboost")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--background", type=int, default=3000)
    parser.add_argument("--max-records", type=int, default=9000)
    args, extra = parser.parse_known_args()

    step(f"python {sys.version.split()[0]}")
    ensure_java()
    ensure_maxent()
    ensure_packages()

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_sdm.py"),
        "--species",
        args.species,
        "--models",
        args.models,
        "--background",
        str(args.background),
        "--max-records",
        str(args.max_records),
    ]
    if args.fast:
        command.append("--fast")
    command.extend(extra)

    step("running the panel")
    subprocess.run(command, check=True)

    table_path = RESULTS / "comparison_table.csv"
    if table_path.exists():
        step("comparison table")
        print(table_path.read_text(), flush=True)

    drive = Path("/content/drive/MyDrive")
    if drive.exists():
        target = drive / "sdm-results"
        step(f"copying artefacts to {target}")
        shutil.copytree(RESULTS, target, dirs_exist_ok=True)
    else:
        step("no Drive mount, artefacts stay on the VM until it is released")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
