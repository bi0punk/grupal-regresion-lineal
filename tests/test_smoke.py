"""Smoke tests para el notebook académico de regresión lineal.

Validan estructura del notebook sin ejecutarlo (pesa ~1.6MB con outputs).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "DESARROLLO_Trabajo_Grupal_2025_v2.ipynb"


def test_notebook_is_valid_json():
    assert NOTEBOOK.exists(), "Notebook no encontrado"
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert "cells" in data
    assert len(data["cells"]) > 0


def test_notebook_has_code_and_markdown_cells():
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    kinds = {cell.get("cell_type") for cell in data["cells"]}
    assert "code" in kinds
    assert "markdown" in kinds


def test_dataset_not_tracked():
    """El CSV de 13MB no debe commitearse."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    assert "historia_climatica.csv" not in tracked, "El dataset CSV está commiteado"


def test_requirements_listed():
    req = (REPO_ROOT / "requirements.txt").read_text()
    for dep in ("numpy", "pandas", "scikit-learn", "matplotlib"):
        assert dep in req, f"{dep} no está en requirements.txt"
