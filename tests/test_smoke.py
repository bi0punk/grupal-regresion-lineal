"""Smoke tests para el script académico de regresión lineal.

Validan estructura del repositorio y que el script sea sintácticamente
válido sin ejecutarlo (requiere dataset de 13MB y dependencias pesadas).
"""

from __future__ import annotations

import ast
import subprocess  # nosec B404
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "regression_analyzer.py"


def test_script_exists():
    assert SCRIPT.exists(), "regression_analyzer.py no encontrado"


def test_script_is_valid_python():
    ast.parse(SCRIPT.read_text(encoding="utf-8"))
    assert SCRIPT.stat().st_size > 0


def test_dataset_not_tracked():
    """El CSV de 13MB no debe commitearse."""
    tracked = subprocess.run(  # nosec B603, B607
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    assert "historia_climatica.csv" not in tracked, "El dataset CSV está commiteado"


def test_requirements_listed():
    req = (REPO_ROOT / "requirements.txt").read_text()
    for dep in ("numpy", "pandas", "scikit-learn", "matplotlib"):
        assert dep in req, f"{dep} no está en requirements.txt"
