# regresion-lineal

[![CI](https://github.com/bi0punk/grupal-regresion-lineal/actions/workflows/ci.yml/badge.svg)](https://github.com/bi0punk/grupal-regresion-lineal/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Academic group project on temperature prediction using regression models for the course "Fundamentos del Machine Learning" (2025). Implements and compares multiple regression algorithms on historical climate data.

## Tabla de contenidos

- [Características](#características)
- [Stack](#stack)
- [Estructura](#estructura)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Tests](#tests)
- [CI](#ci)
- [Datos](#datos)
- [Limitaciones](#limitaciones)
- [Licencia](#licencia)

## Características

- Pipeline completo de regresión sobre datos climáticos históricos.
- Comparación de múltiples modelos: regresión lineal, Ridge, Lasso, ElasticNet, árbol de decisión.
- Imputación (KNN), escalado (StandardScaler), división train/test.
- Métricas: MSE, MAE, R², MAPE.
- Visualización con matplotlib/seaborn.

## Stack

- **Lenguaje**: Python 3.12+
- **Entorno**: Jupyter Notebook
- **ML**: scikit-learn (modelos, preprocesamiento, métricas)
- **Datos**: pandas, numpy
- **Visualización**: matplotlib, seaborn
- **Calidad**: ruff (excluye `.ipynb`), pytest

## Estructura

```
regresion-lineal/
├── DESARROLLO_Trabajo_Grupal_2025_v2.ipynb   # Notebook principal (entregable)
├── historia_climatica.csv                      # Dataset (NO commiteado, ver Datos)
├── requirements.txt
├── pyproject.toml                              # config pytest + ruff
├── tests/test_smoke.py                         # smoke tests del notebook
└── .github/workflows/ci.yml
```

## Requisitos

- Python 3.12+
- Jupyter (para ejecutar el notebook)

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

Coloca el dataset `historia_climatica.csv` en la raíz del repo (ver [Datos](#datos)) y abre el notebook:

```bash
jupyter notebook DESARROLLO_Trabajo_Grupal_2025_v2.ipynb
```

El notebook espera el CSV en el mismo directorio.

## Tests

```bash
pytest -q
```

Los smoke tests (`tests/test_smoke.py`) validan estructura del notebook (JSON válido, celdas code+markdown), que el dataset no esté commiteado y que las deps figuren en `requirements.txt`. No ejecutan el notebook (pesa ~1.6MB con outputs).

## CI

GitHub Actions (`.github/workflows/ci.yml`) sobre Python 3.12:

- `ruff check .` (excluye `.ipynb` para no mutar el entregable académico)
- `pytest -q`

## Datos

El dataset `historia_climatica.csv` (~13MB, ~96k filas horarias de clima 2006–) **no se commitea** por tamaño. Se excluye vía `.gitignore`. Para reproducir:

- Consíguelo de la fuente original del curso, o
- Si ya lo tienes localmente, colócalo en la raíz del repo antes de abrir el notebook.

Columnas: `Summary, Precip Type, Temperature (C), Humidity, Wind Speed (km/h), Wind Bearing (degrees), Visibility (km), Loud Cover, Pressure (millibars), Daily Summary, Fecha`.

## Limitaciones

- El notebook es el entregable académico; sus celdas no se auto-formatean ni se lintean para preservar outputs y flujo de ejecución.
- Sin endpoint/servicio: es un análisis batch en notebook.

## Licencia

MIT — ver [LICENSE](LICENSE).
