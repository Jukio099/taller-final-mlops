"""
Pruebas básicas de validación del pipeline de ML.
Ejecutar con: pytest src/test_pipeline.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from train import load_config, load_data, preprocess, train  # noqa: E402


CFG_PATH = "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CFG_PATH)


@pytest.fixture(scope="module")
def df_raw(cfg):
    return load_data(cfg["data"]["raw_path"])


@pytest.fixture(scope="module")
def df_clean(cfg, df_raw):
    return preprocess(df_raw.copy(), cfg)


# ── Tests de datos ─────────────────────────────────────────────────────────────

def test_raw_csv_exists(cfg):
    assert os.path.exists(cfg["data"]["raw_path"]), \
        "No se encuentra data/subastas_raw.csv"


def test_raw_min_rows(df_raw):
    assert len(df_raw) > 1000, \
        f"Se esperaban >1000 filas, se encontraron {len(df_raw)}"


def test_raw_required_columns(df_raw):
    required = [
        "precio_final_kg", "tipo_subasta", "tipo_codigo",
        "peso_total_kg", "peso_promedio_kg", "procedencia",
        "precio_base_kg", "cantidad_animales",
    ]
    for col in required:
        assert col in df_raw.columns, f"Columna faltante: {col}"


def test_target_no_nulls(df_raw):
    assert df_raw["precio_final_kg"].isnull().sum() == 0, \
        "La variable objetivo tiene valores nulos"


# ── Tests de preprocesamiento ──────────────────────────────────────────────────

def test_clean_has_fewer_outliers(df_raw, df_clean, cfg):
    pct = cfg["preprocessing"]
    q_low = df_raw["precio_final_kg"].quantile(pct["outlier_lower_pct"])
    q_high = df_raw["precio_final_kg"].quantile(pct["outlier_upper_pct"])
    outliers_raw = ((df_raw["precio_final_kg"] < q_low) |
                    (df_raw["precio_final_kg"] > q_high)).sum()
    outliers_clean = ((df_clean["precio_final_kg"] < q_low) |
                      (df_clean["precio_final_kg"] > q_high)).sum()
    assert outliers_clean < outliers_raw, \
        "El preprocesamiento no eliminó outliers del target"


def test_clean_no_nulls(df_clean):
    total_nulls = df_clean.isnull().sum().sum()
    assert total_nulls == 0, f"Quedan {total_nulls} nulos tras limpiar"


def test_clean_categoricals_encoded(df_clean):
    for col in ["tipo_subasta", "tipo_codigo", "procedencia"]:
        assert df_clean[col].dtype in [np.int32, np.int64, int], \
            f"Columna '{col}' no fue codificada numéricamente"


def test_date_features_extracted(df_clean):
    assert "mes" in df_clean.columns, "Falta la columna 'mes'"
    assert "dia_semana" in df_clean.columns, "Falta la columna 'dia_semana'"


# ── Tests del modelo ───────────────────────────────────────────────────────────

def test_model_trains_and_predicts(df_clean, cfg):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    target = cfg["preprocessing"]["target"]
    X = df_clean.drop(columns=[target]).values[:500]
    y = df_clean[target].values[:500]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Entrenamiento rápido (pocos árboles) solo para verificar que funciona
    small_cfg = dict(cfg)
    small_cfg["model"] = dict(cfg["model"])
    small_cfg["model"]["n_estimators"] = 10

    model = train(X_train_sc, y_train, small_cfg)
    preds = model.predict(X_test_sc)

    assert len(preds) == len(y_test), "Número de predicciones incorrecto"
    assert all(np.isfinite(preds)), "El modelo generó predicciones no finitas"


def test_model_r2_positive(df_clean, cfg):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    target = cfg["preprocessing"]["target"]
    X = df_clean.drop(columns=[target]).values[:2000]
    y = df_clean[target].values[:2000]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    small_cfg = dict(cfg)
    small_cfg["model"] = dict(cfg["model"])
    small_cfg["model"]["n_estimators"] = 20

    model = train(X_train_sc, y_train, small_cfg)
    preds = model.predict(X_test_sc)
    r2 = r2_score(y_test, preds)

    assert r2 > 0, f"El modelo tiene R² negativo ({r2:.4f}), algo está muy mal"
