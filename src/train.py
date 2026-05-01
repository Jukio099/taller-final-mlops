"""
Pipeline de ML - Predicción del precio final por kg en subastas ganaderas.
Dataset: tabla 'subastas' extraída de Supabase (53 888 registros reales).
Modelo: XGBoost Regressor  |  Métricas: RMSE y R²
"""
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.sklearn


# ── 1. Configuración ──────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── 2. Carga de datos ─────────────────────────────────────────────────────────
def load_data(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    print(f"[carga]   {df.shape[0]:,} filas, {df.shape[1]} columnas")
    return df


# ── 3. Preprocesamiento ───────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    prep = cfg["preprocessing"]

    # Eliminar columnas no informativas
    df = df.drop(columns=prep["drop_columns"], errors="ignore")

    # Extraer características temporales desde fecha_subasta
    df["fecha_subasta"] = pd.to_datetime(df["fecha_subasta"], errors="coerce")
    df["mes"] = df["fecha_subasta"].dt.month
    df["dia_semana"] = df["fecha_subasta"].dt.dayofweek
    df = df.drop(columns=["fecha_subasta"])

    # Imputar nulos numéricos con la mediana
    for col in ["peso_total_kg", "peso_promedio_kg", "precio_base_kg"]:
        df[col] = df[col].fillna(df[col].median())

    # Eliminar outliers del target (percentiles 1 % – 99 %)
    target = prep["target"]
    q_low = df[target].quantile(prep["outlier_lower_pct"])
    q_high = df[target].quantile(prep["outlier_upper_pct"])
    before = len(df)
    df = df[(df[target] >= q_low) & (df[target] <= q_high)].copy()
    print(f"[limpieza] outliers eliminados: {before - len(df):,} filas")

    # Agrupar procedencias poco frecuentes como 'Otra'
    top_proc = (
        df["procedencia"]
        .value_counts()
        .nlargest(prep["top_procedencias"])
        .index
    )
    df["procedencia"] = df["procedencia"].where(
        df["procedencia"].isin(top_proc), other="Otra"
    )

    # Codificación de variables categóricas
    for col in ["tipo_subasta", "tipo_codigo", "procedencia"]:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    print(f"[limpieza] dataset final: {df.shape[0]:,} filas")
    return df


# ── 4. Entrenamiento ──────────────────────────────────────────────────────────
def train(X_train: np.ndarray, y_train: pd.Series, cfg: dict) -> XGBRegressor:
    m = cfg["model"]
    model = XGBRegressor(
        n_estimators=m["n_estimators"],
        max_depth=m["max_depth"],
        learning_rate=m["learning_rate"],
        subsample=m["subsample"],
        random_state=m["random_state"],
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# ── 5. Pipeline principal ─────────────────────────────────────────────────────
def main():
    cfg = load_config()

    # Carga
    df_raw = load_data(cfg["data"]["raw_path"])

    # Preprocesamiento
    df_clean = preprocess(df_raw, cfg)

    # Guardar CSV limpio para evidencia del profe
    df_clean.to_csv(cfg["data"]["clean_path"], index=False)
    print(f"[guardado] {cfg['data']['clean_path']}")

    # División train/test
    target = cfg["preprocessing"]["target"]
    X = df_clean.drop(columns=[target])
    y = df_clean[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["model"]["test_size"],
        random_state=cfg["model"]["random_state"],
    )

    # Escalado
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Entrenamiento
    print("[entrenamiento] XGBoost...")
    model = train(X_train_sc, y_train, cfg)

    # Evaluación
    y_pred = model.predict(X_test_sc)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    print(f"[métricas]  RMSE = {rmse:,.2f}  |  R² = {r2:.4f}")

    # ── MLflow tracking ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run() as run:
        # Parámetros
        mlflow.log_params({
            "n_estimators": cfg["model"]["n_estimators"],
            "max_depth": cfg["model"]["max_depth"],
            "learning_rate": cfg["model"]["learning_rate"],
            "subsample": cfg["model"]["subsample"],
            "test_size": cfg["model"]["test_size"],
            "top_procedencias": cfg["preprocessing"]["top_procedencias"],
            "outlier_lower_pct": cfg["preprocessing"]["outlier_lower_pct"],
            "outlier_upper_pct": cfg["preprocessing"]["outlier_upper_pct"],
        })

        # Métricas
        mlflow.log_metrics({"RMSE": rmse, "R2": r2})

        # Firma y ejemplo de entrada
        signature = infer_signature(X_train_sc, model.predict(X_train_sc))
        input_example = X_train_sc[:3]

        # Artefacto del modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="subastas_precio_model",
            signature=signature,
            input_example=input_example,
        )

        run_id = run.info.run_id

    print("=" * 55)
    print(f"  run_id : {run_id}")
    print(f"  RMSE   : {rmse:,.2f} COP/kg")
    print(f"  R²     : {r2:.4f}")
    print("=" * 55)
    print("  mlflow ui  ->  http://localhost:5000")
    print("=" * 55)


if __name__ == "__main__":
    main()
