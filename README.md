# Pipeline de ML — Predicción de Precio en Subastas Ganaderas

Pipeline reproducible de machine learning con tracking MLflow y automatización CI/CD mediante GitHub Actions.

---

## Por qué este dataset

La tabla **subastas** proviene de una base de datos propia construida en Supabase a partir de boletines reales de subastas ganaderas en Antioquia, Colombia. Contiene **53 888 registros** de lotes de ganado subastados entre 2023 y 2025.

Este dataset es relevante para mi emprendimiento: participo activamente en el sector ganadero de la región y necesito herramientas que me permitan anticipar el precio de venta por kilogramo antes de llevar el ganado a subasta. Predecir `precio_final_kg` a partir de variables como el tipo de animal, el peso, la procedencia y el precio base me da una ventaja real al momento de decidir cuándo y dónde vender.

**Fuente:** base de datos propia en Supabase, cargada desde PDFs oficiales de subastas ganaderas (Subasta Ganadera de Colombia – boletines públicos).

---

## Dataset: columnas principales

| Columna | Descripción |
|---|---|
| `tipo_subasta` | Tipo de evento (Tradicional, Especial GYR, Mulares) |
| `tipo_codigo` | Código del tipo de animal (HV=Hembra Vientre, HL=Hembra Levante, ML=Macho Levante…) |
| `peso_total_kg` | Peso total del lote en kg |
| `peso_promedio_kg` | Peso promedio por animal en kg |
| `procedencia` | Municipio de origen del ganado |
| `precio_base_kg` | Precio de salida por kg (COP) |
| `cantidad_animales` | Número de animales en el lote |
| `fecha_subasta` | Fecha de la subasta |
| **`precio_final_kg`** | **Variable objetivo: precio de cierre por kg (COP)** |

---

## Estructura del proyecto

```
taller-final-mlops/
├── .github/
│   └── workflows/
│       └── ml.yml          # Pipeline CI/CD GitHub Actions
├── data/
│   ├── subastas_raw.csv    # Datos originales extraídos de Supabase
│   └── subastas_clean.csv  # Generado automáticamente por train.py
├── src/
│   ├── train.py            # Script principal del pipeline
│   └── test_pipeline.py    # Pruebas de validación (pytest)
├── config.yaml             # Hiperparámetros y rutas
├── Makefile                # Automatización de tareas
├── requirements.txt        # Dependencias
└── README.md
```

---

## Instalación y uso local

### Requisitos
- Python 3.11+
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/taller-final-mlops.git
cd taller-final-mlops

# 2. Instalar dependencias
make install

# 3. (Opcional) Verificar que todo está en orden
make test

# 4. Ejecutar el pipeline completo
make train

# 5. Ver resultados en MLflow UI
mlflow ui
# Abrir: http://localhost:5000
```

También se puede ejecutar directamente:

```bash
python src/train.py
```

---

## Pipeline de ML

### Preprocesamiento (`src/train.py`)

1. **Eliminación de columnas** no informativas: `id`, `archivo_fuente`, `creado_en`, `hora_subasta`, `numero_lote`, `numero_boletin`.
2. **Ingeniería de características** desde `fecha_subasta`: extracción de `mes` y `dia_semana`.
3. **Imputación de nulos** en `peso_total_kg`, `peso_promedio_kg` y `precio_base_kg` con la mediana.
4. **Eliminación de outliers** del target: percentiles 1 % – 99 % (~1 060 filas removidas).
5. **Agrupación de procedencias** poco frecuentes como `"Otra"` (se conservan las 30 más comunes).
6. **Codificación** de variables categóricas con `LabelEncoder`.
7. **Escalado** con `StandardScaler`.

### Modelo

- **Algoritmo:** XGBoost Regressor
- **Tarea:** Regresión (predicción de precio continuo en COP/kg)
- **Métricas:** RMSE (error cuadrático medio) y R²

### Tracking con MLflow

Se registran:
- **Parámetros:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `test_size`, `top_procedencias`, percentiles de outlier.
- **Métricas:** `RMSE`, `R2`.
- **Modelo:** `mlflow.sklearn.log_model` con firma (`signature`) y ejemplo de entrada (`input_example`).
- **URI de tracking:** `mlruns/` (local, dentro del repositorio).

---

## CI/CD con GitHub Actions

El archivo `.github/workflows/ml.yml` ejecuta automáticamente en cada push a `main`:

| Paso | Comando |
|---|---|
| Instalar dependencias | `make install` |
| Lint con flake8 | `make lint` |
| Pruebas de validación | `make test` |
| Entrenamiento del modelo | `make train` |
| Guardar artefacto | `mlruns/` → artifact `mlflow-model-subastas` |

---

## Evidencia MLflow

Al ejecutar `make train` se genera la carpeta `mlruns/` con el experimento `subastas_precio_kg`.

Para ver el modelo registrado:
```bash
mlflow ui
```
Navegar a → **Experiments → subastas_precio_kg → [run más reciente] → Artifacts → subastas_precio_model**

---

## Configuración de hiperparámetros

Todos los parámetros se centralizan en `config.yaml`:

```yaml
model:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
  test_size: 0.2
  random_state: 42
```

Modifica estos valores y vuelve a ejecutar `make train` para experimentar.

---

## Dependencias principales

| Librería | Versión | Uso |
|---|---|---|
| pandas | 2.2.3 | Manipulación de datos |
| numpy | 1.26.4 | Cálculos numéricos |
| scikit-learn | 1.5.2 | Preprocesamiento y métricas |
| xgboost | 2.1.3 | Modelo de ML |
| mlflow | 2.19.0 | Tracking de experimentos |
| pyyaml | 6.0.2 | Lectura de config.yaml |
| pytest | 8.3.5 | Pruebas automatizadas |
| flake8 | 7.1.1 | Análisis de calidad del código |
