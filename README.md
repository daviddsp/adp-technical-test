# HR Topic Classification Project

Este proyecto clasifica mensajes de Recursos Humanos en 8 categorías utilizando un modelo **DistilBERT** ajustado (*fine-tuned*).

## Estructura del Proyecto
- `data/`: Contiene los datasets originales.
- `data/split/`: **(Nuevo)** Datasets divididos físicamente (`train.csv`, `val.csv`, `test.csv`).
- `notebooks/`: Evaluación visual y análisis del modelo (`distilbert_evaluation.ipynb`).
- `saved_model/`: El modelo final entrenado y su tokenizador.
- `prepare_data.py`: Script para realizar la división reproducible de datos.
- `train.py`: Pipeline de entrenamiento que consume los datos divididos.
- `predict.py`: Clase `TopicPredictor` para inferencia en producción.
- `pyproject.toml` / `uv.lock`: Gestión de dependencias con `uv`.

## Instalación rápida con `uv`
Este proyecto utiliza [uv](https://github.com/astral-sh/uv) para una gestión de dependencias ultra-rápida.

```bash
# 1. Sincronizar entorno y dependencias
uv sync

# 2. Preparar los datos (Split 70/15/15)
uv run prepare_data.py
```

## Uso

### Entrenamiento
Para re-entrenar el modelo usando los datos aislados:
```bash
uv run train.py
```

### Evaluación Visual
Para ver las métricas detalladas y gráficas (Matriz de Confusión, Distribución de Confianza):
```bash
uv run jupyter notebook notebooks/distilbert_evaluation.ipynb
```

### Pruebas de Requerimientos
Para validar que el modelo cumple con el umbral de confianza (60%) y el enrutamiento único:
```bash
uv run test_model.py
```

## Metodología Pro
- **Aislamiento Total:** Los datos de prueba se separan físicamente antes del entrenamiento para evitar el *Data Leakage*.
- **Umbral de Confianza:** Las predicciones con < 60% de confianza se marcan como "Operation not supported".
- **Reproducibilidad:** Uso de `uv.lock` y semillas fijas en el split de datos.
