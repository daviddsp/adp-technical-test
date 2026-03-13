# HR Topic Classification Project

Fine-tuned **DistilBERT** model for HR message routing across 8 categories.

## Project Architecture
- `data/split/`: Partitioned datasets (Train/Val/Test) to prevent leakage.
- `saved_model/`: Final fine-tuned weights and tokenizer.
- `notebooks/`: Evaluation visuals and confidence analysis.
- `prepare_data.py`: Pre-processing script for stratified data splitting.
- `train.py`: Full training pipeline with EarlyStopping.
- `predict.py`: Core inference logic (threshold: 0.60).
- `pyproject.toml`: Modern dependency management via **uv**.

## Getting Started

### 1. Environment Setup
Requires [uv](https://github.com/astral-sh/uv).
```bash
uv sync
```

### 2. Data Preparation
Splits original data into train/val/test files:
```bash
uv run prepare_data.py
```

### 3. Training
Fine-tunes the model and exports the best checkpoint to `./saved_model`:
```bash
uv run train.py
```

### 4. Evaluation & Tests
To run requirements-based tests:
```bash
uv run test_model.py
```
To explore detailed metrics and charts:
```bash
uv run jupyter notebook notebooks/distilbert_evaluation.ipynb
```

## Core Features
- **Semantic Understanding:** Transformer-based architecture to resolve keyword ambiguity.
- **Safety Threshold:** Rejects low-confidence predictions (< 60%) as "Unsupported".
- **Reproduction:** Locked dependencies and fixed random seeds for stable results.
