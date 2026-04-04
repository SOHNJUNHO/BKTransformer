# Knowledge Tracing

A simple knowledge tracing training pipeline built with PyTorch Lightning.

## Project Layout
```
knowledge-tracing/
  artifacts/
    checkpoints/
    lightning_logs/
  data/
    raw/
    processed/
      icecream_3rd.csv
      icecream_8th_processed.csv
      skill_builder_preprocessed.csv
  notebooks/
    assistment2009_preprocessing.ipynb
  src/
    knowledge_tracing/
      main.py
      train.py
      data/
        datasets.py
        validation.py
      models/
        bk_transformer.py
        bk_transformer_baseline.py
```

## Setup
```bash
uv sync --extra dev --frozen
```

`uv` will use the PyTorch CPU index automatically on Linux and PyPI on macOS.

Optional notebook dependencies:
```bash
uv sync --extra dev --extra notebooks --frozen
```

## Running Training
From the repo root:
```bash
uv run --frozen python -m knowledge_tracing.main \
  --data_path data/processed/skill_builder_preprocessed.csv \
  --model_type upgraded \
  --max_epochs 50
```

Checkpoints and logs will be written to `artifacts/checkpoints` and `artifacts/lightning_logs`.

## Key Files
- `src/knowledge_tracing/main.py`: CLI entrypoint.
- `src/knowledge_tracing/data/datasets.py`: preferred dataset loader and collate module.
- `src/knowledge_tracing/models/bk_transformer.py`: preferred upgraded model path.
- `src/knowledge_tracing/models/bk_transformer_baseline.py`: preferred baseline model path.

## Notes
- Input CSV must include columns: `user_id`, `skill_id`, `correct`.
- Pass exactly one processed CSV to `--data_path` for each training run.
- Available processed datasets live under `data/processed/`.
