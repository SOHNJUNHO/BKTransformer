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
      assistment2009_processed.csv
  notebooks/
    assistment2009_preprocessing.ipynb
  src/
    knowledge_tracing/
      cli.py
      main.py
      train.py
      data/
        datasets.py
        preprocessing.py
        validation.py
      models/
        bk_transformer.py
        bk_transformer_baseline.py
        model.py
        model_original.py
      training/
        lightning_module.py
```

## Setup
Pick the extra that matches your hardware:
```bash
uv sync --extra cu124   # CUDA 12.4
uv sync --extra cu121   # CUDA 12.1
uv sync --extra cpu     # CPU only
```

Optional notebook dependencies:
```bash
uv sync --extra cu124 --extra notebooks
```

## Running Training
From the repo root:
```bash
PYTHONPATH=src python -m knowledge_tracing.cli \
  --data_path data/processed/assistment2009_processed.csv \
  --model_type upgraded \
  --max_epochs 50
```

Checkpoints and logs will be written to `artifacts/checkpoints` and `artifacts/lightning_logs`.

## Key Files
- `src/knowledge_tracing/cli.py`: preferred CLI entrypoint.
- `src/knowledge_tracing/data/datasets.py`: preferred dataset loader and collate module.
- `src/knowledge_tracing/training/lightning_module.py`: Lightning training wrapper.
- `src/knowledge_tracing/models/bk_transformer.py`: preferred upgraded model path.
- `src/knowledge_tracing/models/bk_transformer_baseline.py`: preferred baseline model path.

## Notes
- Input CSV must include columns: `user_id`, `skill_id`, `correct`.
- Use `--order_column` only when you want the training pipeline to enforce per-user ordering at runtime.
- The preferred sample dataset location is `data/processed/assistment2009_processed.csv`.
