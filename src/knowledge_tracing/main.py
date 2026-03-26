import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

try:
    from .data.datasets import get_data_loaders
    from .data.validation import validate_dataframe
    from .train import NeuralBKTLightning
except ImportError:
    from knowledge_tracing.data.datasets import get_data_loaders
    from knowledge_tracing.data.validation import validate_dataframe
    from knowledge_tracing.train import NeuralBKTLightning

@dataclass
class Config:
    n_skills: int
    n_embd: int = 256
    n_layer: int = 3
    n_head: int = 4
    block_size: int = 512
    dropout: float = 0.1
    model_type: str = 'upgraded'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_type',
        type=str,
        default='upgraded',
        choices=['upgraded', 'original'],
        help='Model type.'
    )
    parser.add_argument(
        '--data_path',
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / 'data'
            / 'processed'
            / 'assistment2009_processed.csv'
        ),
        help='Path to preprocessed CSV containing user_id, skill_id, correct.'
    )
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=818)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--order_column',
        type=str,
        default=None,
        help='Optional interaction ordering column to enforce per-user sorting at runtime.'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=Path,
        default=Path(__file__).resolve().parents[2] / 'artifacts' / 'checkpoints'
    )
    parser.add_argument(
        '--log_dir',
        type=Path,
        default=Path(__file__).resolve().parents[2] / 'artifacts' / 'lightning_logs'
    )
    return parser.parse_args()


def build_model(config: Config):
    if config.model_type == 'upgraded':
        try:
            from .models.bk_transformer import BKTransformer
        except ImportError:
            from knowledge_tracing.models.bk_transformer import BKTransformer
        return BKTransformer(config)

    try:
        from .models.bk_transformer_baseline import BKTransformerBaseline
    except ImportError:
        from knowledge_tracing.models.bk_transformer_baseline import BKTransformerBaseline
    return BKTransformerBaseline(config)


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = [str(column).strip().lstrip('\ufeff') for column in df.columns]
    return df


def main(args: argparse.Namespace):
    if not args.data_path.exists():
        raise FileNotFoundError(f'Data file not found: {args.data_path}')

    df = load_dataframe(args.data_path)
    validate_dataframe(df)

    config = Config(
        n_skills=int(df['skill_id'].nunique()) + 1,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        block_size=args.block_size,
        dropout=args.dropout,
        model_type=args.model_type,
    )

    train_loader, val_loader, test_loader = get_data_loaders(
        df,
        block_size=config.block_size,
        batch_size=args.batch_size,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        seed=args.seed,
        order_column=args.order_column,
    )

    model = build_model(config)
    lightning_model = NeuralBKTLightning(model, lr=args.lr)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(args.checkpoint_dir),
        filename=f'{config.model_type}-best-{{epoch:02d}}-{{val_auc:.4f}}',
        monitor='val_auc',
        mode='max',
        save_top_k=1,
        verbose=True
    )
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        verbose=True
    )

    logger = CSVLogger(save_dir=str(args.log_dir), name=config.model_type)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(lightning_model, train_loader, val_loader)

    best_checkpoint = checkpoint_callback.best_model_path
    if not best_checkpoint:
        raise RuntimeError(
            'No best checkpoint was created. '
            'Check training/validation loop and monitored metric.'
        )

    test_results = trainer.test(
        lightning_model,
        test_loader,
        ckpt_path=best_checkpoint
    )


    print("\n" + "="*50)
    print("FINAL RESULTS (for paper):")
    print("="*50)
    print(f"Test AUC:  {test_results[0]['test_auc']:.4f}")
    print(f"Test Acc:  {test_results[0]['test_acc']:.4f}")
    print(f"Test Loss: {test_results[0]['test_loss']:.4f}")
    print("="*50)
    
    return test_results


if __name__ == '__main__':
    cli_args = parse_args()
    pl.seed_everything(cli_args.seed, workers=True)
    main(cli_args)
