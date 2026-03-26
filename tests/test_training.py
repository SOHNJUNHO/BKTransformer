import pandas as pd
import pytest
import pytorch_lightning as pl
import torch

from knowledge_tracing.data.preprocessing import get_data_loaders
from knowledge_tracing.main import Config, build_model
from knowledge_tracing.train import NeuralBKTLightning

N_SKILLS = 5


def make_synthetic_df(n_users=6, interactions_per_user=5):
    rows = []
    for user_id in range(n_users):
        for i in range(interactions_per_user):
            rows.append({
                'user_id': user_id,
                'skill_id': (user_id + i) % N_SKILLS,
                'correct': i % 2,
            })
    return pd.DataFrame(rows)


def make_tiny_config(model_type='upgraded'):
    return Config(
        n_skills=N_SKILLS + 1,
        n_embd=16,
        n_layer=1,
        n_head=2,
        block_size=10,
        dropout=0.0,
        model_type=model_type,
    )


@pytest.mark.parametrize("model_type", ["upgraded", "original"])
def test_training_step_runs(model_type):
    df = make_synthetic_df()
    config = make_tiny_config(model_type=model_type)

    train_loader, val_loader, _ = get_data_loaders(
        df,
        block_size=config.block_size,
        batch_size=4,
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
        seed=42,
    )

    model = build_model(config)
    lightning_model = NeuralBKTLightning(model, lr=1e-3)

    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator='cpu',
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(lightning_model, train_loader, val_loader)


@pytest.mark.parametrize("model_type", ["upgraded", "original"])
def test_forward_pass_loss_is_finite(model_type):
    config = make_tiny_config(model_type=model_type)
    model = build_model(config)

    B, T = 2, 5
    obs = torch.zeros(B, T, 2)
    obs[..., 0] = torch.randint(0, N_SKILLS, (B, T)).float()
    obs[..., 1] = torch.randint(0, 2, (B, T)).float()

    output = torch.zeros(B, T, 2)
    output[..., 0] = torch.randint(0, N_SKILLS, (B, T)).float()
    output[..., 1] = torch.randint(0, 2, (B, T)).float()

    corrects, latents, params, loss = model(obs, output)

    assert corrects.shape == (B, T, 1)
    assert loss.isfinite()
    assert loss > 0
