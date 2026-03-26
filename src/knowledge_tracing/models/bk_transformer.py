import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings


class SwiGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.gate_up_proj = nn.Linear(in_dim, 2 * hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.dropout(x)
        return self.down_proj(x)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=config.block_size)

    def forward(self, x):
        batch_size, seq_len, channels = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        y = self.resid_dropout(self.out_proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        hidden_dim = int(8 / 3 * config.n_embd)
        self.mlp = SwiGLU(config.n_embd, hidden_dim, dropout=config.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BKTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_skills = config.n_skills
        self.skill_emb = nn.Embedding(config.n_skills, config.n_embd)
        self.correct_emb = nn.Embedding(2, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.PReLU(),
            nn.Linear(config.n_embd, config.n_embd),
            nn.PReLU(),
            nn.Linear(config.n_embd, self.n_skills * 4),
        )
        layers = []
        for _ in range(4):
            layers.extend([nn.Linear(config.n_embd, config.n_embd), nn.ReLU()])
        layers.extend([nn.Linear(config.n_embd, 5)])
        self.skill_params = nn.Sequential(*layers)

    def forward(self, obs, output, lambd=None):
        if lambd is None:
            lambd = [50, 50, 50, 1]
        batch_size, seq_len, dims = obs.shape
        assert dims == 2, "Expected obs shape (B, T, 2)"
        skill_ids = obs[..., 0].long()
        corrects = obs[..., 1].long().clamp(0, 1)
        token_embeddings = self.skill_emb(skill_ids) + self.correct_emb(corrects)
        x = self.drop(token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        logit_diff = self.head(x)
        loss = 0.0

        skill_indices = torch.arange(self.n_skills, device=obs.device)
        logits = self.skill_params(self.skill_emb(skill_indices))

        oparams = logits[..., 1:].reshape(1, -1)
        params = torch.sigmoid(oparams + logit_diff).view(batch_size, seq_len, -1, 4)
        oparams = torch.sigmoid(oparams.view(-1, 4))

        loss = loss + (
            lambd[0]
            * (
                F.relu(params[..., 0] - (1 - params[..., 3]) / (params[..., 2] + 1e-6)).mean()
                + F.relu(
                    oparams[..., 0] - (1 - oparams[..., 3]) / (oparams[..., 2] + 1e-6)
                ).mean()
            )
            + lambd[1]
            * (
                F.relu(params[..., 2] - 0.5).mean() + F.relu(oparams[..., 2] - 0.5).mean()
            )
            + lambd[2]
            * (
                F.relu(params[..., 3] - 0.5).mean() + F.relu(oparams[..., 3] - 0.5).mean()
            )
            + lambd[3] * torch.mean(logit_diff**2)
            + lambd[3]
            * (
                torch.mean((logit_diff[:, 1:] - logit_diff[:, :-1]) ** 2)
                if logit_diff.shape[1] > 1
                else 0
            )
        )

        corrects = torch.zeros_like(params[..., -1])
        latent = torch.sigmoid(logits[..., 0].repeat((batch_size, 1)))

        latents = []
        for i in range(seq_len):
            latent = torch.clamp(latent, min=1e-5, max=1 - 1e-5)
            latents.append(latent)
            correct, latent = self.extract_latent_correct(
                params[:, i].view(batch_size, -1, 4),
                latent,
                true_correct=output[:, i, -1],
                skills=torch.where(output[:, i, 0] == -1000, 0, output[:, i, 0]).long(),
            )
            corrects[:, i] = correct.squeeze()

        skill_idx = torch.where(output[..., 0] == -1000, 0, output[..., 0]).long()
        corrects = torch.gather(corrects, dim=-1, index=skill_idx.unsqueeze(-1))
        mask = output[..., 1] != -1000
        loss = loss + F.binary_cross_entropy(corrects[mask], output[..., 1:][mask])

        return corrects, latents, params, loss

    def extract_latent_correct(self, params, latent, true_correct, skills):
        learning, guess, slip = params[..., 0], params[..., 2], params[..., 3]
        correct = latent * (1 - slip) + (1 - latent) * guess
        k_t1 = (latent * (1 - slip)) / (latent * (1 - slip) + (1 - latent) * guess)
        k_t0 = (latent * slip) / (latent * slip + (1 - latent) * (1 - guess))
        k_t = torch.clone(latent)

        k_t[range(len(k_t)), skills] = torch.where(
            true_correct > 0.5, k_t1[range(len(k_t)), skills], k_t0[range(len(k_t)), skills]
        )
        k_t[range(len(k_t)), skills] = (
            k_t[range(len(k_t)), skills]
            + (1 - k_t[range(len(k_t)), skills]) * learning[range(len(k_t)), skills]
        )
        return correct, torch.clamp(k_t, 1e-4, 1 - 1e-4)


__all__ = ["BKTransformer"]
