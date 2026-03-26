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
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        #q, k = self.rope(q, k)
        q = self.rope(q)
        k = self.rope(k)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        hidden_dim = int(8/3 * config.n_embd)
        self.mlp = SwiGLU(config.n_embd, hidden_dim, dropout=config.dropout)

    def forward(self, x):
        # Pre-Normalization
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
        self.head = nn.Sequential(nn.Linear(config.n_embd, config.n_embd), 
                                  nn.PReLU(), 
                                  nn.Linear(config.n_embd, config.n_embd),
                                  nn.PReLU(), 
                                  nn.Linear(config.n_embd, self.n_skills*4))
        layers = []
        for _ in range(4):
            layers.extend([nn.Linear(config.n_embd, config.n_embd), nn.ReLU()])
        layers.extend([nn.Linear(config.n_embd, 5)])
        self.skill_params = nn.Sequential(*layers)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, obs, output, lambd=None):
        if lambd is None:
            lambd = [50, 50, 50, 1]
        B, T, D = obs.shape
        assert D == 2, "Expected obs shape (B, T, 2)"
        skill_ids = obs[..., 0].long() #suppose to be sequencial starting from 0 to n_skills-1
        corrects = obs[..., 1].long().clamp(0, 1)
        token_embeddings = self.skill_emb(skill_ids) + self.correct_emb(corrects)
        # In the original paper, token embeddings pass through deeper projection stacks.
        x = self.drop(token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        # logit_diff: (B, T, vocab_size=n_skills*4)
        logit_diff = self.head(x)

        loss = 0.0

        #if self.config.bkt:
        skill_indices = torch.arange(self.n_skills, device=obs.device)
        # logits: (n_skills, 5)
        logits = self.skill_params(self.skill_emb(skill_indices))

        oparams = logits[..., 1:].reshape(1, -1)
        # oparams: (1, n_skills*4)
        # logit_diff: (B, T, n_skills*4)

        # params: (B, T, n_skills, 4)
        # personalized BKT parameters per student per time step
        params = torch.sigmoid(oparams + logit_diff).view(B, T, -1, 4)

        # oparams: (n_skills, 4)
        # global,static baseline BKT parameters for each skill
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
            + lambd[3] * torch.mean(logit_diff ** 2)
            + lambd[3]
            * (
                torch.mean((logit_diff[:, 1:] - logit_diff[:, :-1]) ** 2)
                if logit_diff.shape[1] > 1
                else 0
            )
        )

        # corrects: (B, T, n_skills), all zeros
        # latent: (B, n_skills), initial knowledge parameter for each skill * batch size
        corrects = torch.zeros_like(params[..., -1])
        latent = torch.sigmoid(logits[..., 0].repeat((B, 1)))
        # corrects example - [[0,0,0],[0,0,0]] , suppose there are three skills
        # latent example - [[0.8, 0.7],[0.8, 0.7]], suppose batch size is 2

        #if prior is not None:
        #    latent = prior.detach()  # Use the final state from previous batch

        latents = []
        for i in range(T):
            latent = torch.clamp(latent, min = 1e-5, max = 1 - 1e-5)
            latents.append(latent)
            correct, latent = self.extract_latent_correct(
                params[:, i].view(B, -1, 4),
                latent,
                true_correct=output[:, i, -1],
                skills=torch.where(output[:, i, 0] == -1000, 0, output[:, i, 0]).long(),
            )
            corrects[:, i] = correct.squeeze()

        #if output is not None:
        # skill_idx: (B, T), skill indices at each timestep, padding -1000
        skill_idx = torch.where(output[..., 0] == -1000, 0, output[..., 0]).long()
        # only need the prediction corresponding to the actual skill attempted at the timestep
        corrects = torch.gather(corrects, dim=-1, index=skill_idx.unsqueeze(-1))
        mask = output[..., 1] != -1000
        loss = loss + F.binary_cross_entropy(corrects[mask], output[..., 1:][mask])

        return corrects, latents, params, loss


    def extract_latent_correct(self, params, latent, true_correct, skills):
        learning, guess, slip = params[..., 0], params[..., 2], params[..., 3]
        # learning, guess, slip: (B, n_skills)

        # params: (B, n_skills, 4), 4 parameters for each user_id, each timestep
        # latent: (B, n_skills), initial knowledge parameter for each skill

        # correct: (B, n_skills), predicted correctness for each skill, 정답을 맞혔을 확률
        correct = latent * (1 - slip) + (1 - latent) * guess
        # k_t1: (B, n_skills), posterior knowledge if answered correctly, 알고 정답을 맞혔을 확률 
        # k_t0: posterior knowledge if answered incorrectly.
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

        # valid_mask = true_correct != -1000
        # if valid_mask.any():
        #     valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        #     valid_skills = skills[valid_idx]
            
        #     k_t[valid_idx, valid_skills] = torch.where(
        #         true_correct[valid_idx] > 0.5, 
        #         k_t1[valid_idx, valid_skills], 
        #         k_t0[valid_idx, valid_skills]
        #     )
        #     k_t[valid_idx, valid_skills] = k_t[valid_idx, valid_skills] + \
        #         (1 - k_t[valid_idx, valid_skills]) * l[valid_idx, valid_skills]
        
        # return correct, torch.clamp(k_t, 1e-4, 1 - 1e-4)
