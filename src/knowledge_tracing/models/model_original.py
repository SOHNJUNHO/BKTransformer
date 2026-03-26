import torch
import torch.nn as nn
import torch.nn.functional as F


class BKTransformer_Origin(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_skills = config.n_skills # skill_counts + 1
        self.skill_emb = nn.Embedding(config.n_skills, config.n_embd)
        self.correct_emb = nn.Embedding(2, config.n_embd) # n_embed = 256
        self.drop = nn.Dropout(config.dropout) # 0.1 

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True  # cleaner normalization order
        )
        self.blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layer  #n_layer = 3
        )

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

        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

    def forward(self, obs, output, lambd=None):
        if lambd is None:
            lambd = [50, 50, 50, 1]
        B, T, D = obs.shape
        assert D == 2, "Expected obs shape (B, T, 2)"
        skill_ids = obs[..., 0].long() #suppose to be sequencial starting from 0 to n_skills+1
        corrects = obs[..., 1].long().clamp(0, 1)
        token_embeddings = self.skill_emb(skill_ids) + self.correct_emb(corrects)
        positions = torch.arange(0, T, dtype=torch.long, device=obs.device)
        position_embeddings = self.pos_emb(positions)  # (T, n_embd) -> broadcasts to (B, T, n_embd)
        
        x = self.drop(token_embeddings + position_embeddings)
        causal_mask = torch.triu(torch.ones(T, T, device=obs.device), diagonal=1).bool()
        x = self.blocks(x, mask=causal_mask, is_causal=False)
        x = self.ln_f(x)

        # logit_diff: (B, T, (n_skills+1)*4)
        logit_diff = self.head(x)

        loss = 0.0

        skill_indices = torch.arange(self.n_skills, device=obs.device)
        # logits: (n_skills+1, 5)
        logits = self.skill_params(self.skill_emb(skill_indices))

        oparams = logits[..., 1:].reshape(1, -1)
        # first dimension used for latent, initial knowledge parameter
        # oparams: (1, (n_skills+1)*4)
        # logit_diff: (B, T, (n_skills+1)*4)

        # params: (B, T, (n_skills+1), 4)
        # personalized BKT parameters per student per time step
        params = torch.sigmoid(oparams + logit_diff).view(B, T, -1, 4)

        # oparams: ((n_skills+1), 4)
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

        corrects = torch.zeros_like(params[..., -1])
        latent = torch.sigmoid(logits[..., 0].repeat((B, 1)))

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
            corrects[:, i] = correct


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

        # params: (B, n_skills+1, 4), 4 parameters for each user_id, each timestep
        # latent: (B, n_skills+1), initial knowledge parameter for each skill

        # correct: (B, n_skills), predicted correctness for each skill, 정답을 맞혔을 확률
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
