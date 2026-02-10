import torch
import torch.nn as nn


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        embed_dim=256,
        n_layers=4,
        n_heads=8,
        seq_len=20
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Embeddings
        self.state_embed = nn.Linear(state_dim, embed_dim)
        self.action_embed = nn.Linear(act_dim, embed_dim)
        self.return_embed = nn.Linear(1, embed_dim)
        self.carbon_embed = nn.Linear(1, embed_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, seq_len, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Output head
        self.predict_action = nn.Linear(embed_dim, act_dim)

    def forward(self, states, actions, returns, carbons):

        # Add feature dimension
        returns = returns.unsqueeze(-1)
        carbons = carbons.unsqueeze(-1)

        s = self.state_embed(states)
        a = self.action_embed(actions)
        r = self.return_embed(returns)
        c = self.carbon_embed(carbons)

        # Combine tokens
        x = s + a + r + c + self.pos_embed[:, :states.shape[1], :]

        x = self.transformer(x)

        action_preds = self.predict_action(x)

        return action_preds
