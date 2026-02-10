import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CementTrajectoryDataset
from models.decision_transformer import DecisionTransformer
from environment import CementEnv


# =====================================================
# Config
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
SEQ_LEN = 20
EPOCHS = 50
LR = 1e-4


# =====================================================
# Load Dataset
# =====================================================
dataset = CementTrajectoryDataset(
    "trajectory_dataset.pt",
    seq_len=SEQ_LEN
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

state_dim = dataset.samples[0]["states"].shape[1]
act_dim = 2


# =====================================================
# Initialize Model
# =====================================================
model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    seq_len=SEQ_LEN
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()


# =====================================================
# Training Loop
# =====================================================
print("Training started...")

for epoch in range(EPOCHS):

    total_loss = 0

    for states, actions, returns, carbons in loader:

        states = states.to(DEVICE)
        actions = actions.to(DEVICE)
        returns = returns.to(DEVICE)
        carbons = carbons.to(DEVICE)

        optimizer.zero_grad()

        action_preds = model(states, actions, returns, carbons)

        loss = criterion(action_preds, actions)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "cemformer_model.pt")

print("Training completed and model saved.")
