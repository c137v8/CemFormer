import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# ====================== UPDATED MODEL WITH BOGUE ======================
class CementSurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 11)        # 4 perf + 3 moduli + 4 Bogue phases
        )
    
    def forward(self, x):
        out = self.network(x)
        
        perf = out[:, :4]                    # cost, emissions, risk, strength
        
        # Raw predictions
        raw_LSF = out[:, 4]
        raw_SM  = out[:, 5]
        raw_AM  = out[:, 6]
        raw_C3S = out[:, 7]
        raw_C2S = out[:, 8]
        raw_C3A = out[:, 9]
        raw_C4AF = out[:, 10]
        
        # Soft projection for moduli (helps training stability)
        LSF = torch.sigmoid(raw_LSF) * 0.20 + 0.80   # ~0.80 - 1.00
        SM  = torch.sigmoid(raw_SM)  * 2.5 + 1.5     # ~1.5 - 4.0
        AM  = torch.sigmoid(raw_AM)  * 3.5 + 0.5     # ~0.5 - 4.0
        
        # Soft projection for phases (allow small negatives but penalize heavily later)
        C3S  = torch.relu(raw_C3S) * 1.2     # encourage positive
        C2S  = torch.relu(raw_C2S)
        C3A  = torch.relu(raw_C3A)
        C4AF = torch.relu(raw_C4AF)
        
        moduli = torch.stack([LSF, SM, AM], dim=1)
        phases = torch.stack([C3S, C2S, C3A, C4AF], dim=1)
        
        return perf, moduli, phases


# ====================== PHYSICS LOSSES ======================
def moduli_loss(moduli, lambda_mod=10.0):
    LSF, SM, AM = moduli[:, 0], moduli[:, 1], moduli[:, 2]
    
    lsf_pen = torch.relu(LSF - 0.98) + torch.relu(0.88 - LSF)
    sm_pen  = torch.relu(SM - 3.0) + torch.relu(2.0 - SM)
    am_pen  = torch.relu(AM - 3.0) + torch.relu(1.0 - AM)
    
    return lambda_mod * (lsf_pen + sm_pen + am_pen).mean()


def bogue_loss(phases, lambda_bogue=5.0):
    C3S, C2S, C3A, C4AF = phases[:, 0], phases[:, 1], phases[:, 2], phases[:, 3]
    
    penalties = []
    
    # 1. Non-negativity (already using ReLU, but extra safety)
    penalties.append(torch.relu(-C3S).mean())
    penalties.append(torch.relu(-C2S).mean())
    penalties.append(torch.relu(-C3A).mean())
    penalties.append(torch.relu(-C4AF).mean())
    
    # 2. Reasonable sum of main phases (typically 95-105%)
    total_phases = C3S + C2S + C3A + C4AF
    sum_penalty = torch.relu(total_phases - 105.0) + torch.relu(90.0 - total_phases)
    penalties.append(sum_penalty.mean())
    
    # 3. Realistic ranges (common in clinker)
    penalties.append(torch.relu(C3S - 75.0).mean())   # Alite usually <75%
    penalties.append(torch.relu(C3A - 12.0).mean())   # Aluminate usually <12%
    
    total_bogue_loss = torch.stack(penalties).sum()
    return lambda_bogue * total_bogue_loss


# ====================== TRAINING LOOP ======================
df = pd.read_csv("cement_dataset.csv")

X = df[["clinker", "flyash", "slag", "limestone", "temp", "fuel"]].values
Y = df[["cost", "emissions", "risk", "strength"]].values

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = CementSurrogateModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
data_loss_fn = nn.MSELoss()

epochs = 300
lambda_mod = 10.0
lambda_bogue = 6.0

for epoch in range(epochs):
    total_loss = 0.0
    total_data = 0.0
    total_mod = 0.0
    total_bogue = 0.0
    
    for xb, yb in loader:
        pred_perf, pred_moduli, pred_phases = model(xb)
        
        loss_data = data_loss_fn(pred_perf, yb)
        loss_mod  = moduli_loss(pred_moduli, lambda_mod)
        loss_bogue = bogue_loss(pred_phases, lambda_bogue)
        
        loss = loss_data + loss_mod + loss_bogue
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_data += loss_data.item()
        total_mod += loss_mod.item()
        total_bogue += loss_bogue.item()
    
    if (epoch + 1) % 30 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Total: {total_loss/len(loader):.4f} | "
              f"Data: {total_data/len(loader):.4f} | "
              f"Moduli: {total_mod/len(loader):.4f} | "
              f"Bogue: {total_bogue/len(loader):.4f}")

torch.save(model.state_dict(), "cement_surrogate_physics_bogue.pt")
print("Physics + Bogue informed model saved.")