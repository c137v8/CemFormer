import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from surrogate_model import CementSurrogateModel


# Load dataset
df = pd.read_csv("cement_dataset.csv")

X = df[[
    "clinker",
    "flyash",
    "slag",
    "limestone",
    "temp",
    "fuel"
]].values

Y = df[[
    "cost",
    "emissions",
    "risk",
    "strength"
]].values


X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)


dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=128, shuffle=True)


model = CementSurrogateModel()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.MSELoss()


epochs = 200

for epoch in range(epochs):

    total_loss = 0

    for xb, yb in loader:

        pred = model(xb)

        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss {total_loss:.4f}")


torch.save(model.state_dict(), "cement_surrogate.pt")

print("Model saved.")