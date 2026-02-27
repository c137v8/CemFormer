import torch
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from sklearn.preprocessing import StandardScaler
import pandas as pd

# -----------------------------------------------------------
# Load Trained MT-PINN
# -----------------------------------------------------------

from train_mtpinn import MTPINN  # reuse model class

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MTPINN(input_dim=16)
model.load_state_dict(torch.load("mtpinn_model.pt", map_location=device))
model.to(device)
model.eval()

# Load scaler from dataset (re-fit same way as training)
df = pd.read_csv("cement_lhs_dataset.csv")
input_cols = df.columns[:16]
X_raw = df[input_cols].values
scaler = StandardScaler()
scaler.fit(X_raw)

# -----------------------------------------------------------
# Define NSGA-II Problem
# -----------------------------------------------------------

class CementOptimizationProblem(Problem):

    def __init__(self):

        # Decision variables only (8)
        xl = np.array([70, 0, 0, 0, 1400, 0.5, 80, 20])
        xu = np.array([95, 25, 35, 15, 1500, 4.0, 120, 40])

        super().__init__(
            n_var=8,
            n_obj=4,
            n_constr=0,
            xl=xl,
            xu=xu
        )

    def _evaluate(self, X, out, *args, **kwargs):

        n = X.shape[0]

        # Expand to 16 inputs by adding fixed scenario variables
        scenario = np.tile([
            63, 21, 5, 3,      # CaO, SiO2, Al2O3, Fe2O3
            90, 50, 0.2, 1500  # fuel_price, carbon_tax, transport_cost, demand
        ], (n, 1))

        full_input = np.hstack([X, scenario])

        # Normalize using training scaler
        full_input = scaler.transform(full_input)

        x_tensor = torch.tensor(full_input, dtype=torch.float32).to(device)

        with torch.no_grad():
            strength, emissions, cost, risk = model(x_tensor)

        strength = strength.cpu().numpy().flatten()
        emissions = emissions.cpu().numpy().flatten()
        cost = cost.cpu().numpy().flatten()
        risk = risk.cpu().numpy().flatten()

        # Define objective vector
        F = np.column_stack([
            -strength,     # maximize strength
            emissions,     # minimize emissions
            cost,          # minimize cost
            risk           # minimize risk
        ])

        out["F"] = F


# -----------------------------------------------------------
# Run NSGA-II
# -----------------------------------------------------------

problem = CementOptimizationProblem()

algorithm = NSGA2(pop_size=100)

termination = get_termination("n_gen", 100)

result = minimize(
    problem,
    algorithm,
    termination,
    seed=42,
    verbose=True
)

# -----------------------------------------------------------
# Save Pareto Front
# -----------------------------------------------------------

pareto_solutions = result.X
pareto_objectives = result.F

np.save("pareto_solutions.npy", pareto_solutions)
np.save("pareto_objectives.npy", pareto_objectives)

print("Optimization complete. Pareto front saved.")
