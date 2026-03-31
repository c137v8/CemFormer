import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination

import torch
from surrogate.surrogate_model import CementSurrogateModel


# ======================================================
# Multi-Objective Cement Optimization Problem
# ======================================================
class CementOptimizationProblem(Problem):

    def __init__(self):

        super().__init__(
            n_var=6,
            n_obj=4,
            n_constr=1,
            xl=np.array([70, 0, 0, 0, 1400, 80]),
            xu=np.array([95, 25, 35, 15, 1500, 120])
        )

        # Load surrogate model
        self.model = CementSurrogateModel()
        self.model.load_state_dict(torch.load("cement_surrogate.pt"))
        self.model.eval()

    def _evaluate(self, X, out, *args, **kwargs):

        F = []
        G = []

        for row in X:

            clinker, flyash, slag, limestone, temp, fuel = row

            # -----------------------------
            # Constraint: Blend must be <=100
            # -----------------------------
            blend_sum = clinker + flyash + slag + limestone
            g1 = blend_sum - 100

            # -----------------------------
            # Surrogate prediction
            # -----------------------------
            x_tensor = torch.tensor(row, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                pred = self.model(x_tensor)

            cost, emissions, risk, strength = pred.squeeze().cpu().numpy()

            # Objectives
            F.append([
                cost,
                emissions,
                risk,
                -strength
            ])

            G.append([g1])

        out["F"] = np.array(F)
        out["G"] = np.array(G)


# ======================================================
# Run Optimization
# ======================================================
def run_nsga2():

    problem = CementOptimizationProblem()

    algorithm = NSGA2(
        pop_size=120,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 200)

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True,
        save_history=True
    )

    return result


# ======================================================
# Pareto Plot
# ======================================================
def plot_pareto(result):

    F = result.F

    plt.figure(figsize=(6,5))
    plt.scatter(F[:,0], F[:,1])
    plt.xlabel("Cost ($/ton)")
    plt.ylabel("CO2 Emissions (tCO2)")
    plt.title("Pareto Front: Cost vs Emissions")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("pareto_front.png", dpi=300)
    plt.close()


# ======================================================
# Convergence Plot
# ======================================================
def plot_convergence(result):

    history = result.history

    best_cost = []
    best_emissions = []
    best_risk = []
    best_strength = []

    for algo in history:
        F = algo.pop.get("F")
        best_cost.append(np.min(F[:,0]))
        best_emissions.append(np.min(F[:,1]))
        best_risk.append(np.min(F[:,2]))
        best_strength.append(-np.min(F[:,3]))

    generations = range(1, len(best_cost)+1)

    # COST
    plt.figure(figsize=(6,5))
    plt.plot(generations, best_cost)
    plt.title("Cost Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best Cost")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cost_convergence.png", dpi=300)
    plt.close()

    # EMISSIONS
    plt.figure(figsize=(6,5))
    plt.plot(generations, best_emissions)
    plt.title("Emissions Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best Emissions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("emissions_convergence.png", dpi=300)
    plt.close()

    # STRENGTH
    plt.figure(figsize=(6,5))
    plt.plot(generations, best_strength)
    plt.title("Strength Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best Strength")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("strength_convergence.png", dpi=300)
    plt.close()

    # RISK
    plt.figure(figsize=(6,5))
    plt.plot(generations, best_risk)
    plt.title("Risk Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Best Risk")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("risk_convergence.png", dpi=300)
    plt.close()

# ======================================================
# Main
# ======================================================
if __name__ == "__main__":

    result = run_nsga2()

    print("Optimization Completed")
    print("Pareto Solutions Found:", len(result.F))

    print("\nPareto Decision Variables")
    print(result.X)

    print("\nPareto Objectives")
    print(result.F)

    plot_pareto(result)
    plot_convergence(result)