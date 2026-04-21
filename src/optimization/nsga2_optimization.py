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

from surrogate.surrogate_model import CementSurrogateModel   # Make sure this imports the NEW model

# ======================================================
# Multi-Objective Cement Optimization Problem (Physics-Informed)
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
        
        # Load the NEW physics-informed model
        self.model = CementSurrogateModel()
        checkpoint_path = "cement_surrogate_physics_bogue.pt"   # ← Change if your filename is different
        
        try:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
            print(f"Successfully loaded physics-informed model from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have trained and saved the new model first!")
            raise
        
        self.model.eval()

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        G = []
        
        for row in X:
            clinker, flyash, slag, limestone, temp, fuel = row
            
            # -----------------------------
            # Constraint: Total blend <= 100%
            # -----------------------------
            blend_sum = clinker + flyash + slag + limestone
            g1 = blend_sum - 100.0
            
            # -----------------------------
            # Surrogate prediction with Physics + Bogue
            # -----------------------------
            x_tensor = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                pred_perf, pred_moduli, pred_phases = self.model(x_tensor)
            
            # Extract performance objectives
            cost, emissions, risk, strength = pred_perf.squeeze().cpu().numpy()
            
            # (Optional) You can also access moduli and phases for analysis
            LSF, SM, AM = pred_moduli.squeeze().cpu().numpy()
            C3S, C2S, C3A, C4AF = pred_phases.squeeze().cpu().numpy()
            
            # Objectives: Minimize cost, emissions, risk | Maximize strength
            F.append([
                cost,
                emissions,
                risk,
                -strength          # negative because we minimize in pymoo
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
# Plotting Functions (unchanged)
# ======================================================
def plot_pareto(result):
    F = result.F
    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 0], F[:, 1], c=F[:, 3], cmap='viridis')  # color by strength
    plt.colorbar(label='Strength (higher is better)')
    plt.xlabel("Cost ($/ton)")
    plt.ylabel("CO₂ Emissions (tCO₂/ton)")
    plt.title("Pareto Front: Cost vs Emissions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pareto_front.png", dpi=300)
    plt.close()

def plot_3d_pareto(result):
    F = result.F  # final population
    
    cost = F[:, 0]
    emissions = F[:, 1]
    strength = -F[:, 3]
    
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(cost, emissions, strength)
    
    ax.set_title("3D Pareto Front")
    ax.set_xlabel("Cost")
    ax.set_ylabel("Emissions")
    ax.set_zlabel("Strength")
    
    plt.tight_layout()
    plt.savefig("3d_pareto.png", dpi=300)
    plt.show()

def plot_convergence(result):
    history = result.history
    best_cost = []
    best_emissions = []
    best_risk = []
    best_strength = []
    
    for algo in history:
        F = algo.pop.get("F")
        best_cost.append(np.min(F[:, 0]))
        best_emissions.append(np.min(F[:, 1]))
        best_risk.append(np.min(F[:, 2]))
        best_strength.append(-np.min(F[:, 3]))   # convert back to positive strength
    
    generations = range(1, len(best_cost) + 1)
    
    # Plot Cost
    plt.figure(figsize=(6, 5))
    plt.plot(generations, best_cost)
    plt.title("Best Cost Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Cost ($/ton)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cost_convergence.png", dpi=300)
    plt.close()
    
    # Plot Emissions
    plt.figure(figsize=(6, 5))
    plt.plot(generations, best_emissions)
    plt.title("Best Emissions Convergence")
    plt.xlabel("Generation")
    plt.ylabel("CO₂ Emissions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("emissions_convergence.png", dpi=300)
    plt.close()
    
    # Plot Strength
    plt.figure(figsize=(6, 5))
    plt.plot(generations, best_strength)
    plt.title("Best Strength Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Strength")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("strength_convergence.png", dpi=300)
    plt.close()
    
    # Plot Risk
    plt.figure(figsize=(6, 5))
    plt.plot(generations, best_risk)
    plt.title("Best Risk Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Risk")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("risk_convergence.png", dpi=300)
    plt.close()


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    result = run_nsga2()
    
    print("Optimization Completed!")
    print(f"Pareto Solutions Found: {len(result.F)}")
    
    print("\nPareto Decision Variables (first 5):")
    print(result.X[:5])
    
    print("\nPareto Objectives (first 5):")
    print(result.F[:5])
    
    plot_pareto(result)
    plot_convergence(result)
    plot_3d_pareto(result)
    
    # Optional: Save results
    np.save("pareto_solutions_X.npy", result.X)
    np.save("pareto_solutions_F.npy", result.F)
    print("Results saved as .npy files.")