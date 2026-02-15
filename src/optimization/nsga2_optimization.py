import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
import matplotlib.pyplot as plt

from src.chemistry.cement_chemistry import CementChemistry


# ======================================================
# Multi-Objective Cement Optimization Problem
# ======================================================
class CementOptimizationProblem(Problem):

    def __init__(self):

        self.chem = CementChemistry()

        # Decision variables:
        # clinker_pct, fly_ash_pct, slag_pct, limestone_pct,
        # kiln_temp, fuel_input
        super().__init__(
            n_var=6,
            n_obj=4,
            n_constr=3,
            xl=np.array([70, 0, 0, 0, 1400, 80]),
            xu=np.array([95, 25, 35, 15, 1500, 120])
        )

    def _evaluate(self, X, out, *args, **kwargs):

        F = []
        G = []

        for row in X:

            clinker, flyash, slag, limestone, temp, fuel = row

            # -----------------------------
            # Constraint 1: Blend sum <= 100
            # -----------------------------
            blend_sum = clinker + flyash + slag + limestone
            g1 = blend_sum - 100

            # -----------------------------
            # Chemistry assumptions
            # -----------------------------
            CaO = 65
            SiO2 = 22
            Al2O3 = 5
            Fe2O3 = 3

            # LSF constraint
            lsf = self.chem.lime_saturation_factor(
                CaO, SiO2, Al2O3, Fe2O3
            )
            g2 = 0.90 - lsf  # require LSF ≥ 0.90

            # Silica modulus constraint
            sm = self.chem.silica_modulus(
                SiO2, Al2O3, Fe2O3
            )
            g3 = sm - 3.2  # require SM ≤ 3.2

            # -----------------------------
            # Strength
            # -----------------------------
            scm_fraction = (flyash + slag) / 100
            strength = self.chem.strength_prediction_bolomey(
                cement_kg=400,
                water_kg=180,
                age_days=28,
                scm_fraction=scm_fraction
            )

            # -----------------------------
            # Emissions
            # -----------------------------
            emissions_dict = self.chem.calculate_co2_emissions(
                clinker_mass_ton=1.0,
                fuel_mass_ton=fuel / 1000
            )
            emissions = emissions_dict["total_emissions_tCO2"]

            # -----------------------------
            # Cost Model
            # -----------------------------
            fuel_price = 100
            carbon_tax = 50
            transport_cost = 0.2
            demand = 1500

            material_cost = clinker * 0.8 + flyash * 0.3
            fuel_cost = (fuel / 1000) * fuel_price
            carbon_cost = emissions * carbon_tax
            logistics_cost = transport_cost * (demand / 1000)

            total_cost = material_cost + fuel_cost + carbon_cost + logistics_cost

            # -----------------------------
            # Risk Score
            # -----------------------------
            temp_dev = abs(temp - 1450) / 100
            risk = np.clip(
                0.5 * temp_dev +
                0.3 * scm_fraction +
                0.2 * max(0, (80 - clinker) / 100),
                0, 1
            )

            # Objectives:
            # Minimize cost
            # Minimize emissions
            # Minimize risk
            # Maximize strength → minimize (-strength)
            F.append([
                total_cost,
                emissions,
                risk,
                -strength
            ])

            G.append([g1, g2, g3])

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
        verbose=True
    )

    return result


# ======================================================
# Visualization
# ======================================================
def plot_pareto(result):

    F = result.F

    plt.figure(figsize=(8,6))
    plt.scatter(F[:, 0], F[:, 1])
    plt.xlabel("Cost ($/ton)")
    plt.ylabel("Emissions (tCO2)")
    plt.title("Pareto Front: Cost vs Emissions")
    plt.grid(True)
    plt.show()


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":

    result = run_nsga2()

    print("Optimization Completed")
    print("Pareto Solutions Found:", len(result.F))

    plot_pareto(result)
