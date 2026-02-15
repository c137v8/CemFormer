import numpy as np
import pandas as pd
import os


class CementSyntheticDataGenerator:

    def __init__(self, seed=42, output_dir="synthetic_data"):
        np.random.seed(seed)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Materials CSV
    # ---------------------------------------------------------
    def generate_materials(self, n_materials=4):

        data = []
        for i in range(n_materials):
            data.append({
                "name": f"Material_{i}",
                "CaO_pct": np.random.uniform(40, 65),
                "SiO2_pct": np.random.uniform(10, 25),
                "Al2O3_pct": np.random.uniform(2, 8),
                "Fe2O3_pct": np.random.uniform(1, 6),
                "LOI_pct": np.random.uniform(1, 10),
                "price_per_ton": np.random.uniform(20, 80)
            })

        df = pd.DataFrame(data)
        df.to_csv(f"{self.output_dir}/materials.csv", index=False)

    # ---------------------------------------------------------
    # Fuels CSV
    # ---------------------------------------------------------
    def generate_fuels(self, n_fuels=3):

        data = []
        for i in range(n_fuels):
            data.append({
                "name": f"Fuel_{i}",
                "calorific_value_MJ_per_kg": np.random.uniform(15, 30),
                "moisture_pct": np.random.uniform(5, 20),
                "ash_pct": np.random.uniform(5, 25),
                "emission_factor_tCO2_per_ton": np.random.uniform(2.5, 3.5),
                "price_per_ton": np.random.uniform(50, 150)
            })

        df = pd.DataFrame(data)
        df.to_csv(f"{self.output_dir}/fuels.csv", index=False)

    # ---------------------------------------------------------
    # Emission Factors CSV
    # ---------------------------------------------------------
    def generate_emission_factors(self):

        data = {
            "calcination_tCO2_per_ton_clinker": [0.52],
            "electricity_tCO2_per_MWh": [np.random.uniform(0.6, 0.9)],
            "transport_tCO2_per_ton_km": [np.random.uniform(0.08, 0.15)]
        }

        df = pd.DataFrame(data)
        df.to_csv(f"{self.output_dir}/emission_factors.csv", index=False)

    # ---------------------------------------------------------
    # Demand CSV
    # ---------------------------------------------------------
    def generate_demand(self, n_regions=5, horizon=12):

        data = []
        for region in range(n_regions):
            for month in range(horizon):
                data.append({
                    "region": f"Region_{region}",
                    "month": month + 1,
                    "demand_tons": np.random.uniform(5000, 20000)
                })

        df = pd.DataFrame(data)
        df.to_csv(f"{self.output_dir}/demand.csv", index=False)

    # ---------------------------------------------------------
    # Capacity CSV
    # ---------------------------------------------------------
    def generate_capacity(self):

        data = {
            "kiln_capacity_tpd": [np.random.uniform(3000, 6000)],
            "mill_capacity_tpd": [np.random.uniform(2500, 5500)],
            "storage_capacity_tons": [np.random.uniform(50000, 150000)],
            "initial_inventory_tons": [np.random.uniform(10000, 30000)]
        }

        df = pd.DataFrame(data)
        df.to_csv(f"{self.output_dir}/capacity.csv", index=False)

    # ---------------------------------------------------------
    # Logistics CSV
    # ---------------------------------------------------------
    def generate_logistics(self, n_regions=5):

        distance_matrix = np.random.uniform(50, 500, (n_regions, n_regions))
        df_distance = pd.DataFrame(
            distance_matrix,
            columns=[f"Region_{i}" for i in range(n_regions)]
        )
        df_distance.to_csv(f"{self.output_dir}/distance_matrix.csv", index=False)

        freight_rate = pd.DataFrame({
            "freight_rate_per_ton_km": [np.random.uniform(0.05, 0.15)]
        })
        freight_rate.to_csv(f"{self.output_dir}/freight_rate.csv", index=False)

    # ---------------------------------------------------------
    # Energy Prices CSV
    # ---------------------------------------------------------
    def generate_energy_prices(self):

        data = {
            "electricity_price_per_MWh": [np.random.uniform(50, 120)],
            "coal_price_per_ton": [np.random.uniform(80, 150)],
            "petcoke_price_per_ton": [np.random.uniform(70, 130)]
        }

        df = pd.DataFrame(data)
        df.to_csv(f"{self.output_dir}/energy_prices.csv", index=False)

    # ---------------------------------------------------------
    # Generate All
    # ---------------------------------------------------------
    def generate_all(self):

        self.generate_materials()
        self.generate_fuels()
        self.generate_emission_factors()
        self.generate_demand()
        self.generate_capacity()
        self.generate_logistics()
        self.generate_energy_prices()


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":

    generator = CementSyntheticDataGenerator()
    generator.generate_all()

    print("Synthetic CSV dataset generated successfully.")
