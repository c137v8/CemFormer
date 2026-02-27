import numpy as np
import pandas as pd
import os


class IndustrialCementDataGenerator:

    def __init__(self, seed=42, output_dir="industrial_synthetic_data"):
        np.random.seed(seed)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # RAW MATERIALS (Realistic Types)
    # ---------------------------------------------------------
    def generate_materials(self):

        materials = [
            "Limestone",
            "Clay",
            "Gypsum",
            "Fly_Ash",
            "GGBS"
        ]

        data = []
        for name in materials:
            data.append({
                "name": name,
                "CaO_pct": np.random.uniform(40, 65),
                "SiO2_pct": np.random.uniform(10, 25),
                "Al2O3_pct": np.random.uniform(2, 8),
                "Fe2O3_pct": np.random.uniform(1, 6),
                "MgO_pct": np.random.uniform(0, 5),
                "LOI_pct": np.random.uniform(1, 10),
                "price_per_ton": np.random.uniform(20, 100),
                "waste_gate_fee_per_ton": np.random.uniform(-30, 20)
            })

        pd.DataFrame(data).to_csv(
            f"{self.output_dir}/materials.csv", index=False
        )

    # ---------------------------------------------------------
    # FUELS (Fossil + Alternative)
    # ---------------------------------------------------------
    def generate_fuels(self):

        fuels = ["Coal", "PetCoke", "RDF", "Biomass"]

        data = []
        for name in fuels:
            data.append({
                "name": name,
                "calorific_value_MJ_per_kg": np.random.uniform(15, 32),
                "moisture_pct": np.random.uniform(5, 25),
                "ash_pct": np.random.uniform(5, 30),
                "volatile_pct": np.random.uniform(10, 40),
                "emission_factor_tCO2_per_ton": np.random.uniform(1.8, 3.5),
                "price_per_ton": np.random.uniform(50, 150)
            })

        pd.DataFrame(data).to_csv(
            f"{self.output_dir}/fuels.csv", index=False
        )

    # ---------------------------------------------------------
    # CARBON PRICING
    # ---------------------------------------------------------
    def generate_carbon_policy(self, horizon=12):

        data = []
        for month in range(horizon):
            data.append({
                "month": month + 1,
                "carbon_tax_per_tCO2": np.random.uniform(20, 120)
            })

        pd.DataFrame(data).to_csv(
            f"{self.output_dir}/carbon_policy.csv", index=False
        )

    # ---------------------------------------------------------
    # QUALITY SPECIFICATIONS
    # ---------------------------------------------------------
    def generate_quality_specs(self):

        data = {
            "min_28day_strength_MPa": [42],
            "max_setting_time_minutes": [300],
            "max_free_lime_pct": [2.0],
            "LSF_min": [0.90],
            "SM_max": [3.2]
        }

        pd.DataFrame(data).to_csv(
            f"{self.output_dir}/quality_specs.csv", index=False
        )

    # ---------------------------------------------------------
    # DEMAND FORECAST
    # ---------------------------------------------------------
    def generate_demand(self, n_regions=5, horizon=12):

        data = []
        for region in range(n_regions):
            for month in range(horizon):
                data.append({
                    "region": f"Region_{region}",
                    "month": month + 1,
                    "demand_tons": np.random.uniform(5000, 25000),
                    "cement_grade": np.random.choice(["OPC", "PPC"])
                })

        pd.DataFrame(data).to_csv(
            f"{self.output_dir}/demand.csv", index=False
        )

    # ---------------------------------------------------------
    # MACHINE CAPACITY & STORAGE
    # ---------------------------------------------------------
    def generate_capacity(self):

        data = {
            "kiln_capacity_tpd": [np.random.uniform(3000, 6000)],
            "mill_capacity_tpd": [np.random.uniform(2500, 5500)],
            "crusher_capacity_tph": [np.random.uniform(500, 1500)],
            "preheater_efficiency_pct": [np.random.uniform(85, 95)],
            "storage_raw_meal_tons": [np.random.uniform(50000, 150000)],
            "storage_clinker_tons": [np.random.uniform(30000, 100000)],
            "storage_cement_tons": [np.random.uniform(20000, 80000)],
            "initial_inventory_tons": [np.random.uniform(10000, 40000)]
        }

        pd.DataFrame(data).to_csv(
            f"{self.output_dir}/capacity.csv", index=False
        )

    # ---------------------------------------------------------
    # FLEET AVAILABILITY
    # ---------------------------------------------------------
    def generate_fleet(self):

        data = {
            "num_trucks": [np.random.randint(20, 80)],
            "num_railcars": [np.random.randint(5, 30)],
            "avg_truck_capacity_tons": [np.random.uniform(20, 35)],
            "avg_railcar_capacity_tons": [np.random.uniform(60, 100)]
        }

        pd.DataFrame(data).to_csv(
            f"{self.output_dir}/fleet.csv", index=False
        )

    # ---------------------------------------------------------
    # LOGISTICS DISTANCE MATRIX
    # ---------------------------------------------------------
    def generate_logistics(self, n_regions=5):

        matrix = np.random.uniform(50, 500, (n_regions, n_regions))
        pd.DataFrame(
            matrix,
            columns=[f"Region_{i}" for i in range(n_regions)]
        ).to_csv(
            f"{self.output_dir}/distance_matrix.csv",
            index=False
        )

    # ---------------------------------------------------------
    # KILN TELEMETRY TIME SERIES
    # ---------------------------------------------------------
    def generate_kiln_telemetry(self, hours=168):

        data = []
        for t in range(hours):
            data.append({
                "hour": t,
                "kiln_temperature_C": np.random.normal(1450, 15),
                "pressure_kPa": np.random.normal(101, 5),
                "O2_pct": np.random.uniform(1.5, 3.5),
                "CO_ppm": np.random.uniform(100, 1000),
                "fuel_flow_kgph": np.random.uniform(8000, 12000)
            })

        pd.DataFrame(data).to_csv(
            f"{self.output_dir}/kiln_telemetry.csv",
            index=False
        )

    # ---------------------------------------------------------
    # ENERGY LOAD PROFILE
    # ---------------------------------------------------------
    def generate_energy_profile(self, hours=168):

        data = []
        for t in range(hours):
            data.append({
                "hour": t,
                "electricity_MWh": np.random.uniform(30, 80),
                "thermal_MJ": np.random.uniform(50000, 120000)
            })

        pd.DataFrame(data).to_csv(
            f"{self.output_dir}/energy_profile.csv",
            index=False
        )

    # ---------------------------------------------------------
    # GENERATE ALL
    # ---------------------------------------------------------
    def generate_all(self):

        self.generate_materials()
        self.generate_fuels()
        self.generate_carbon_policy()
        self.generate_quality_specs()
        self.generate_demand()
        self.generate_capacity()
        self.generate_fleet()
        self.generate_logistics()
        self.generate_kiln_telemetry()
        self.generate_energy_profile()


if __name__ == "__main__":

    generator = IndustrialCementDataGenerator()
    generator.generate_all()

    print("Industrial-grade synthetic dataset generated successfully.")