import numpy as np


class cement_chemistry:
    """
    Cement chemistry and thermodynamic toolkit
    """

    def __init__(self):

        # ------------------------------
        # Molecular Weights (g/mol)
        # ------------------------------
        self.mw = {
            "CaCO3": 100.09,
            "CaO": 56.08,
            "CO2": 44.01
        }

        # ------------------------------
        # Emission Factors
        # ------------------------------
        self.calcination_factor = 0.525  # t CO2 / t clinker
        self.fuel_emission_factors = {
            "coal": 2.8,          # t CO2 / t fuel
            "natural_gas": 2.0
        }

        # ------------------------------
        # Energy Data
        # ------------------------------
        self.calorific_values = {
            "coal": 25,           # MJ/kg
            "natural_gas": 50
        }

        self.theoretical_heat_GJ_per_ton = 1.7  # Reaction heat approx.


    # ======================================================
    # 1. Bogue Equations
    # ======================================================
    def bogue_equations(self, CaO, SiO2, Al2O3, Fe2O3):
        """
        Estimate clinker phase composition (%)
        """

        C3S = 4.071 * CaO - 7.602 * SiO2 - 1.429 * Fe2O3 - 6.718 * Al2O3
        C2S = 2.867 * SiO2 - 0.7544 * C3S
        C3A = 2.650 * Al2O3 - 1.692 * Fe2O3
        C4AF = 3.043 * Fe2O3

        # Clamp to physical bounds
        phases = {
            "C3S": np.clip(C3S, 0, 100),
            "C2S": np.clip(C2S, 0, 100),
            "C3A": np.clip(C3A, 0, 100),
            "C4AF": np.clip(C4AF, 0, 100)
        }

        return phases


    # ======================================================
    # 2. CO2 Emissions
    # ======================================================
    def calculate_co2_emissions(self, clinker_mass_ton, fuel_mass_ton, fuel_type="coal"):

        process_emissions = clinker_mass_ton * self.calcination_factor

        combustion_emissions = (
            fuel_mass_ton *
            self.fuel_emission_factors.get(fuel_type, 2.8)
        )

        total_emissions = process_emissions + combustion_emissions

        return {
            "process_emissions_tCO2": process_emissions,
            "combustion_emissions_tCO2": combustion_emissions,
            "total_emissions_tCO2": total_emissions
        }


    # ======================================================
    # 3. Mass Balance Check
    # ======================================================
    def mass_balance(self, limestone_input_ton, clinker_output_ton):

        expected_co2_loss = limestone_input_ton * 0.44
        mass_error = abs(
            limestone_input_ton - (clinker_output_ton + expected_co2_loss)
        )

        tolerance = 0.05 * limestone_input_ton

        return {
            "mass_error_ton": mass_error,
            "within_tolerance": mass_error <= tolerance
        }


    # ======================================================
    # 4. Energy Balance / Efficiency
    # ======================================================
    def energy_balance(self, fuel_mass_ton, fuel_type="coal"):

        calorific_value = self.calorific_values.get(fuel_type, 25)

        energy_input_GJ = fuel_mass_ton * 1000 * calorific_value / 1000

        efficiency = (
            self.theoretical_heat_GJ_per_ton /
            energy_input_GJ
        ) * 100 if energy_input_GJ > 0 else 0

        return {
            "energy_input_GJ": energy_input_GJ,
            "thermal_efficiency_percent": efficiency
        }


    # ======================================================
    # 5. Strength Prediction (Bolomey-Féret simplified)
    # ======================================================
    def strength_prediction_bolomey(self, cement_kg, water_kg, age_days=28, scm_fraction=0):

        w_c_ratio = water_kg / cement_kg

        base_strength = 50 * (1 / (w_c_ratio + 0.1))

        age_factor = np.log(age_days + 1) / np.log(29)

        scm_penalty = 1 - 0.2 * scm_fraction

        strength = base_strength * age_factor * scm_penalty

        return strength


    # ======================================================
    # Optional: Lime Saturation Factor (LSF)
    # ======================================================
    def lime_saturation_factor(self, CaO, SiO2, Al2O3, Fe2O3):

        denominator = 2.8 * SiO2 + 1.2 * Al2O3 + 0.65 * Fe2O3

        if denominator == 0:
            return 0

        LSF = CaO / denominator

        return LSF


    # ======================================================
    # Optional: Silica Modulus (SM)
    # ======================================================
    def silica_modulus(self, SiO2, Al2O3, Fe2O3):

        denominator = Al2O3 + Fe2O3

        if denominator == 0:
            return 0

        SM = SiO2 / denominator

        return SM
