import numpy as np
import pandas as pd
from chemistry.cement_chemistry import cement_chemistry

chem = cement_chemistry()

rows = []

N = 20000   # dataset size

for _ in range(N):

    clinker = np.random.uniform(70,95)
    flyash = np.random.uniform(0,25)
    slag = np.random.uniform(0,35)
    limestone = np.random.uniform(0,15)

    temp = np.random.uniform(1400,1500)
    fuel = np.random.uniform(80,120)

    scm_fraction = (flyash + slag) / 100

    strength = chem.strength_prediction_bolomey(
        cement_kg=400,
        water_kg=180,
        age_days=28,
        scm_fraction=scm_fraction
    )

    emissions = chem.calculate_co2_emissions(
        clinker_mass_ton=clinker/100,
        fuel_mass_ton=fuel/1000
    )["total_emissions_tCO2"]

    fuel_price = 100
    carbon_tax = 50
    transport_cost = 0.2
    demand = 1500

    material_cost = (
        clinker * 0.8 +
        flyash * 0.3 +
        slag * 0.5 +
        limestone * 0.2
    )

    fuel_cost = (fuel/1000)*fuel_price
    carbon_cost = emissions*carbon_tax
    logistics_cost = transport_cost*(demand/1000)

    cost = material_cost + fuel_cost + carbon_cost + logistics_cost

    temp_dev = abs(temp-1450)/100

    risk = np.clip(
        0.5*temp_dev +
        0.3*scm_fraction +
        0.2*max(0,(80-clinker)/100),
        0,1
    )

    rows.append([
        clinker,flyash,slag,limestone,temp,fuel,
        cost,emissions,risk,strength
    ])

df = pd.DataFrame(rows,columns=[
    "clinker",
    "flyash",
    "slag",
    "limestone",
    "temp",
    "fuel",
    "cost",
    "emissions",
    "risk",
    "strength"
])

df.to_csv("cement_dataset.csv",index=False)

print("Dataset generated.")