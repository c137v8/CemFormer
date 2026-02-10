import numpy as np


class CementEnv:
    """
    Minimal Cement + Logistics Environment
    ---------------------------------------
    State includes:
        - inventory
        - fuel_used
        - electricity_used
        - vehicle positions
        - vehicle availability
        - order demands
        - order deadlines

    Action:
        (vehicle_id, order_id)
    """

    def __init__(
        self,
        num_vehicles=20,
        num_orders=50,
        max_inventory=1000,
        grid_size=100,
        max_steps=200
    ):
        self.num_vehicles = num_vehicles
        self.num_orders = num_orders
        self.max_inventory = max_inventory
        self.grid_size = grid_size
        self.max_steps = max_steps

        # Carbon factors
        self.FUEL_FACTOR = 2.4      # kg CO2 per unit fuel
        self.ELEC_FACTOR = 0.82     # kg CO2 per kWh
        self.TRANS_FACTOR = 0.12    # kg CO2 per km

        # Cost factors
        self.FUEL_COST = 80
        self.ELEC_COST = 8

        self.reset()

    # ==========================================================
    # Reset Environment
    # ==========================================================
    def reset(self):

        self.step_count = 0

        self.inventory = float(self.max_inventory)
        self.fuel_used = 0.0
        self.electricity_used = 0.0

        # Vehicles
        self.vehicle_positions = np.random.rand(
            self.num_vehicles, 2
        ) * self.grid_size

        self.vehicle_available = np.ones(self.num_vehicles)

        # Orders
        self.order_positions = np.random.rand(
            self.num_orders, 2
        ) * self.grid_size

        self.order_demands = np.random.randint(
            5, 20, self.num_orders
        ).astype(float)

        self.order_deadlines = np.random.randint(
            5, 15, self.num_orders
        ).astype(float)

        return self._get_state()

    # ==========================================================
    # Distance Function
    # ==========================================================
    def _distance(self, v_pos, o_pos):
        return np.linalg.norm(v_pos - o_pos)

    # ==========================================================
    # Step Function
    # ==========================================================
    def step(self, action):

        self.step_count += 1

        vehicle_id, order_id = action

        info = {
            "cost": 0.0,
            "carbon": 0.0,
            "delivered": 0.0
        }

        # Invalid actions penalty
        if (
            vehicle_id >= self.num_vehicles
            or order_id >= self.num_orders
            or self.vehicle_available[vehicle_id] == 0
            or self.order_demands[order_id] <= 0
            or self.inventory <= 0
        ):
            return self._get_state(), -1000, False, info

        # Compute distance
        dist = self._distance(
            self.vehicle_positions[vehicle_id],
            self.order_positions[order_id]
        )

        # Update vehicle position
        self.vehicle_positions[vehicle_id] = \
            self.order_positions[order_id]

        # Fuel and electricity usage
        fuel_used = dist * 0.05
        elec_used = 2.0

        self.fuel_used += fuel_used
        self.electricity_used += elec_used

        # Delivery
        delivered = min(10.0, self.order_demands[order_id])
        self.order_demands[order_id] -= delivered
        self.inventory -= delivered

        # Cost
        cost = (
            fuel_used * self.FUEL_COST +
            elec_used * self.ELEC_COST
        )

        # Carbon
        carbon = (
            fuel_used * self.FUEL_FACTOR +
            elec_used * self.ELEC_FACTOR +
            dist * self.TRANS_FACTOR
        )

        # Deadline decay
        self.order_deadlines -= 1

        sla_penalty = np.sum(
            (self.order_deadlines <= 0) &
            (self.order_demands > 0)
        ) * 50

        reward = - (cost + carbon + sla_penalty)

        info["cost"] = cost
        info["carbon"] = carbon
        info["delivered"] = delivered

        done = (
            np.all(self.order_demands <= 0)
            or self.step_count >= self.max_steps
        )

        return self._get_state(), reward, done, info

    # ==========================================================
    # State Representation
    # ==========================================================
    def _get_state(self):

        return np.concatenate([
            [self.inventory],
            [self.fuel_used],
            [self.electricity_used],
            self.vehicle_positions.flatten(),
            self.vehicle_available,
            self.order_demands,
            self.order_deadlines
        ])

    # ==========================================================
    # Utility
    # ==========================================================
    def state_dim(self):
        return len(self._get_state())

    def action_space(self):
        return (self.num_vehicles, self.num_orders)


# ==============================================================
# Test Script
# ==============================================================

if __name__ == "__main__":

    env = CementEnv(
        num_vehicles=5,
        num_orders=10,
        max_steps=50
    )

    state = env.reset()
    done = False

    total_cost = 0
    total_carbon = 0

    while not done:
        v = np.random.randint(env.num_vehicles)
        o = np.random.randint(env.num_orders)

        state, reward, done, info = env.step((v, o))

        total_cost += info["cost"]
        total_carbon += info["carbon"]

    print("Simulation Finished")
    print("Total Cost:", total_cost)
    print("Total Carbon:", total_carbon)
