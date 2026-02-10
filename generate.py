import torch
import numpy as np
from environment import CementEnv


# ==========================================================
# Collect One Episode
# ==========================================================
def collect_episode(env, policy="random"):

    states = []
    actions = []
    costs = []
    carbons = []

    state = env.reset()
    done = False

    while not done:

        # -------------------------------
        # Choose Action
        # -------------------------------
        if policy == "random":
            vehicle = np.random.randint(env.num_vehicles)
            order = np.random.randint(env.num_orders)

        elif policy == "greedy":

            # Select random vehicle
            vehicle = np.random.randint(env.num_vehicles)

            # Compute distance from vehicle to all orders
            distances = np.linalg.norm(
                env.order_positions - env.vehicle_positions[vehicle],
                axis=1
            )

            # Mask completed orders
            valid_mask = env.order_demands > 0

            if np.sum(valid_mask) == 0:
                break

            masked_distances = np.where(valid_mask, distances, np.inf)

            order = np.argmin(masked_distances)

        else:
            raise ValueError("Unknown policy type")

        action = (vehicle, order)

        # -------------------------------
        # Step Environment
        # -------------------------------
        next_state, reward, done, info = env.step(action)

        # -------------------------------
        # Store Transition
        # -------------------------------
        states.append(state)
        actions.append([vehicle, order])
        costs.append(info["cost"])
        carbons.append(info["carbon"])

        state = next_state

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "costs": np.array(costs),
        "carbons": np.array(carbons)
    }


# ==========================================================
# Generate Multiple Episodes
# ==========================================================
def generate_dataset(num_episodes=200):

    env = CementEnv()

    dataset = []

    for i in range(num_episodes):

        if i % 2 == 0:
            traj = collect_episode(env, policy="random")
        else:
            traj = collect_episode(env, policy="greedy")

        dataset.append(traj)

        print(f"Episode {i+1}/{num_episodes} collected")

    return dataset


# ==========================================================
# Add Returns-To-Go (For Decision Transformer)
# ==========================================================
def add_returns_to_go(dataset):

    for traj in dataset:

        costs = traj["costs"]
        carbons = traj["carbons"]

        # Reverse cumulative sum
        returns_to_go = np.flip(np.cumsum(np.flip(costs)))
        carbon_to_go = np.flip(np.cumsum(np.flip(carbons)))

        traj["returns_to_go"] = returns_to_go
        traj["carbon_to_go"] = carbon_to_go

    return dataset


# ==========================================================
# Main Execution
# ==========================================================
if __name__ == "__main__":

    print("Generating dataset...")

    dataset = generate_dataset(num_episodes=300)

    print("Adding returns-to-go...")

    dataset = add_returns_to_go(dataset)

    torch.save(dataset, "trajectory_dataset.pt")

    print("Dataset saved successfully.")
