import torch
import numpy as np
from torch.utils.data import Dataset


class CementTrajectoryDataset(Dataset):
    def __init__(self, data_path, seq_len=20):

        self.seq_len = seq_len
        self.data = torch.load(data_path, weights_only=False)


        # Flatten all trajectories into one list
        self.samples = []

        for traj in self.data:

            states = traj["states"]
            actions = traj["actions"]
            returns = traj["returns_to_go"]
            carbons = traj["carbon_to_go"]

            length = len(states)

            # Create sliding windows
            for i in range(length - seq_len):

                self.samples.append({
                    "states": states[i:i+seq_len],
                    "actions": actions[i:i+seq_len],
                    "returns": returns[i:i+seq_len],
                    "carbons": carbons[i:i+seq_len]
                })

        # Convert to tensors
        self._convert_to_tensor()

        # Normalize states
        self._normalize_states()

    # ---------------------------------------------------------
    def _convert_to_tensor(self):

        for sample in self.samples:

            sample["states"] = torch.tensor(
                sample["states"], dtype=torch.float32
            )

            sample["actions"] = torch.tensor(
                sample["actions"], dtype=torch.float32
            )

            sample["returns"] = torch.tensor(
                sample["returns"], dtype=torch.float32
            )

            sample["carbons"] = torch.tensor(
                sample["carbons"], dtype=torch.float32
            )

    # ---------------------------------------------------------
    def _normalize_states(self):

        all_states = torch.cat(
            [s["states"] for s in self.samples], dim=0
        )

        self.state_mean = all_states.mean(0)
        self.state_std = all_states.std(0) + 1e-6

        for sample in self.samples:
            sample["states"] = (
                sample["states"] - self.state_mean
            ) / self.state_std

    # ---------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ---------------------------------------------------------
    def __getitem__(self, idx):

        sample = self.samples[idx]

        return (
            sample["states"],
            sample["actions"],
            sample["returns"],
            sample["carbons"]
        )
