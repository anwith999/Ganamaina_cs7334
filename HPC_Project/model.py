"""
Neural Network for Regression (PyTorch)
- MSE loss (not cross-entropy)
- Linear output (no softmax)
- Supports layer splitting for model parallelism
"""
import torch, torch.nn as nn


class RegressionNN(nn.Module):
    """Multi-layer regression network: Input -> [Hidden+ReLU] x N -> Linear output."""
    def __init__(self, input_dim=128, hidden_dim=512, output_dim=8, n_hidden=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x): return self.network(x)


class ModelParallelStage(nn.Module):
    """A subset of consecutive layers for model parallelism."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden, rank, world_size):
        super().__init__()
        self.rank, self.world_size = rank, world_size
        all_mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            all_mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        all_mods.append(nn.Linear(hidden_dim, output_dim))

        # Distribute layers evenly
        n = len(all_mods)
        per = [0]*world_size
        for i in range(n): per[i % world_size] += 1
        start = sum(per[:rank]); end = start + per[rank]
        self.my_layers = nn.Sequential(*all_mods[start:end])
        self.is_first = (rank == 0)
        self.is_last = (rank == world_size - 1)

    def forward(self, x): return self.my_layers(x)


def model_size_mb(m):
    return sum(p.numel()*p.element_size() for p in m.parameters()) / (1024*1024)
