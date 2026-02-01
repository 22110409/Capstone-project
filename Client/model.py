import torch
import torch.nn as nn

class LogisticModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x)
    


def fedavg(client_weights, client_sizes):
    total = sum(client_sizes)
    avg = {}

    for k in client_weights[0].keys():
        avg[k] = sum(w[k] * (sz / total) for w, sz in zip(client_weights, client_sizes)
        )

    return avg
