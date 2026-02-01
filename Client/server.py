import torch
from Client.model import LogisticModel, fedavg


class FederatedServer:
    def __init__(self, input_dim, device="cpu"):
        self.device = device
        self.global_model = LogisticModel(input_dim).to(device)

    def get_global_weights(self):
        return {
            k: v.detach().cpu()
            for k, v in self.global_model.state_dict().items()
        }

    def set_global_weights(self, weights):
        self.global_model.load_state_dict(weights)

    def federated_round(self, clients, epochs=10, lr=0.001):
        client_weights = []
        client_sizes = []

        global_weights = self.get_global_weights()

        for client in clients:
            # send global weights only
            client.set_weights(global_weights)

            weights, size = client.train(
                epochs=epochs,
                lr=lr
            )

            client_weights.append(weights)
            client_sizes.append(size)

        new_global_weights = fedavg(
            client_weights,
            client_sizes
        )

        self.set_global_weights(new_global_weights)

    def train(self, clients, rounds=10, epochs=10, lr=0.001):
        # initialize client models onetime
        for client in clients:
            client.model = LogisticModel(
                self.global_model.fc.in_features
            ).to(self.device)

        for r in range(rounds):
            self.federated_round(
                clients,
                epochs=epochs,
                lr=lr
            )
            print(f"Federated round {r + 1} completed")
# server flow 
'''# the idea here is the model global_model  get_global_weights 
send to clients whyu? to let clients train locally and return updated weights

          then we have fedavg on model.py do avg weights set_global_weights '''