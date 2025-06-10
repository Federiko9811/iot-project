import copy
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


class FederatedClient:
    def __init__(self, client_id: int, model: nn.Module, dataloader: DataLoader):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.dataloader = dataloader
        self.dataset_size = len(dataloader.dataset)

    def update_model(self, global_weights: Dict):
        """Update local model with global weights"""
        self.model.load_state_dict(global_weights)

    def local_train(
        self, epochs: int = 5, learning_rate: float = 0.001
    ) -> Tuple[Dict, int]:
        """Train model locally and return updated weights"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for batch in self.dataloader:
                inputs, labels = batch

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return copy.deepcopy(self.model.state_dict()), self.dataset_size

    def evaluate(self) -> Dict:
        """Evaluate local model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in self.dataloader:
                inputs, labels = batch
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = total_loss / len(self.dataloader) if len(self.dataloader) > 0 else 0

        return {"accuracy": accuracy, "loss": avg_loss}
