import copy
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter


class FederatedServer:
    def __init__(
        self,
        model: nn.Module,
        num_clients: int,
        client_fraction: float = 1.0,
        logger: TensorBoardLogger = None,
    ):
        self.global_model = model
        self.num_clients = num_clients
        self.client_fraction = client_fraction
        self.round_history = []

        # TensorBoard integration
        self.logger = logger
        if self.logger:
            self.writer = self.logger.experiment
        else:
            # Fallback to direct SummaryWriter
            self.writer = SummaryWriter(log_dir="logs/federated_learning")

    def select_clients(self, round_num: int) -> List[int]:
        """Select subset of clients for this round"""
        num_selected = max(1, int(self.client_fraction * self.num_clients))
        np.random.seed(round_num)
        selected_clients = np.random.choice(
            range(self.num_clients), size=num_selected, replace=False
        ).tolist()
        return selected_clients

    def aggregate_weights(
        self, client_weights: List[Dict], client_sizes: List[int]
    ) -> Dict:
        """Implement FedAvg algorithm"""
        total_samples = sum(client_sizes)
        aggregated_weights = copy.deepcopy(client_weights[0])

        for key in aggregated_weights.keys():
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])

            for i, client_weight in enumerate(client_weights):
                weight = client_sizes[i] / total_samples
                aggregated_weights[key] += weight * client_weight[key]

        return aggregated_weights

    def update_global_model(self, aggregated_weights: Dict):
        """Update global model with aggregated weights"""
        self.global_model.load_state_dict(aggregated_weights)

    def get_global_weights(self) -> Dict:
        """Get current global model weights"""
        return copy.deepcopy(self.global_model.state_dict())

    def evaluate_global_model(self, test_dataloader, round_num: int) -> Dict:
        """Evaluate global model on test data and log to TensorBoard"""
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in test_dataloader:
                inputs, labels = batch
                outputs = self.global_model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_dataloader)

        # Log to TensorBoard
        self.writer.add_scalar("Global/Accuracy", accuracy, round_num)
        self.writer.add_scalar("Global/Loss", avg_loss, round_num)

        return {"accuracy": accuracy, "loss": avg_loss}

    def log_client_metrics(
        self,
        client_accuracies: List[float],
        client_losses: List[float],
        selected_clients: List[int],
        round_num: int,
    ):
        """Log individual client metrics to TensorBoard"""
        avg_client_acc = np.mean(client_accuracies)
        avg_client_loss = np.mean(client_losses)

        # Log average client metrics
        self.writer.add_scalar("Clients/Average_Accuracy", avg_client_acc, round_num)
        self.writer.add_scalar("Clients/Average_Loss", avg_client_loss, round_num)

        # Log individual client metrics
        for i, (client_id, acc, loss) in enumerate(
            zip(selected_clients, client_accuracies, client_losses)
        ):
            self.writer.add_scalar(f"Client_{client_id}/Accuracy", acc, round_num)
            self.writer.add_scalar(f"Client_{client_id}/Loss", loss, round_num)

    def log_model_weights(self, round_num: int):
        """Log model weight histograms to TensorBoard"""
        for name, param in self.global_model.named_parameters():
            self.writer.add_histogram(f"Global_Weights/{name}", param, round_num)
