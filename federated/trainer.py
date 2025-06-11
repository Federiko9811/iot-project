import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from client import FederatedClient
from datamodule import FederatedPostureDataModule
from model import PostureMLP
from server import FederatedServer


class FederatedTrainer:
    def __init__(
        self,
        csv_file: str,
        num_clients: int = 5,
        num_rounds: int = 50,
        local_epochs: int = 5,
        client_fraction: float = 1.0,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        iid: bool = True,
        save_dir: str = "logs",
        experiment_name: str = "federated_posture",
    ):
        self.csv_file = csv_file
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.client_fraction = client_fraction
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.iid = iid

        # Setup TensorBoard logger (Lightning style)
        self.logger = TensorBoardLogger(
            save_dir=save_dir,
            name=experiment_name,
            version=None,  # Auto-increment version
        )

        # Create checkpoint directory
        self.checkpoint_dir = os.path.join(self.logger.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize components
        self.datamodule = FederatedPostureDataModule(
            csv_file=csv_file,
            num_clients=num_clients,
            batch_size=batch_size,
            iid=iid,
            augment_data=True,  # Enable augmentation
            augment_factor=10.0,  # Double the dataset size
            use_smote=True,  # Use SMOTE for balanced generation
            noise_std=2,  # Noise level
            augment_prob=0.9,  # 50% chance of augmentation per sample
        )

        self.global_model = PostureMLP()
        self.server = FederatedServer(
            self.global_model, num_clients, client_fraction, logger=self.logger
        )

        # Training history
        self.history = {
            "round": [],
            "global_accuracy": [],
            "global_loss": [],
            "client_accuracies": [],
        }

    def setup_clients(self):
        """Setup federated data and create clients"""
        self.datamodule.setup("fit")
        self.datamodule.setup("test")

        # Create clients
        self.clients = []
        for i in range(self.num_clients):
            client_dataloader = self.datamodule.get_client_dataloader(i)
            client = FederatedClient(i, self.global_model, client_dataloader)
            self.clients.append(client)

        # Log data distribution info to TensorBoard
        data_info = self.datamodule.get_client_data_info()

        # Create data distribution visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        client_ids = []
        good_posture_counts = []
        bad_posture_counts = []

        for client_id, info in data_info.items():
            client_ids.append(client_id.replace("client_", "Client "))
            good_posture_counts.append(info["class_distribution"].get(1, 0))
            bad_posture_counts.append(info["class_distribution"].get(0, 0))

        x = range(len(client_ids))
        width = 0.35

        ax.bar(
            [i - width / 2 for i in x],
            bad_posture_counts,
            width,
            label="Bad Posture",
            color="red",
            alpha=0.7,
        )
        ax.bar(
            [i + width / 2 for i in x],
            good_posture_counts,
            width,
            label="Good Posture",
            color="green",
            alpha=0.7,
        )

        ax.set_xlabel("Clients")
        ax.set_ylabel("Number of Samples")
        ax.set_title(
            f'Data Distribution Across Clients ({"IID" if self.iid else "Non-IID"})'
        )
        ax.set_xticks(x)
        ax.set_xticklabels(client_ids)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.logger.experiment.add_figure("Data_Distribution", fig, 0)
        plt.close(fig)

        print("Data distribution across clients:")
        for client_id, info in data_info.items():
            print(f"{client_id}: {info}")

    def train_federated(self):
        """Main federated training loop with TensorBoard logging"""
        print(f"Starting Federated Learning with {self.num_clients} clients")
        print(f"Data distribution: {'IID' if self.iid else 'Non-IID'}")
        print(f"TensorBoard logs will be saved to: {self.logger.log_dir}")
        print("-" * 60)

        test_dataloader = self.datamodule.test_dataloader()

        # Log hyperparameters
        hparams = {
            "num_clients": self.num_clients,
            "num_rounds": self.num_rounds,
            "local_epochs": self.local_epochs,
            "client_fraction": self.client_fraction,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "iid": self.iid,
        }
        self.logger.log_hyperparams(hparams)

        for round_num in range(self.num_rounds):
            print(f"Round {round_num + 1}/{self.num_rounds}")

            # Select clients for this round
            selected_clients = self.server.select_clients(round_num)
            print(f"Selected clients: {selected_clients}")

            # Collect client updates
            client_weights = []
            client_sizes = []
            client_accuracies = []
            client_losses = []

            for client_id in selected_clients:
                client = self.clients[client_id]

                # Update client with global model
                client.update_model(self.server.get_global_weights())

                # Local training
                weights, size = client.local_train(
                    epochs=self.local_epochs, learning_rate=self.learning_rate
                )

                # Evaluate client
                client_eval = client.evaluate()

                client_weights.append(weights)
                client_sizes.append(size)
                client_accuracies.append(client_eval["accuracy"])
                client_losses.append(client_eval["loss"])

                print(
                    f"  Client {client_id}: Accuracy = {client_eval['accuracy']:.2f}%"
                )

            # Log client metrics to TensorBoard
            self.server.log_client_metrics(
                client_accuracies, client_losses, selected_clients, round_num
            )

            # Aggregate weights using FedAvg
            aggregated_weights = self.server.aggregate_weights(
                client_weights, client_sizes
            )
            self.server.update_global_model(aggregated_weights)

            # Log model weights every 5 rounds
            if round_num % 5 == 0:
                self.server.log_model_weights(round_num)

            # Evaluate global model
            global_eval = self.server.evaluate_global_model(test_dataloader, round_num)

            # Store history
            self.history["round"].append(round_num + 1)
            self.history["global_accuracy"].append(global_eval["accuracy"])
            self.history["global_loss"].append(global_eval["loss"])
            self.history["client_accuracies"].append(np.mean(client_accuracies))

            # Save checkpoint
            if round_num % 10 == 0 or round_num == self.num_rounds - 1:
                checkpoint_path = os.path.join(
                    self.logger.log_dir,
                    "checkpoints",
                    f"federated-round-{round_num:02d}-acc-{global_eval['accuracy']:.2f}.ckpt",
                )
                torch.save(
                    {
                        "round": round_num,
                        "model_state_dict": self.global_model.state_dict(),
                        "global_accuracy": global_eval["accuracy"],
                        "global_loss": global_eval["loss"],
                        "hyperparameters": hparams,
                    },
                    checkpoint_path,
                )

            print(
                f"  Global Model: Accuracy = {global_eval['accuracy']:.2f}%, Loss = {global_eval['loss']:.4f}"
            )
            print(f"  Average Client Accuracy = {np.mean(client_accuracies):.2f}%")
            print("-" * 60)

        print(f"\nFederated Training completed!")
        print(f"TensorBoard logs saved to: {self.logger.log_dir}")
        print(f"To view results, run: tensorboard --logdir={self.logger.save_dir}")
