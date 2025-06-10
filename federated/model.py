import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import Accuracy, ConfusionMatrix


class PostureMLP(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Define the MLP architecture
        self.fc1 = nn.Linear(4, 64)  # 4 input features
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 2 classes for binary classification
        self.dropout = nn.Dropout(0.2)

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Define metrics for tracking accuracy
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")

        # Confusion matrix for test evaluation
        self.confusion_matrix = ConfusionMatrix(task="binary")

        # Store learning rate
        self.learning_rate = learning_rate

        # Class names for visualization
        self.class_names = ["Bad Posture", "Good Posture"]

    def forward(self, x):
        # Define the forward pass
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        # This defines what happens in one training step
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        # Log histograms every 100 steps
        if batch_idx % 100 == 0:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.logger.experiment.add_histogram(
                        f"weights/{name}", param, self.global_step
                    )
                    self.logger.experiment.add_histogram(
                        f"gradients/{name}", param.grad, self.global_step
                    )

        return loss

    def validation_step(self, batch, batch_idx):
        # This defines what happens in one validation step
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        # This defines what happens in one test step
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, y)

        # Update confusion matrix
        self.confusion_matrix.update(preds, y)

        # Log metrics
        self.log("test_loss", loss)
        self.log("test_acc", acc)

        return loss

    def on_validation_epoch_end(self):
        # Log feature distributions every 5 epochs
        if self.current_epoch % 5 == 0:
            try:
                # Get validation dataloader
                val_dataloader = self.trainer.datamodule.val_dataloader()

                # Get a batch of validation data
                batch = next(iter(val_dataloader))
                features, labels = batch
                features = features[:100]  # Take first 100 samples
                labels = labels[:100]

                # Move to device
                features = features.to(self.device)

                # Get predictions
                with torch.no_grad():
                    logits = self(features)
                    preds = torch.argmax(logits, dim=1)

                # Create feature distribution plot
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                feature_names = [
                    "Neck Angle",
                    "Torso Angle",
                    "Shoulders Offset",
                    "Relative Neck Angle",
                ]

                for i, (ax, feature_name) in enumerate(zip(axes.flat, feature_names)):
                    good_posture_mask = labels == 1
                    bad_posture_mask = labels == 0

                    ax.hist(
                        features[good_posture_mask, i].cpu().numpy(),
                        alpha=0.7,
                        label="Good Posture",
                        bins=20,
                        color="green",
                    )
                    ax.hist(
                        features[bad_posture_mask, i].cpu().numpy(),
                        alpha=0.7,
                        label="Bad Posture",
                        bins=20,
                        color="red",
                    )
                    ax.set_title(f"{feature_name} Distribution")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.suptitle(
                    f"Feature Distributions - Epoch {self.current_epoch}", y=1.02
                )

                # Log to tensorboard
                self.logger.experiment.add_figure(
                    "feature_distributions", fig, self.current_epoch
                )
                plt.close(fig)

            except Exception as e:
                print(f"Could not log feature distributions: {e}")

    def on_test_epoch_end(self):
        # Compute and log confusion matrix
        cm = self.confusion_matrix.compute()

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm.cpu().numpy(),
            annot=True,
            fmt="d",
            ax=ax,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()

        # Log to tensorboard
        self.logger.experiment.add_figure("confusion_matrix", fig, self.current_epoch)
        plt.close(fig)

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_start(self):
        """Log model graph when training starts"""
        try:
            # Get a sample from the training dataloader
            sample_batch = next(iter(self.trainer.datamodule.train_dataloader()))
            sample_input = sample_batch[0][:1]  # Take just one sample

            # Move to same device as model
            sample_input = sample_input.to(self.device)

            # Log the model graph
            self.logger.experiment.add_graph(self, sample_input)
            print("Model graph logged to TensorBoard")

        except Exception as e:
            print(f"Could not log model graph: {e}")
