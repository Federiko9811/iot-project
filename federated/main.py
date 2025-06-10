# federated/federated_main.py
import os
import sys

import numpy as np
import torch

from trainer import FederatedTrainer

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

BATCH_SIZE = 64
NUM_WORKERS = 15
NUM_CLIENTS = 5
NUM_ROUNDS = 30
LOCAL_EPOCHS = 5

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # IID Federated Learning
    print("=" * 80)
    print("FEDERATED LEARNING - IID")
    print("=" * 80)

    fed_trainer_iid = FederatedTrainer(
        csv_file="../datasets/train.csv",  # Adjust path as needed
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        client_fraction=1.0,
        learning_rate=0.001,
        batch_size=BATCH_SIZE,
        iid=True,
        save_dir="logs",
        experiment_name="federated_posture_iid",
    )

    fed_trainer_iid.setup_clients()
    fed_trainer_iid.train_federated()

    # Non-IID Federated Learning
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING - NON-IID")
    print("=" * 80)

    fed_trainer_non_iid = FederatedTrainer(
        csv_file="../datasets/train.csv",  # Adjust path as needed
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        client_fraction=1.0,
        learning_rate=0.001,
        batch_size=BATCH_SIZE,
        iid=False,
        save_dir="logs",
        experiment_name="federated_posture_non_iid",
    )

    fed_trainer_non_iid.setup_clients()
    fed_trainer_non_iid.train_federated()

    print(f"\nAll experiments completed!")
    print(f"To view all results, run: tensorboard --logdir=logs")
