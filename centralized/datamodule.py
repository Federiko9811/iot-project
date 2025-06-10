import lightning as pl
import pandas as pd
import torch
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler


class PostureDataModule(pl.LightningDataModule):
    def __init__(self, csv_file: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.scaler = StandardScaler()

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self):
        # Load and validate data (called only once and on single GPU)
        df = pd.read_csv(self.csv_file)
        # Basic validation
        required_columns = [
            "neck_angle",
            "torso_angle",
            "shoulders_offset",
            "relative_neck_angle",
            "good_posture",
        ]
        assert all(
            col in df.columns for col in required_columns
        ), f"Missing required columns: {required_columns}"

    def setup(self, stage: str) -> None:
        # Load data
        df = pd.read_csv(self.csv_file)

        # Separate features and target
        X = df[
            ["neck_angle", "torso_angle", "shoulders_offset", "relative_neck_angle"]
        ].values
        y = df["good_posture"].astype(int).values

        # Setup datasets for different stages
        if stage == "fit":
            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Create entire dataset
            entire_dataset = TensorDataset(
                torch.FloatTensor(X_scaled), torch.LongTensor(y)
            )

            # Calculate splits based on dataset length
            total_size = len(entire_dataset)
            train_size = int(0.8 * total_size)  # 80% for training
            val_size = total_size - train_size  # 20% for validation

            self.train_ds, self.val_ds = random_split(
                entire_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test":
            # For test, we'll use the same data but scaled with the fitted scaler
            # In practice, you might want a separate test file
            X_scaled = self.scaler.transform(X)

            self.test_ds = TensorDataset(
                torch.FloatTensor(X_scaled), torch.LongTensor(y)
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )
