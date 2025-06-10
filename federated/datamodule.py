from typing import List

import lightning as pl
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset


class AugmentedPostureDataset(Dataset):
    """Custom dataset with real-time augmentation"""

    def __init__(
        self, features, labels, augment=True, noise_std=0.05, augment_prob=0.5
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.noise_std = noise_std
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx].clone()
        y = self.labels[idx]

        if self.augment and torch.rand(1) < self.augment_prob:
            x = self._augment_sample(x)

        return x, y

    def _augment_sample(self, x):
        """Apply augmentation to a single sample"""
        augmented = x.clone()

        # 1. Add Gaussian noise to simulate sensor noise
        noise = torch.normal(0, self.noise_std, size=x.shape)
        augmented += noise

        # 2. Small angle variations (±2 degrees converted to your scale)
        angle_noise = torch.normal(0, 0.02, size=x.shape)
        augmented += angle_noise

        # 3. Simulate slight measurement inconsistencies
        # Add correlated noise between neck and torso angles (they're related)
        if len(x) >= 2:  # neck_angle and torso_angle
            correlation_noise = torch.normal(0, 0.01, size=(1,)).item()  # Get scalar value
            augmented[0] += correlation_noise  # neck_angle
            augmented[1] += correlation_noise * 0.5  # torso_angle (less correlated)

        return augmented


class FederatedPostureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_file: str,
        num_clients: int = 5,
        batch_size: int = 32,
        num_workers: int = 4,
        iid: bool = True,
        alpha: float = 0.5,
        augment_data: bool = True,
        augment_factor: float = 2.0,  # How much to increase dataset size
        use_smote: bool = True,
        noise_std: float = 0.05,
        augment_prob: float = 0.5,
    ):
        super().__init__()
        self.csv_file = csv_file
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.iid = iid
        self.alpha = alpha
        self.augment_data = augment_data
        self.augment_factor = augment_factor
        self.use_smote = use_smote
        self.noise_std = noise_std
        self.augment_prob = augment_prob

        self.scaler = StandardScaler()
        self.client_datasets = []
        self.test_ds = None

    def _generate_synthetic_samples(self, X, y):
        """Generate synthetic samples using multiple techniques"""
        synthetic_X, synthetic_y = [], []

        if self.use_smote and len(np.unique(y)) > 1:
            # Use SMOTE for balanced synthetic generation
            smote = SMOTE(random_state=42, k_neighbors=min(3, len(X) - 1))
            try:
                X_smote, y_smote = smote.fit_resample(X, y)
                # Only keep the synthetic samples (SMOTE returns original + synthetic)
                n_original = len(X)
                synthetic_X.append(X_smote[n_original:])
                synthetic_y.append(y_smote[n_original:])
            except ValueError:
                print("SMOTE failed, falling back to noise-based augmentation")

        # Noise-based augmentation
        n_synthetic = int(len(X) * (self.augment_factor - 1))
        if n_synthetic > 0:
            # Randomly select samples to augment
            indices = np.random.choice(len(X), size=n_synthetic, replace=True)

            for idx in indices:
                original_sample = X[idx].copy()
                original_label = y[idx]

                # Add controlled noise
                noise = np.random.normal(0, self.noise_std, size=original_sample.shape)
                synthetic_sample = original_sample + noise

                # Add some physiologically reasonable variations
                # For posture data, small angle changes are realistic
                angle_variation = np.random.normal(0, 0.02, size=original_sample.shape)
                synthetic_sample += angle_variation

                synthetic_X.append(synthetic_sample)
                synthetic_y.append(original_label)

        if synthetic_X:
            if len(synthetic_X) == 1:
                return synthetic_X[0], np.array(synthetic_y)
            else:
                return np.vstack(synthetic_X), np.hstack(synthetic_y)
        else:
            return np.array([]), np.array([])

    def _add_regularization_noise(self, X, y):
        """Add regularization through controlled data corruption"""
        noisy_X = []
        noisy_y = []

        # Create "hard" examples by adding more noise to some samples
        n_hard_examples = int(len(X) * 0.1)  # 10% hard examples
        hard_indices = np.random.choice(len(X), size=n_hard_examples, replace=False)

        for idx in hard_indices:
            sample = X[idx].copy()
            label = y[idx]

            # Add stronger noise to create challenging examples
            strong_noise = np.random.normal(0, self.noise_std * 2, size=sample.shape)
            noisy_sample = sample + strong_noise

            noisy_X.append(noisy_sample)
            noisy_y.append(label)

        return np.array(noisy_X), np.array(noisy_y)

    def setup(self, stage: str) -> None:
        # Load original data
        df = pd.read_csv(self.csv_file)
        X_original = df[
            ["neck_angle", "torso_angle", "shoulders_offset", "relative_neck_angle"]
        ].values
        y_original = df["good_posture"].astype(int).values

        print(f"Original dataset size: {len(X_original)}")

        # Apply augmentation if enabled
        if self.augment_data and stage == "fit":
            # Generate synthetic samples
            X_synthetic, y_synthetic = self._generate_synthetic_samples(
                X_original, y_original
            )

            # Add regularization noise
            X_noisy, y_noisy = self._add_regularization_noise(X_original, y_original)

            # Combine all data
            X_combined = [X_original]
            y_combined = [y_original]

            if len(X_synthetic) > 0:
                X_combined.append(X_synthetic)
                y_combined.append(y_synthetic)
                print(f"Added {len(X_synthetic)} synthetic samples")

            if len(X_noisy) > 0:
                X_combined.append(X_noisy)
                y_combined.append(y_noisy)
                print(f"Added {len(X_noisy)} noisy regularization samples")

            X = np.vstack(X_combined)
            y = np.hstack(y_combined)

            print(
                f"Augmented dataset size: {len(X)} (factor: {len(X) / len(X_original):.2f}x)"
            )
        else:
            X, y = X_original, y_original

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        if stage == "fit":
            # Partition data across clients
            if self.iid:
                client_indices = self._partition_data_iid(len(X_scaled))
            else:
                client_indices = self._partition_data_non_iid(y)

            # Create client datasets with augmentation
            self.client_datasets = []
            for indices in client_indices:
                if len(indices) > 0:
                    client_X = X_scaled[indices]
                    client_y = y[indices]

                    # Create augmented dataset for this client
                    client_dataset = AugmentedPostureDataset(
                        client_X,
                        client_y,
                        augment=self.augment_data,
                        noise_std=self.noise_std,
                        augment_prob=self.augment_prob,
                    )
                    self.client_datasets.append(client_dataset)
                else:
                    # Fallback for empty client
                    self.client_datasets.append(
                        AugmentedPostureDataset(X_scaled[:1], y[:1], augment=False)
                    )

        if stage == "test":
            # Don't augment test data
            self.test_ds = TensorDataset(
                torch.FloatTensor(X_scaled), torch.LongTensor(y)
            )

    def _partition_data_iid(self, dataset_size: int) -> List[List[int]]:
        """Partition data indices in IID manner"""
        indices = np.random.permutation(dataset_size)
        client_indices = np.array_split(indices, self.num_clients)
        return [idx.tolist() for idx in client_indices]

    def _partition_data_non_iid(self, labels: np.ndarray) -> List[List[int]]:
        """Partition data indices in non-IID manner using Dirichlet distribution"""
        num_classes = len(np.unique(labels))
        client_indices = [[] for _ in range(self.num_clients)]

        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            np.random.shuffle(class_indices)

            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            proportions = np.cumsum(proportions)

            start_idx = 0
            for client_id in range(self.num_clients):
                end_idx = int(proportions[client_id] * len(class_indices))
                client_indices[client_id].extend(class_indices[start_idx:end_idx])
                start_idx = end_idx

        for client_id in range(self.num_clients):
            np.random.shuffle(client_indices[client_id])

        return client_indices

    def get_client_dataloader(self, client_id: int) -> DataLoader:
        """Get dataloader for specific client"""
        if client_id >= len(self.client_datasets):
            raise ValueError(
                f"Client {client_id} does not exist. Only {len(self.client_datasets)} clients available."
            )

        return DataLoader(
            self.client_datasets[client_id],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def get_client_data_info(self) -> dict:
        """Get information about data distribution across clients"""
        info = {}
        for i, dataset in enumerate(self.client_datasets):
            # Count labels in the dataset
            labels = [dataset.labels[j].item() for j in range(len(dataset))]
            unique, counts = np.unique(labels, return_counts=True)
            info[f"client_{i}"] = {
                "total_samples": len(dataset),
                "class_distribution": dict(zip(unique.tolist(), counts.tolist())),
            }
        return info


# class FederatedPostureDataModule(pl.LightningDataModule):
#     def __init__(
#             self,
#             csv_file: str,
#             num_clients: int = 5,
#             batch_size: int = 32,
#             num_workers: int = 4,
#             iid: bool = True,
#             alpha: float = 0.5  # For Dirichlet distribution in non-IID case
#     ):
#         super().__init__()
#         self.csv_file = csv_file
#         self.num_clients = num_clients
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.iid = iid
#         self.alpha = alpha
#
#         self.scaler = StandardScaler()
#         self.client_datasets = []
#         self.test_ds = None
#
#     def prepare_data(self):
#         # Load and validate data
#         df = pd.read_csv(self.csv_file)
#         required_columns = ['neck_angle', 'torso_angle', 'shoulders_offset', 'relative_neck_angle', 'good_posture']
#         assert all(col in df.columns for col in required_columns), f"Missing required columns: {required_columns}"
#
#     def _partition_data_iid(self, dataset_size: int) -> List[List[int]]:
#         """Partition data indices in IID manner"""
#         indices = np.random.permutation(dataset_size)
#         client_indices = np.array_split(indices, self.num_clients)
#         return [idx.tolist() for idx in client_indices]
#
#     def _partition_data_non_iid(self, labels: np.ndarray) -> List[List[int]]:
#         """Partition data indices in non-IID manner using Dirichlet distribution"""
#         num_classes = len(np.unique(labels))
#         client_indices = [[] for _ in range(self.num_clients)]
#
#         for class_id in range(num_classes):
#             class_indices = np.where(labels == class_id)[0]
#             np.random.shuffle(class_indices)
#
#             # Use Dirichlet distribution to determine how to split this class
#             proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
#             proportions = np.cumsum(proportions)
#
#             start_idx = 0
#             for client_id in range(self.num_clients):
#                 end_idx = int(proportions[client_id] * len(class_indices))
#                 client_indices[client_id].extend(class_indices[start_idx:end_idx])
#                 start_idx = end_idx
#
#         # Shuffle indices for each client
#         for client_id in range(self.num_clients):
#             np.random.shuffle(client_indices[client_id])
#
#         return client_indices
#
#     def setup(self, stage: str) -> None:
#         # Load data
#         df = pd.read_csv(self.csv_file)
#
#         # Separate features and target
#         X = df[['neck_angle', 'torso_angle', 'shoulders_offset', 'relative_neck_angle']].values
#         y = df['good_posture'].astype(int).values
#
#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)
#
#         # Create full dataset
#         full_dataset = TensorDataset(
#             torch.FloatTensor(X_scaled),
#             torch.LongTensor(y)
#         )
#
#         if stage == "fit":
#             # Partition data across clients
#             if self.iid:
#                 client_indices = self._partition_data_iid(len(full_dataset))
#             else:
#                 client_indices = self._partition_data_non_iid(y)
#
#             # Create client datasets
#             self.client_datasets = []
#             for indices in client_indices:
#                 if len(indices) > 0:  # Ensure client has data
#                     client_dataset = Subset(full_dataset, indices)
#                     self.client_datasets.append(client_dataset)
#                 else:
#                     # If no data, give at least one sample
#                     self.client_datasets.append(Subset(full_dataset, [0]))
#
#         if stage == "test":
#             self.test_ds = full_dataset
#
#     def get_client_dataloader(self, client_id: int) -> DataLoader:
#         """Get dataloader for specific client"""
#         if client_id >= len(self.client_datasets):
#             raise ValueError(f"Client {client_id} does not exist. Only {len(self.client_datasets)} clients available.")
#
#         return DataLoader(
#             self.client_datasets[client_id],
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=True,
#             persistent_workers=True if self.num_workers > 0 else False,
#         )
#
#     def test_dataloader(self) -> EVAL_DATALOADERS:
#         return DataLoader(
#             self.test_ds,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#             persistent_workers=True if self.num_workers > 0 else False,
#         )
#
#     def get_client_data_info(self) -> dict:
#         """Get information about data distribution across clients"""
#         info = {}
#         for i, dataset in enumerate(self.client_datasets):
#             labels = [dataset.dataset.tensors[1][idx].item() for idx in dataset.indices]
#             unique, counts = np.unique(labels, return_counts=True)
#             info[f'client_{i}'] = {
#                 'total_samples': len(dataset),
#                 'class_distribution': dict(zip(unique.tolist(), counts.tolist()))
#             }
#         return info
