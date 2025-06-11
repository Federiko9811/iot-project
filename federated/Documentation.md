# Federated Learning for Posture Classification - Code Walkthrough

## Overview

This project implements a federated learning system for posture classification using PyTorch Lightning and TensorBoard. The system trains a neural network to classify posture as "good" or "bad" using sensor data (neck angle, torso angle, shoulders offset, and relative neck angle) across multiple simulated clients without centralizing their data.

## Architecture Overview

The federated learning system consists of six main components:

1. **Model** (`model.py`) - Neural network architecture
2. **Data Module** (`datamodule.py`) - Data loading and distribution
3. **Client** (`client.py`) - Individual client training logic
4. **Server** (`server.py`) - Central aggregation server
5. **Trainer** (`trainer.py`) - Orchestrates the federated training
6. **Main** (`main.py`) - Entry point and experiment configuration

## File-by-File Code Analysis

### 1. Model Architecture (`model.py`)

#### Core Neural Network
```python
class PostureMLP(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        # MLP with 3 layers: 4→64→32→2
        self.fc1 = nn.Linear(4, 64)   # Input: 4 posture features
        self.fc2 = nn.Linear(64, 32)  # Hidden layer
        self.fc3 = nn.Linear(32, 2)   # Output: 2 classes (good/bad posture)
        self.dropout = nn.Dropout(0.2) # Regularization
```

**Key Features:**
- **Input**: 4 features (neck_angle, torso_angle, shoulders_offset, relative_neck_angle)
- **Architecture**: Simple 3-layer MLP with ReLU activations and dropout
- **Output**: Binary classification (good posture vs bad posture)

#### Training Logic
The model implements PyTorch Lightning's standard training loop:

1. **Forward Pass**: Data flows through the three linear layers with ReLU activations
2. **Loss Calculation**: Uses CrossEntropyLoss for classification
3. **Metrics Tracking**: Tracks accuracy for train/validation/test phases
4. **Visualization**: Creates feature distribution plots and confusion matrices

#### Advanced Features
- **TensorBoard Integration**: Logs model graphs, weight histograms, and feature distributions
- **Confusion Matrix**: Visualizes classification performance
- **Feature Analysis**: Plots distribution of input features by class

### 2. Data Management (`datamodule.py`)

#### Federated Data Distribution

The data module handles two critical aspects:

**A. Data Augmentation**
```python
class AugmentedPostureDataset(Dataset):
    def _augment_sample(self, x):
        # 1. Gaussian noise (sensor noise simulation)
        noise = torch.normal(0, self.noise_std, size=x.shape)
        
        # 2. Small angle variations (±2 degrees)
        angle_noise = torch.normal(0, 0.02, size=x.shape)
        
        # 3. Correlated noise between neck and torso angles
```

**B. Client Data Partitioning**

**IID (Independent and Identically Distributed):**
```python
def _partition_data_iid(self, dataset_size: int):
    indices = np.random.permutation(dataset_size)
    client_indices = np.array_split(indices, self.num_clients)
```
- Each client gets a random subset of data
- Data distribution is similar across clients

**Non-IID (Statistical Heterogeneity):**
```python
def _partition_data_non_iid(self, labels: np.ndarray):
    # Uses Dirichlet distribution for uneven class distribution
    proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
```
- Clients have different class distributions
- More realistic scenario for federated learning

#### Data Augmentation Techniques

1. **SMOTE (Synthetic Minority Oversampling)**:
   - Generates synthetic samples for class balance
   - Creates new samples by interpolating between existing ones

2. **Noise-Based Augmentation**:
   - Adds Gaussian noise to simulate sensor variations
   - Includes correlated noise between related features

3. **Regularization Samples**:
   - Creates "hard examples" with increased noise
   - Improves model robustness

### 3. Client Implementation (`client.py`)

#### Local Training Process

Each client maintains its own copy of the global model and trains locally:

```python
def local_train(self, epochs: int = 5, learning_rate: float = 0.001):
    self.model.train()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for batch in self.dataloader:
            # Standard PyTorch training loop
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
```

**Key Responsibilities:**
1. **Model Updates**: Receives global model weights from server
2. **Local Training**: Trains on local data for specified epochs
3. **Weight Sharing**: Returns updated weights and dataset size
4. **Local Evaluation**: Measures performance on local data

### 4. Server Coordination (`server.py`)

#### FedAvg Algorithm Implementation

The server implements the Federated Averaging (FedAvg) algorithm:

```python
def aggregate_weights(self, client_weights: List[Dict], client_sizes: List[int]):
    total_samples = sum(client_sizes)
    
    for key in aggregated_weights.keys():
        aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])
        
        for i, client_weight in enumerate(client_weights):
            weight = client_sizes[i] / total_samples  # Weighted by dataset size
            aggregated_weights[key] += weight * client_weight[key]
```

**Server Functions:**
1. **Client Selection**: Randomly selects subset of clients per round
2. **Weight Aggregation**: Combines client updates using weighted averaging
3. **Global Model Update**: Updates global model with aggregated weights
4. **Performance Monitoring**: Evaluates global model and logs metrics

#### TensorBoard Integration
- Logs global model performance
- Tracks individual client metrics
- Records model weight histograms
- Enables comprehensive experiment monitoring

### 5. Training Orchestration (`trainer.py`)

#### Federated Training Loop

The trainer orchestrates the entire federated learning process:

```python
def train_federated(self):
    for round_num in range(self.num_rounds):
        # 1. Client Selection
        selected_clients = self.server.select_clients(round_num)
        
        # 2. Local Training
        for client_id in selected_clients:
            client.update_model(self.server.get_global_weights())
            weights, size = client.local_train(epochs=self.local_epochs)
            
        # 3. Aggregation
        aggregated_weights = self.server.aggregate_weights(client_weights, client_sizes)
        
        # 4. Global Update
        self.server.update_global_model(aggregated_weights)
        
        # 5. Evaluation
        global_eval = self.server.evaluate_global_model(test_dataloader, round_num)
```

#### Experiment Management
- **Hyperparameter Logging**: Records all training parameters
- **Checkpoint Saving**: Saves model states at regular intervals
- **Data Visualization**: Creates data distribution plots
- **Performance Tracking**: Maintains training history

### 6. Experiment Execution (`main.py`)

#### Comparative Analysis Setup

The main script runs two experiments:

1. **IID Experiment**:
   ```python
   fed_trainer_iid = FederatedTrainer(
       iid=True,
       experiment_name="federated_posture_iid"
   )
   ```

2. **Non-IID Experiment**:
   ```python
   fed_trainer_non_iid = FederatedTrainer(
       iid=False,
       experiment_name="federated_posture_non_iid"
   )
   ```

## Federated Learning Process Flow

### Step-by-Step Execution:

1. **Initialization**:
   - Load posture dataset from CSV
   - Create global model (PostureMLP)
   - Initialize server and clients
   - Set up TensorBoard logging

2. **Data Distribution**:
   - Apply data augmentation (SMOTE, noise injection)
   - Partition data across clients (IID or non-IID)
   - Create client-specific dataloaders

3. **Training Rounds** (repeated for `num_rounds`):
   - **Client Selection**: Server selects subset of clients
   - **Model Distribution**: Send global weights to selected clients
   - **Local Training**: Each client trains for `local_epochs`
   - **Weight Collection**: Gather updated weights from clients
   - **Aggregation**: Server combines weights using FedAvg
   - **Global Update**: Update global model with aggregated weights
   - **Evaluation**: Test global model performance

4. **Monitoring and Logging**:
   - Log individual client performance
   - Track global model accuracy and loss
   - Save model checkpoints
   - Create visualizations (data distribution, confusion matrices)

## Key Concepts Implemented

### 1. Federated Averaging (FedAvg)
- Weighted averaging based on client dataset sizes
- Preserves data privacy by only sharing model weights
- Proven effective for non-IID data distributions

### 2. Data Heterogeneity Simulation
- **IID**: Uniform data distribution across clients
- **Non-IID**: Realistic heterogeneous data using Dirichlet distribution
- Enables comparison of federated learning performance

### 3. Data Augmentation for Federated Learning
- **SMOTE**: Balances class distributions
- **Noise Injection**: Simulates sensor variations
- **Real-time Augmentation**: Applies transformations during training

### 4. Comprehensive Monitoring
- **TensorBoard Integration**: Visual experiment tracking
- **Client-level Metrics**: Individual performance monitoring
- **Model Analysis**: Weight distributions and feature analysis

## Hyperparameters and Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `num_clients` | 5 | Number of federated clients |
| `num_rounds` | 30 | Federated training rounds |
| `local_epochs` | 5 | Local training epochs per round |
| `client_fraction` | 1.0 | Fraction of clients selected per round |
| `learning_rate` | 0.001 | Learning rate for local training |
| `batch_size` | 64 | Batch size for training |
| `augment_factor` | 10.0 | Data augmentation multiplier |

## Expected Outputs

### Performance Metrics
- **Global Model Accuracy**: Performance on centralized test set
- **Client Accuracies**: Individual client performance
- **Training Loss**: Model convergence tracking

### Visualizations
- **Data Distribution**: Class balance across clients
- **Learning Curves**: Accuracy and loss over rounds
- **Confusion Matrices**: Classification performance analysis
- **Feature Distributions**: Input data characteristics

### Checkpoints
- Model states saved every 10 rounds
- Include hyperparameters and performance metrics
- Enable experiment reproducibility

## Running the Experiments

1. **Prerequisites**:
   ```bash
   pip install torch lightning tensorboard scikit-learn imbalanced-learn
   ```

2. **Data Preparation**:
   - Ensure `train.csv` exists with columns: `neck_angle`, `torso_angle`, `shoulders_offset`, `relative_neck_angle`, `good_posture`

3. **Execution**:
   ```bash
   python main.py
   ```

4. **Monitoring**:
   ```bash
   tensorboard --logdir=logs
   ```

## Research Applications

This implementation enables investigation of:
- **Federated vs Centralized Learning**: Performance comparison
- **Data Heterogeneity Impact**: IID vs Non-IID scenarios
- **Client Participation**: Effect of client selection strategies
- **Data Augmentation**: Benefits in federated settings
- **Privacy-Preserving ML**: Healthcare applications

The code provides a complete framework for federated learning research with comprehensive logging, visualization, and experimental comparison capabilities.