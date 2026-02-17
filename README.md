# Knowledge Distillation for Fundus Image Classification

This project implements **knowledge distillation** techniques for training student models to classify fundus (retinal) images. Knowledge distillation is a model compression technique where a smaller student model learns from a larger, pre-trained teacher model to improve its performance while maintaining efficiency.

## Project Overview

The knowledge distillation folder contains a complete pipeline for:
- Training teacher models (VGG16, ResNet18, ViT, Swin) on fundus image classification tasks
- Training student models using knowledge distillation with combined loss functions
- Evaluating the effectiveness of zoom-level augmentation strategies
- Analyzing confidence intervals and statistical significance of results

## Directory Structure

```
knowledge_distillation/
├── src/
│   ├── main.py                      # Main training script with knowledge distillation pipeline
│   ├── zoom_experiment.py           # Experiment script for evaluating zoom/dilation levels
│   ├── check_overlaps.py            # Statistical analysis of confidence intervals
│   └── utils/
│       ├── dataset.py               # Dataset classes (FundusClsDataset, FundusClsDatasetZoom, PairedFundusDataset)
│       ├── model.py                 # Model architectures (ResNet18, VGG16, ViT, Swin classifiers)
│       ├── training.py              # Training and testing functions
│       ├── losses.py                # Knowledge distillation loss functions
│       ├── transforms.py            # Image augmentation transforms
│       ├── metrics.py               # Evaluation metrics
│       ├── visualization.py         # Visualization and logging utilities
│       ├── seed.py                  # Random seed management
│       ├── generals.py              # General utility functions
│       └── zoom_experiment.py       # Zoom level evaluation utilities
├── config/
│   └── config.yaml                  # Hydra configuration file
├── models/                          # Directory for saved model checkpoints
├── results/                         # Directory for results and metrics
├── logs/                            # Directory for training logs
├── outputs/                         # Directory for Hydra outputs
├── exec.sh                          # Docker execution script
└── README.md                        # This file
```

## Key Features

### 1. Knowledge Distillation Pipeline
- **Teacher-Student Architecture**: Trains large teacher models to guide smaller student models
- **Soft Targets**: Uses softmax distributions from teacher predictions as soft labels for student training
- **Combined Loss**: Combines cross-entropy loss with knowledge distillation loss via weighted combination
  - `loss = beta * KD_loss + (1 - beta) * CE_loss`

### 2. Model Architectures Supported
- **ResNet18**: Lightweight convolutional neural network
- **VGG16**: Deeper CNN architecture
- **ViT (Vision Transformer)**: Transformer-based image classification
- **Swin Transformer**: Hierarchical vision transformer

### 3. Data Augmentation Strategies
- **Zoom/Dilation Augmentation**: Applies varying dilation percentages to focus on different regions of fundus images
- **Paired Dataset**: Creates paired teacher-student samples for knowledge distillation
- **Train-Val-Test Split**: Supports configurable data splits with stratified k-fold cross-validation

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confidence intervals for statistical significance testing
- Cross-validation performance analysis

## Configuration

The training pipeline is configured via `config/config.yaml`. Key parameters include:

```yaml
# Model & Training
n_epochs: 500              # Number of training epochs
batch_size: 16             # Batch size
RANDOM_SEED: 42            # Random seed for reproducibility

# Knowledge Distillation
temperature: 3.0           # Temperature for softening probability distributions
beta: 0.5                  # Weight for KD loss (vs CE loss)
soft_w: 0.75              # Weight for soft targets
ce_w: 0.25                # Weight for cross-entropy

# Data Augmentation
dilation_percentage: 1.0   # Level of zoom/dilation augmentation

# Paths
base_dir: /home/virginia/PerspectiveStudy
data_dir: /home/virginia/EyeCancerDetection
csv_file: fundus_data_final.csv

# Training Configuration
teacher_lr: 1e-4          # Teacher learning rate
student_lr: 1e-4          # Student learning rate
weight_decay: 1e-4        # L2 regularization
patience: 30              # Early stopping patience
```

## Usage

### Basic Training with Knowledge Distillation

```bash
python src/main.py
```

This will:
1. Load the fundus image dataset from configured paths
2. Train a teacher model on the full dataset
3. Train a student model using knowledge distillation from the teacher
4. Save models, logs, and results to the specified directories

### Docker Execution

To run the training in a Docker container (GPU-enabled):

```bash
bash exec.sh
```

The `exec.sh` script will:
- Mount necessary directories from the host
- Install required dependencies (wandb, albumentations)
- Run the main training script inside the container

### Zoom Level Evaluation

To evaluate different zoom/dilation levels:

```bash
python src/zoom_experiment.py
```

This script performs k-fold cross-validation for different zoom levels and reports mean F1 scores.

### Statistical Analysis

To check for statistical significance of results:

```bash
python src/check_overlaps.py
```

This analyzes confidence intervals of different model variants.

## Key Scripts

### `src/main.py`
Main training script implementing the knowledge distillation pipeline:
- Loads configurations using Hydra
- Prepares datasets with optional zoom augmentation
- Trains teacher and student models
- Saves checkpoints and metrics
- Logs results in CSV format

### `src/zoom_experiment.py`
Evaluates the impact of different zoom/dilation levels on model performance:
- Performs k-fold stratified cross-validation
- Tests zoom levels: 0.0, 0.5, 1.0, and Original
- Reports mean and standard deviation of F1 scores

### `src/check_overlaps.py`
Statistical analysis tool for confidence intervals:
- Compares performance confidence intervals between student and distilled student models
- Determines statistical significance of performance differences

## Utilities

### Dataset Classes
- **FundusClsDataset**: Standard fundus image classification dataset
- **FundusClsDatasetZoom**: Dataset with zoom/dilation augmentation
- **PairedFundusDataset**: Creates paired samples for teacher-student training

### Loss Functions
- **KD Loss**: Knowledge distillation loss using soft targets
- **Cross-Entropy Loss**: Standard classification loss

### Model Utilities
- Classifier wrappers for different architectures
- Model initialization and checkpoint management

## Dependencies

Key Python packages:
- `torch`: Deep learning framework
- `torchvision`: Image utilities and pre-trained models
- `hydra-core`: Configuration management
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning utilities
- `albumentations`: Image augmentation library
- `wandb`: Experiment tracking (optional)

Install dependencies:
```bash
pip install torch torchvision hydra-core omegaconf pandas numpy scikit-learn albumentations
```

## Output Files

After training, the following are generated:

- **Models**: Saved in `models/` directory
  - Teacher model checkpoint
  - Student model checkpoint
  - Best validation checkpoint

- **Results**: Saved in `results/` directory
  - Metrics CSV files (accuracy, precision, recall, F1)
  - Performance plots and visualizations

- **Logs**: Saved in `logs/` directory
  - Training logs with loss curves
  - Validation metrics per epoch

- **Hydra Outputs**: Automatic output directory with configuration snapshots

## Performance Metrics

The project tracks and reports:
- Per-epoch training and validation losses
- Per-epoch accuracy, precision, recall, F1-score
- Confidence intervals and statistical tests
- Cross-validation performance statistics

## Notes

- The project uses stratified k-fold cross-validation for robust evaluation
- Early stopping is implemented with configurable patience
- All results are reproducible using the fixed random seed
- The pipeline supports both CPU and GPU training (GPU recommended)

## Related Directories

This project is part of a larger PerspectiveStudy framework:
- `../models/`: Core model implementations
- `../datasets/`: Dataset utilities
- `../configs/`: Global configuration templates
- `../trainers/`: Training utilities and managers
- `../RESULTS/`: Aggregated results from various experiments

---

**Last Updated**: February 2026
