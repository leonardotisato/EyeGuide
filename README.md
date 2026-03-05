# Knowledge Distillation for Fundus Image Classification

This project proposes a **knowledge distillation** framework for improving the baseline performance of a CNN in characterizing eye cancer, by guiding it with the greater knowledge of a model that has been trained on more meaningfull and well processed data.
## Project Overview

## Directory Structure

```
knowledge_distillation/
├── src/
│   ├── main.py                      # Main training script with knowledge distillation pipeline
│   ├── zoom_experiment.py           # Experiment script for evaluating zoom/dilation levels
│   └── utils/
│       ├── dataset.py               # Dataset classes (FundusClsDataset, FundusClsDatasetZoom, PairedFundusDataset)
│       ├── model.py                 # Model architectures (ResNet18, VGG16, ViT, Swin classifiers)
│       ├── training.py              # Training and testing functions
│       ├── losses.py                # Knowledge distillation loss functions
│       ├── transforms.py            # Transforms
│       ├── metrics.py               # Evaluation metrics
│       ├── visualization.py         # Visualization and logging utilities
│       ├── seed.py                  # Random seed
│       ├── generals.py              # General utility functions
├── config/
│   └── config.yaml                  # Hydra configuration file
├── models/                          # Directory for saved model checkpoints
├── exec.sh                          # Docker execution script
└── README.md                       
```

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

---

**Last Updated**: February 2026
