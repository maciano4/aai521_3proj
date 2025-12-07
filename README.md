# Flood Detection Segmentation - Spatio-Temporal Analysis

<div align="center">

**University of San Diego**  
AAI-521: Applied Computer Vision for AI

---

**Professor**: Mohammad Yavarimanesh

**Section**: 5  
**Group**: 3

**Contributors**:

- Swapnil Patil
- Christopher Akeibom Toh
- Nelson Arellano Parra

</div>

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Project Architecture](#project-architecture)
5. [Models Implemented](#models-implemented)
6. [Results & Performance](#results--performance)
7. [Installation & Setup](#installation--setup)
8. [Usage Guide](#usage-guide)
9. [Project Structure](#project-structure)
10. [Key Findings](#key-findings)
11. [Contributing](#contributing)
12. [References](#references)

---

## 🎯 Project Overview

This project develops a **spatio-temporal flood detection and damage assessment system** using multi-temporal satellite imagery. The system combines pre- and post-event satellite images to detect flooded areas and assess infrastructure damage through semantic segmentation.

**Key Objectives:**

- Detect flooded regions in satellite imagery with high precision
- Classify damage severity (no-damage, minor, major, destroyed)
- Perform spatio-temporal analysis using paired pre/post imagery
- Evaluate multiple state-of-the-art segmentation architectures
- Optimize models for geospatial and change detection tasks

---

## 🌍 Problem Statement

Flooding causes significant economic and humanitarian impacts globally. Rapid, accurate damage assessment is critical for disaster response and recovery planning. This project leverages computer vision to:

1. **Automatically detect flooded areas** from satellite imagery
2. **Classify building damage severity** based on changes between pre and post-event images
3. **Enable rapid deployment** for emergency response scenarios
4. **Provide quantitative metrics** for impact assessment

---

## 📊 Dataset

### Source

- **SpaceNet 8**: Flood Detection Challenge dataset
- **Format**: Multi-spectral satellite imagery (Maxar WorldView-2/3)
- **Temporal**: Pre-event and post-event image pairs

### Geographic Coverage

- **Training**: Germany and Louisiana-East regions
- **Validation**: 20% stratified split from training
- **Testing**: Louisiana-West (geographically independent)

### Dataset Statistics

- **Total Training Images**: ~2,000 pre/post pairs
- **Image Size**: 1024×1024 pixels (original), 512×512 patches (processed)
- **Spectral Bands**: 8 bands (multispectral) → 6 channels (pre+post RGB)
- **Classes**: 7 semantic classes (background, non-flooded, partially flooded, fully flooded, roads, etc.)
- **Class Distribution**: Highly imbalanced (~5-10% flood pixels)

### Data Organization

```
dataset/
├── raw/
│   ├── train/
│   │   ├── Germany_Training_Public/
│   │   └── Louisiana-East_Training_Public/
│   └── test/
│       └── Louisiana-West_Test_Public/
└── processed/
    ├── train/          # ~1,600 pairs (80%)
    ├── val/            # ~400 pairs (20%)
    └── test/           # ~100 pairs (hold-out test)
```

---

## 🏗️ Project Architecture

### System Pipeline

```
┌─────────────────────┐
│  Raw Satellite Data │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│    Data Preprocessing               │
│  • Band selection (RGB stacking)    │
│  • Patch extraction (512×512)       │
│  • Class balancing (oversampling)   │
│  • Normalization (ImageNet stats)   │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│    Data Augmentation (Albumentations)
│  • Geometric (rotate, flip, crop)   │
│  • Photometric (brightness, contrast)
│  • Spatial (elastic deformation)    │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│    Model Training & Validation      │
│  • 6 segmentation models            │
│  • Mixed precision (FP16)           │
│  • Early stopping + checkpointing   │
│  • Multi-metric tracking            │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│    Evaluation & Inference           │
│  • Per-class metrics                │
│  • Visualization & confusion matrix │
│  • Patch-based full image inference │
└─────────────────────────────────────┘
```

### Data Flow

```
Training Data (Pre/Post Images)
    │
    ├─► Preprocessing Module
    │   • Extract RGB bands
    │   • Tile into 512×512 patches
    │   • Apply normalization
    │   • Balance classes
    │
    ├─► DataLoader (PyTorch)
    │   • Batch sampling
    │   • Augmentation on-the-fly
    │   • Multi-worker loading
    │
    ├─► Model (Segmentation Network)
    │   • Encode: spatial feature extraction
    │   • Decode: upsampling + refinement
    │   • Head: per-pixel classification
    │
    ├─► Loss Computation
    │   • Cross-Entropy (base loss)
    │   • Dice Loss (boundary focus)
    │   • Focal Loss (hard examples)
    │   • Combined Weighted Loss
    │
    ├─► Optimization
    │   • AdamW optimizer
    │   • Learning rate scheduling
    │   • Gradient clipping
    │   • Mixed precision training
    │
    └─► Validation Loop
        • Epoch-level metrics
        • Best model checkpointing
        • Early stopping
```

---

## 🧠 Models Implemented

### 1. **U-Net++** (Nested U-Net)

- **Architecture**: Encoder-Decoder with nested skip connections
- **Paper**: [https://arxiv.org/abs/1807.10165](https://arxiv.org/abs/1807.10165)
- **Encoder**: ResNet-34 (pretrained ImageNet)
- **Key Features**: Dense skip paths, deep supervision capability
- **Val IoU**: 0.7189

### 2. **DeepLabV3+**

- **Architecture**: Atrous Spatial Pyramid Pooling (ASPP) + Decoder
- **Paper**: [https://arxiv.org/abs/1802.02611](https://arxiv.org/abs/1802.02611)
- **Encoder**: ResNet-50 with dilated convolutions
- **Key Features**: Multi-scale context, contextual information
- **Val IoU**: 0.6823

### 3. **SegFormer**

- **Architecture**: Vision Transformer backbone with lightweight decoder
- **Paper**: [https://arxiv.org/abs/2105.15203](https://arxiv.org/abs/2105.15203)
- **Backbone**: MiT-B2 (efficient transformer)
- **Key Features**: Global context, patch-based processing
- **Val IoU**: 0.7501

### 4. **FC-Siam-Diff** (Change Detection)

- **Architecture**: Siamese network for change detection
- **Purpose**: Direct pre/post comparison for flood detection
- **Key Features**: Difference-based feature learning
- **Val IoU**: 0.6657

### 5. **Siamese U-Net++** (Spatio-Temporal)

- **Architecture**: Dual U-Net++ with temporal fusion
- **Purpose**: Joint processing of paired images
- **Key Features**: Feature-level fusion, symmetric processing
- **Val IoU**: 0.7112

### 6. **STANet** (Spatio-Temporal Attention) ⭐ **BEST**

- **Architecture**: Spatial and temporal attention networks
- **Paper**: [https://arxiv.org/abs/2004.08305](https://arxiv.org/abs/2004.08305)
- **Key Features**: Spatio-temporal attention, change focused
- **Val IoU**: **0.7679** (Best performance)

---

## 📈 Results & Performance

### Model Comparison

| Model           | Val Loss | Val IoU    | Val Dice   | Val F1     | Best Epoch |
| --------------- | -------- | ---------- | ---------- | ---------- | ---------- |
| **STANet** ⭐   | 0.0461   | **0.7679** | **0.7779** | **0.7112** | 20         |
| SegFormer       | 0.0470   | 0.7501     | 0.7623     | 0.7079     | 20         |
| U-Net++         | 0.0576   | 0.7189     | 0.7346     | 0.6912     | 20         |
| Siamese U-Net++ | 0.0471   | 0.7112     | 0.7257     | 0.6579     | 20         |
| DeepLabV3+      | 0.0579   | 0.6823     | 0.6957     | 0.6679     | 20         |
| FC-Siam-Diff    | 0.0558   | 0.6657     | 0.6801     | 0.6412     | 20         |

### Key Metrics Explained

- **IoU (Intersection over Union)**: Measures overlap between predicted and ground truth masks
- **Dice Coefficient**: Harmonic mean of precision and recall, sensitive to class imbalance
- **F1-Score**: Balanced metric combining precision and recall
- **Loss**: Combined weighted loss (CE + Dice + Focal)

### Performance Insights

1. **STANet** excels due to spatio-temporal attention mechanisms
2. **Transformer-based models** (SegFormer) show strong generalization
3. **Siamese architectures** provide marginal benefits for change detection
4. **Attention mechanisms** are crucial for handling class imbalance

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 16GB+ RAM
- 100GB+ disk space (for dataset and models)

### Step 1: Clone Repository

```bash
git clone https://github.com/swapnilprakashpatil/aai521_3proj.git
cd aai521_3proj
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
source venv/bin/activate     # Linux/Mac

# Or using conda
conda create -n flood-detection python=3.10
conda activate flood-detection
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

- Download SpaceNet 8 dataset from [SpaceNet on AWS](https://www.drivendata.org/competitions/spacenet/)
- Extract to `dataset/raw/` directory structure as shown above

### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, GPU: {torch.cuda.is_available()}')"
```

---

## 📚 Usage Guide

### 1. Data Preprocessing

```bash
python src/run_preprocessing.py
```

This will:

- Extract RGB bands from multispectral imagery
- Create 512×512 patches from original images
- Apply class balancing (oversample flood pixels)
- Generate train/val/test splits
- Normalize using dataset statistics

### 2. Train Models

```bash
# Train a single model
python src/train.py --model stANet --batch_size 4 --num_epochs 20

# Train all models
python src/train.py --model all --batch_size 4 --num_epochs 20

# Options:
#   --model: stANet, unet++, deeplabv3+, segformer, fc_siam_diff, siamese_unet++
#   --batch_size: 2-16 (adjust based on GPU memory)
#   --num_epochs: default 20
#   --learning_rate: default 1e-4
#   --checkpoint: path to resume training
```

### 3. Evaluate Models

```bash
python src/evaluate.py --model_path outputs/models/stANet_best.pth --test_mode
```

### 4. Run Inference

```bash
python src/inference.py \
    --image_path dataset/raw/test/Louisiana-West_Test_Public/your_image.tif \
    --model_path outputs/models/stANet_best.pth \
    --output_path outputs/results/prediction.png
```

### 5. Jupyter Notebooks

```bash
jupyter notebook

# Available notebooks:
# 01_eda.ipynb - Exploratory Data Analysis
# 02_preprocessing.ipynb - Data Preprocessing Walkthrough
# 03_model_training.ipynb - Training Pipeline
# 04_model_evaluation.ipynb - Results Analysis
# flood_detection.ipynb - Complete Pipeline Demo
```

---

## 📁 Project Structure

```
aai521_3proj/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── src/                               # Source code
│   ├── config.py                      # Configuration & hyperparameters
│   ├── dataset.py                     # PyTorch Dataset class
│   ├── models.py                      # Model architectures (6 models)
│   ├── losses.py                      # Loss functions
│   ├── metrics.py                     # Metric calculations
│   ├── trainer.py                     # Training engine
│   ├── train.py                       # Main training script
│   ├── evaluate.py                    # Model evaluation
│   ├── inference.py                   # Inference pipeline
│   ├── preprocessing.py               # Data preprocessing
│   ├── run_preprocessing.py           # Preprocessing entry point
│   ├── augmentation.py                # Data augmentation (Albumentations)
│   ├── data_loader.py                 # DataLoader utilities
│   ├── gpu_manager.py                 # GPU memory management
│   ├── visualizations.py              # Plotting & visualization
│   ├── eda_utils.py                   # EDA utilities
│   ├── experiment_tracking.py         # Metrics tracking
│   └── __pycache__/                   # Compiled Python files
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb         # Preprocessing walkthrough
│   ├── 03_model_training.ipynb        # Model training
│   ├── 04_model_evaluation.ipynb      # Results analysis
│   └── flood_detection.ipynb          # Complete pipeline
│
├── dataset/                           # Data directory
│   ├── raw/
│   │   ├── train/
│   │   │   ├── Germany_Training_Public/
│   │   │   └── Louisiana-East_Training_Public/
│   │   └── test/
│   │       └── Louisiana-West_Test_Public/
│   └── processed/
│       ├── train/                     # 1,600 training patches
│       ├── val/                       # 400 validation patches
│       └── test/                      # 100 test patches
│
├── outputs/                           # Results & models
│   ├── models/                        # Trained model checkpoints
│   ├── results/                       # Evaluation results
│   ├── samples/                       # Sample predictions
│   └── logs/                          # Training logs
│
└── tmp/                               # Temporary files
```

---

## 🔍 Key Findings

### 1. Spatio-Temporal Attention is Crucial

- **STANet's advantage** comes from its ability to explicitly model temporal changes
- Pre/post image pairs benefit from attention-based comparison mechanisms
- Direct difference-based features (FC-Siam-Diff) underperform spatial reasoning

### 2. Class Imbalance Requires Aggressive Balancing

- Original flood pixels represent ~5% of dataset
- Oversampling flood patches to 50% during training significantly improves IoU
- Weighted loss functions alone insufficient without data-level balancing

### 3. Vision Transformers Show Promise

- **SegFormer** (transformer-based) ranks 2nd, suggesting global context is valuable
- Patch-based processing aligns well with satellite tile structure
- Slightly lower performance than STANet but more efficient

### 4. Multi-Loss Training is Essential

- Combined loss (CE + Dice + Focal) outperforms single-loss approaches
- Dice loss provides boundary-awareness
- Focal loss helps with hard negative examples

### 5. Preprocessing Quality is Critical

- Proper band selection and normalization significantly impact convergence
- CLAHE enhancement helps but can introduce artifacts
- Patch overlap (20% overlap) prevents boundary artifacts

---

## 📖 Training Configuration Details

### Optimization

- **Optimizer**: AdamW (learning rate: 1e-4, weight decay: 1e-5)
- **Scheduler**: Cosine annealing with warm restarts
- **Epochs**: 20 (with early stopping patience: 10)
- **Batch Size**: 4 (GPU memory constraint)
- **Gradient Clipping**: 1.0 (prevents exploding gradients)

### Data Augmentation (Albumentations)

- Geometric: Random rotations (0-45°), flips, elastic deformation
- Photometric: Brightness/contrast adjustments, color jittering
- Spatial: Random cropping (224×224), scaling (0.8-1.2x)
- Dropout: DropOut/CoarseDropout (10% pixel dropout)

### Loss Function

```
L_total = 0.5 * CE_loss + 0.3 * Dice_loss + 0.2 * Focal_loss
```

### Metrics Tracked

- Per-class IoU (7 classes)
- Per-class Precision, Recall, F1
- Overall IoU, Dice, F1 (macro-averaged)
- Validation loss

---

## 🤝 Contributing

This is a class project. For contributions:

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -m 'Add feature'`
3. Push to branch: `git push origin feature/your-feature`
4. Open a Pull Request

---

## 📚 References

### Papers

1. U-Net++: https://arxiv.org/abs/1807.10165
2. DeepLabV3+: https://arxiv.org/abs/1802.02611
3. SegFormer: https://arxiv.org/abs/2105.15203
4. STANet: https://arxiv.org/abs/2004.08305
5. Focal Loss: https://arxiv.org/abs/1708.02002

### Datasets

- SpaceNet 8: https://www.drivendata.org/competitions/spacenet/
- xBD (xView2): https://xview.colorado.edu/

### Libraries & Frameworks

- PyTorch: https://pytorch.org/
- Segmentation Models PyTorch: https://github.com/qubvel/segmentation_models.pytorch
- Albumentations: https://albumentations.ai/
- Rasterio: https://rasterio.readthedocs.io/

### Related Work

- Change Detection in Remote Sensing: https://arxiv.org/abs/2006.03056
- Satellite Image Analysis: https://arxiv.org/abs/2108.08626

---

## 📄 License

This project is developed as part of AAI-521: Applied Computer Vision for AI course at University of San Diego.

---

## 👥 Acknowledgments

- **Dataset**: SpaceNet Foundation & MaxarTechnologies
- **Course**: Professor Mohammad Yavarimanesh (AAI-521, USD)
- **Libraries**: PyTorch, OpenCV, scikit-image, and open-source community

---

**Last Updated**: December 2024  
**Status**: ✅ Complete - Final Submission
