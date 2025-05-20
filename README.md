# TSD-Net: Traffic Sign Detection Network

## Overview
TSD-Net is an improved framework designed to enhance the detection performance of small traffic signs in complex backgrounds. Based on the YOLOv11 architecture, TSD-Net integrates a Feature Enhancement Module (FEM) to expand the network's receptive field and enhance its capability to capture target features. Additionally, it introduces a high-resolution detection branch and an Adaptive Dynamic Feature Fusion (ADFF) detection head to optimize cross-scale feature fusion and preserve critical details of small objects. By incorporating the C3k2 module and dynamic convolution into the network, the framework achieves enhanced feature extraction flexibility while maintaining high computational efficiency.

### Key Features:
- **Feature Enhancement Module (FEM)**: Expands the network's receptive field and enhances small object feature capture
- **High-Resolution Detection Branch**: Preserves critical details of small objects
- **Adaptive Dynamic Feature Fusion (ADFF)**: Optimizes cross-scale feature fusion
- **C3k2 Module and Dynamic Convolution**: Improves feature extraction flexibility with high computational efficiency

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)
- [Dataset](#dataset)
- [License](#license)

## Installation
Clone this repository and install the required dependencies using `pip`

```bash
pip install ultralytics
```

## Usage

### Training

To train the TSD-Net model :

```python
python train.py 
```

### Inference

To run inference using a trained AS-YOLO model:

```python
python val.py 
```

## Dataset

### TT100K Dataset

The dataset should be placed in the root directory and the structure should be:

```bash
TSD-Net/
├── dataset/
│   └── TT100K/
│       ├── images/
│       │   ├── train/
│       │   │   ├── 1.jpg
│       │   │   ├── 2.jpg
│       │   │   └── ...
│       │   └── val/
│       │       ├── 10001.jpg
│       │       ├── 10002.jpg
│       │       └── ...
│       │   
│       └── labels/
│           ├── train/
│           │   ├── 1.txt
│           │   ├── 2.txt
│           │   └── ...
│           └── val/
│               ├── 10001.txt
│               ├── 10002.txt
│               └── ...
```

### CCTSDB2021 Dataset

The dataset should be placed in the root directory and the structure should be:

```bash
TSD-Net/
├── dataset/
│   └── CCTSDB/
│       ├── images/
│       │   ├── train/
│       │   │   ├── image1.jpg
│       │   │   ├── image2.jpg
│       │   │   └── ...
│       │   └── val/
│       │       ├── image1.jpg
│       │       ├── image2.jpg
│       │       └── ...
│       └── labels/
│           ├── train/
│           │   ├── image1.txt
│           │   ├── image2.txt
│           │   └── ...
│           └── val/
│               ├── image1.txt
│               ├── image2.txt
│               └── ...
```

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](./LICENSE) file for details.