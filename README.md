# Physics-Informed Deep Learning with Monotonicity Constraints for Few-Shot Prediction of Poisson’s Ratio in Auxetic Metamaterials

This repository provides the source code used for the study:

**Physics-Informed Deep Learning with Monotonicity Constraints for Few-Shot Prediction of Poisson’s Ratio in Auxetic Metamaterials**

The objective of this study is to develop a physics-informed deep learning framework for predicting the effective Poisson’s ratio of auxetic metamaterials under limited-data and out-of-distribution conditions. A monotonicity-based physics-informed loss is introduced along the structural scale axis, defined by the number of repeated unit cells, to improve extrapolation stability and prediction reliability.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Code Organization](#code-organization)
- [Experimental Scenarios](#experimental-scenarios)
- [Physics-Informed Monotonicity Loss](#physics-informed-monotonicity-loss)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Contact](#contact)

## Overview

Mechanical metamaterials, especially auxetic structures, exhibit geometry-dependent effective mechanical properties such as negative Poisson’s ratio. However, finite element analysis of multi-cell auxetic structures requires high computational cost, especially when the design space includes multiple geometric variables, material contrast, and structural scale.

To address this issue, this repository implements deep learning-based surrogate models for predicting the effective Poisson’s ratio of auxetic metamaterials.

The input variables are:

```text
t1, t2, d, Cell, Er
```

where:

- `t1`: horizontal member thickness
- `t2`: diagonal member thickness
- `d`: inner gap
- `Cell`: number of repeated unit cells
- `Er`: elastic modulus ratio between the auxetic framework and the surrounding matrix

The output variable is:

```text
effective Poisson's ratio
```

The implemented neural network backbones include:

- Multilayer Perceptron (MLP)
- ResNet-style Tabular Network
- TabNet
- FT-Transformer (FTT)

Each model is evaluated with and without the proposed physics-informed monotonicity loss.

## Repository Structure

```text
PIDL-Auxetic-Poisson/
├── README.md
│
├── Primary exploration Scenario - w PI_loss__MLP/
│   ├── Primary_MLP_PI_pretrained.py
│   └── Primary_MLP_PI_finetuning.py
│
├── Primary exploration Scenario - w PI_loss_FTT/
│   ├── Primary_FTT_PI_pretrained.py
│   └── Primary_FTT_PI_finetuning.py
│
├── Primary exploration Scenario - w PI_loss_ResNet-style Tabular Network/
│   ├── Primary_ResNet-style Tabular Network_PI_pretrained.py
│   └── Primary_ResNet-style Tabular Network_PI_finetuning.py
│
├── Primary exploration Scenario - w PI_loss_TabNet/
│   ├── Primary_TabNet_PI_pretrained.py
│   └── Primary_TabNet_PI_finetuning.py
│
├── Primary exploration Scenario - w.o PI_loss_FTT/
│   ├── Primary_FTT_pretrained.py
│   └── Primary_FTT_finetuning.py
│
├── Primary exploration Scenario - w.o PI_loss_MLP/
│   ├── Primary_MLP_pretrained.py
│   └── Primary_MLP_finetuning.py
│
├── Primary exploration Scenario - w.o PI_loss_ResNet-style Tabular Network/
│   ├── Primary_ResNet-style Tabular Network_pretrained.py
│   └── Primary_ResNet-style Tabular Network_finetuning.py
│
├── Primary exploration Scenario - w.o PI_loss_TabNet/
│   ├── Primary_TabNet_pretrained.py
│   └── Primary_TabNet_finetuning.py
│
├── Proximal and Extreme exploration scenario - w PI_loss_FTT/
│   ├── Proximal and Extreme_FTT_PI_pretrained.py
│   └── Proximal and Extreme_FTT_PI_finetuning.py
│
├── Proximal and Extreme exploration scenario - w PI_loss_MLP/
│   ├── Proximal and Extreme_MLP_PI_pretrained.py
│   └── Proximal and Extreme_MLP_PI_finetuning.py
│
├── Proximal and Extreme exploration scenario - w PI_loss_ResNet-style Tabular Network/
│   ├── Proximal and Extreme_ResNet-style Tabular Network_PI_pretrained.py
│   └── Proximal and Extreme_ResNet-style Tabular Network_PI_finetuning.py
│
├── Proximal and Extreme exploration scenario - w PI_loss_TabNet/
│   ├── Proximal and Extreme_TabNet_PI_pretrained.py
│   └── Proximal and Extreme_TabNet_PI_finetuning.py
│
├── Proximal and Extreme exploration scenario - w.o PI_loss_FTT/
│   ├── Proximal and Extreme_FTT_pretrained.py
│   └── Proximal and Extreme_FTT_finetuning.py
│
├── Proximal and Extreme exploration scenario - w.o PI_loss_MLP/
│   ├── Proximal and Extreme_MLP_pretrained.py
│   └── Proximal and Extreme_MLP_finetuning.py
│
├── Proximal and Extreme exploration scenario - w.o PI_loss_ResNet-style Tabular Network/
│   ├── Proximal and Extreme_ResNet-style Tabular Network_pretrained.py
│   └── Proximal and Extreme_ResNet-style Tabular Network_finetuning.py
│
└── Proximal and Extreme exploration scenario - w.o PI_loss_TabNet/
    ├── Proximal and Extreme_TabNet_pretrained.py
    └── Proximal and Extreme_TabNet_finetuning.py
```

## Installation

It is recommended to create an independent virtual environment to avoid package conflicts.

### 1. Install Anaconda

Download and install Anaconda from:

```text
https://www.anaconda.com/download/
```

### 2. Create a virtual environment

```bash
conda create --name pidl_auxetic python=3.10
```

### 3. Activate the virtual environment

```bash
conda activate pidl_auxetic
```

### 4. Install required Python packages

```bash
pip install numpy
pip install pandas
pip install scipy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install openpyxl
pip install tensorflow
```

## Code Organization

Each folder corresponds to a specific experimental condition.

The folder name indicates:

```text
Scenario type + PI loss condition + model backbone
```

For example:

```text
Primary exploration Scenario - w PI_loss__MLP
```

means:

```text
Primary extrapolation scenario
Model: MLP
Physics-informed monotonicity loss: applied
```

Each folder generally contains two Python files:

```text
*_pretrained.py
*_finetuning.py
```

where:

- `*_pretrained.py`: source-domain pre-training code
- `*_finetuning.py`: target-domain few-shot fine-tuning and evaluation code

## Experimental Scenarios

Three extrapolation scenarios are considered in this study.

### Primary Extrapolation Scenario

```text
Source domain: Cell = 1, 3, 5
Target domain: Cell = 7
```

This scenario evaluates whether the model can predict a larger structural scale that is not included in the source-domain training data.

### Proximal Extrapolation Scenario

```text
Source domain: Cell = 1, 3
Target domain: Cell = 5
```

This scenario evaluates near-range extrapolation performance from smaller cell structures to a moderately larger cell structure.

### Extreme Extrapolation Scenario

```text
Source domain: Cell = 1, 3
Target domain: Cell = 7
```

This scenario evaluates far-range extrapolation performance under a more severe out-of-distribution condition.

## Physics-Informed Monotonicity Loss

The proposed physics-informed loss consists of two terms:

```text
L_total = L_MSE + alpha * L_mono
```

where:

- `L_MSE`: regression loss between the predicted and FEA-based effective Poisson’s ratio
- `L_mono`: monotonicity penalty imposed along the Cell direction
- `alpha`: weighting factor for the physics-informed monotonicity loss

The monotonicity loss is designed to suppress physically inconsistent predictions during extrapolation along the structural scale axis. In this study, the Cell variable is treated as an ordered scalar variable in the neural network input, and the monotonicity penalty is computed using the derivative of the model output with respect to the Cell variable.

The default value of the loss weight is:

```text
alpha = 10
```

## Usage

Move to the folder corresponding to the model and scenario that you want to run.

### Example 1. Primary extrapolation scenario with MLP and PI loss

```bash
cd "Primary exploration Scenario - w PI_loss__MLP"
python Primary_MLP_PI_pretrained.py
python Primary_MLP_PI_finetuning.py
```

### Example 2. Primary extrapolation scenario with MLP without PI loss

```bash
cd "Primary exploration Scenario - w.o PI_loss_MLP"
python Primary_MLP_pretrained.py
python Primary_MLP_finetuning.py
```

### Example 3. Proximal and Extreme extrapolation scenario with MLP and PI loss

```bash
cd "Proximal and Extreme exploration scenario - w PI_loss_MLP"
python "Proximal and Extreme_MLP_PI_pretrained.py"
python "Proximal and Extreme_MLP_PI_finetuning.py"
```

### Example 4. Proximal and Extreme extrapolation scenario with MLP without PI loss

```bash
cd "Proximal and Extreme exploration scenario - w.o PI_loss_MLP"
python "Proximal and Extreme_MLP_pretrained.py"
python "Proximal and Extreme_MLP_finetuning.py"
```

The same procedure can be applied to the other model backbones:

```text
MLP
FTT
ResNet-style Tabular Network
TabNet
```

## Results

The proposed physics-informed monotonicity loss improves prediction accuracy and stability, particularly under few-shot and out-of-distribution extrapolation conditions.

The main evaluation metrics are:

```text
MAE
RMSE
R2
Inference time
```

The comparison is performed between:

```text
w PI_loss
w.o PI_loss
```

for each model backbone and each extrapolation scenario.

## License

This repository is released for academic research purposes.

## Contact

Minwook Park  
Department of Mechanical Engineering  
Seoul National University of Science and Technology (Seoultech)
Email: 25510093@seoultech.ac.kr
