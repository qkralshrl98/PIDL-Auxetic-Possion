# PIDL-Auxetic-Possion
# Physics-Informed Deep Learning with Monotonicity Constraints for Few-Shot Prediction of Poisson’s Ratio in Auxetic Metamaterials

This repository provides the source code and dataset used for the study:

**Physics-Informed Deep Learning with Monotonicity Constraints for Few-Shot Prediction of Poisson’s Ratio in Auxetic Metamaterials**

The objective of this study is to develop a physics-informed deep learning framework for predicting the effective Poisson’s ratio of auxetic metamaterials under limited-data and out-of-distribution conditions. A monotonicity-based physics-informed loss is introduced along the structural scale axis, defined by the number of repeated unit cells, to improve extrapolation stability and prediction reliability.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Experimental Scenarios](#experimental-scenarios)
- [Physics-Informed Monotonicity Loss](#physics-informed-monotonicity-loss)
- [Results](#results)
- [License](#license)
- [Contact](#contact)

## Overview

Mechanical metamaterials, especially auxetic structures, exhibit geometry-dependent effective mechanical properties such as negative Poisson’s ratio. However, finite element analysis of multi-cell auxetic structures requires high computational cost, especially when the design space includes multiple geometric variables, material contrast, and structural scale.

To address this issue, this repository implements surrogate models for predicting the effective Poisson’s ratio from the following input variables:

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
- ResNet-style Tabular Network (RNTN)
- TabNet
- FT-Transformer (FTT)

Each model is evaluated with and without the proposed physics-informed monotonicity loss.

## Repository Structure

```text
PIDL-Auxetic-Poisson/
├── data/
│   ├── raw/
│   │   └── original_fea_data.xlsx
│   └── processed/
│       └── poisson_ratio_dataset.csv
├── src/
│   ├── models/
│   │   ├── mlp.py
│   │   ├── rntn.py
│   │   ├── tabnet.py
│   │   └── ftt.py
│   ├── losses/
│   │   └── pi_monotonic_loss.py
│   ├── train.py
│   ├── fine_tune.py
│   └── evaluate.py
├── results/
│   ├── figures/
│   └── tables/
├── README.md
└── requirements.txt
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

Alternatively, if `requirements.txt` is provided:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset was generated using finite element analysis of auxetic metamaterial structures. Each sample consists of geometric design variables, structural scale, material contrast, and the corresponding effective Poisson’s ratio.

The input variables are:

```text
t1, t2, d, Cell, Er
```

The target variable is:

```text
effective Poisson's ratio
```

The dataset includes four structural scale levels:

```text
Cell = 1, 3, 5, 7
```

and three elastic modulus ratio levels:

```text
Er = 10, 20, 30
```

For reproducibility, place the dataset in the following directory:

```text
data/processed/poisson_ratio_dataset.csv
```

The original raw data can be stored in:

```text
data/raw/original_fea_data.xlsx
```

## Usage

### 1. Train a baseline model

```bash
python src/train.py --model mlp --loss mse
```

### 2. Train a physics-informed model

```bash
python src/train.py --model mlp --loss pi --alpha 10
```

### 3. Fine-tune the model under few-shot target-domain adaptation

```bash
python src/fine_tune.py --model mlp --scenario primary --support_size 5
```

### 4. Evaluate the trained model

```bash
python src/evaluate.py --model mlp --scenario primary --support_size 5
```

The model option can be changed as follows:

```text
mlp
rntn
tabnet
ftt
```

## Experimental Scenarios

Three extrapolation scenarios are considered in this study.

### Primary Extrapolation Scenario

```text
Source domain: Cell = 1, 3, 5
Target domain: Cell = 7
```

### Proximal Extrapolation Scenario

```text
Source domain: Cell = 1, 3
Target domain: Cell = 5
```

### Extreme Extrapolation Scenario

```text
Source domain: Cell = 1, 3
Target domain: Cell = 7
```

For each scenario, the model is first pre-trained on the source domain and then fine-tuned using a limited number of support samples from the target domain.

The support sizes are:

```text
5, 10, 50, 100
```

## Physics-Informed Monotonicity Loss

The proposed physics-informed loss consists of two terms:

```text
L_total = L_MSE + alpha * L_mono
```

where `L_MSE` is the regression loss between the predicted and FEA-based effective Poisson’s ratio, and `L_mono` is the monotonicity penalty imposed along the Cell direction.

The monotonicity loss is designed to suppress physically inconsistent predictions during extrapolation along the structural scale axis. In this study, the Cell variable is treated as an ordered scalar variable in the neural network input, and the monotonicity penalty is computed using the derivative of the model output with respect to the Cell variable.

The default value of the loss weight is:

```text
alpha = 10
```

## Results

The proposed physics-informed loss improves prediction accuracy and stability, particularly under few-shot and out-of-distribution extrapolation conditions.

The main evaluation metrics are:

```text
MAE
RMSE
R2
Inference time
```

The results are saved in:

```text
results/tables/
results/figures/
```

## License

This repository is released for academic research purposes.

## Contact

Minwook Park  
Department of Mechanical Engineering  
Seoul National University of Science and Technology  
Email: your-email@example.com
