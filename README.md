# Multi-Task Learning for Predicting House Prices and House Categories

## Overview
This project explores the application of **multi-task learning** to predict both house prices (regression task) and house categories (classification task) using a single model. By leveraging shared layers in the neural network, the model efficiently handles both tasks, achieving strong performance in prediction accuracy and computational efficiency.

---

## Dataset
The dataset consists of **2,919 houses** with **81 features** each. The data was preprocessed and refined to ensure model efficiency. For more details, visit [Kaggle Housing Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

---

## Key Steps

### Data Preprocessing
1. **Missing Value Handling**:
   - Columns with over 30% missing values were removed.
   - Features with less than 30% missing values were imputed:
     - Numerical: Filled with 0.
     - Categorical: Filled with "NA" where appropriate.
     - `LotFrontage`: Imputed using KNN Imputer.

2. **Feature Engineering**:
   - Simplified `HouseStyle` and `BldgType` into generalized categories.
   - Introduced `YearAfterRemod` to capture the recency of house remodels.
   - Created a `HouseCategory` feature using **DBSCAN clustering**, grouping houses based on style, type, and remodeling recency.

3. **Dimensionality Reduction**:
   - Applied **Principal Component Analysis (PCA)** to reduce the dataset to 25 principal components for efficient learning.

---

## Model Architecture
The **multi-task learning model** was built using **PyTorch Lightning** with the following structure:

### Shared Layers
- Input: 25 features
- Three shared layers with:
  - **Linear transformation**: Increasing and decreasing neurons (1024 → 512 → 256).
  - **LeakyReLU activations** and **dropout** (0.1 rate).

### Task-Specific Heads
1. **House Price Prediction (Regression)**:
   - Multiple layers (256 → 512 → 256 → 128 → 64 → 1).
   - Final output: Predicted house price.
   - Loss function: **Mean Squared Error (MSE)**.

2. **House Category Prediction (Classification)**:
   - Multiple layers (256 → 512 → 256 → 128 → 64 → 7).
   - Final output: Predicted house category.
   - Loss function: **CrossEntropyLoss**.

### Optimization
- **Optimizer**: Adam.
- **Hyperparameter Tuning**: Conducted with **Optuna** to optimize learning rates and batch sizes.

---

## Results
1. Initial results (basic setup):
   - **RMSE**: 185,650 (Price Prediction)
   - **Accuracy**: 17% (Category Prediction)

2. Refined model:
   - **RMSE**: 49,398.25 (Price Prediction)
   - **Accuracy**: 56.29% (Category Prediction)

3. Optuna-optimized model:
   - **RMSE**: 52,793
   - **Accuracy**: 59.73%

---

## Future Work
- Incorporate external datasets for enhanced learning.
- Experiment with transformer-based architectures for further improvements.
- Explore additional clustering methods for house categorization.
