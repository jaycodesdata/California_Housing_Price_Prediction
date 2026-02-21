# 🏠 California Housing Price Prediction

## 📌 Project Overview

This project builds an end-to-end Machine Learning pipeline to predict median house values in California districts using demographic, geographic, and housing-related features from the 1990 census dataset.

The goal was to practice the complete ML workflow — from data cleaning and feature engineering to hyperparameter tuning and model interpretation — using XGBoost with GPU acceleration.

---

## 📊 Dataset

- Source: 1990 California Census Housing Data  
- Target Variable: `median_house_value`
- Features include:
  - Geographic data (latitude, longitude, ocean proximity)
  - Economic indicators (median income)
  - Structural attributes (rooms, bedrooms, households, population)

---

## 🛠 Workflow

### 1️⃣ Data Cleaning
- Removed missing values
- Outlier detection using IQR method
- Verified feature distributions

### 2️⃣ Exploratory Data Analysis
- Distribution analysis of housing prices
- Correlation heatmap
- Identification of multicollinearity
- Insight extraction from feature relationships

### 3️⃣ Feature Engineering
Created additional ratio-based features:
- `rooms_per_household`
- `bedrooms_per_room`
- `population_per_household`

These improved the model’s ability to capture housing density effects.

### 4️⃣ Modeling Approach
- Preprocessing using `ColumnTransformer`
- One-hot encoding for categorical features
- XGBoost Regressor (GPU accelerated)
- Hyperparameter tuning using `GridSearchCV`
- Early stopping for regularization

---

## 🚀 Model Performance

| Metric | Value |
|--------|--------|
| RMSE | ~45,953 |
| MAE | ~29,503 |
| R² Score | 0.815 |

The model explains over **81% of the variance** in housing prices.

---

## 🔍 Key Insights

Feature importance analysis reveals that geographic factors, particularly whether a district is inland or near the ocean, are the strongest drivers of housing prices. Median income also plays a significant role, confirming that economic conditions strongly influence property valuation. Structural features such as room ratios contribute moderately, while house age has relatively lower impact.

---

## 🧠 Why XGBoost?

XGBoost was chosen because:

- It handles non-linear relationships effectively.
- It is robust to multicollinearity.
- It includes built-in regularization.
- GPU acceleration improves training efficiency.
- Early stopping prevents overfitting.

---

## 📦 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost (GPU)
- Matplotlib
- Seaborn

---

## 📁 Project Structure
```
california-housing-price-prediction/
│
├── data/
│ └── README.md
├── california_house_Price_prediction.ipynb
└── README.md

```

