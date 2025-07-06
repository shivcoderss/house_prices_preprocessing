# 🏠 House Prices - Data Preprocessing & Feature Engineering App

This is a **Streamlit** web app for preprocessing and feature engineering on the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) dataset.

The goal is to prepare clean, engineered data suitable for use in any machine learning model, with easy-to-understand visualizations and steps.

---

## 🚀 Key Features

- 📊 Load & explore the raw dataset
- 📉 Analyze and visualize missing values
- 🧼 Fill missing values intelligently (using `None`, `mode`, `median`)
- 🛠️ Feature engineering:
  - Total square footage (`TotalSF`)
  - Total number of bathrooms (`TotalBath`)
  - Age of the house (`Age`)
  - Remodeling flags, etc.
- 📐 Log-transform skewed features
- 🧬 One-hot encode categorical variables
- 📤 Final display of preprocessed train and test data

---

## 📁 File Structure

- house-price-preprocessing-app/
- │
- ├── app.py # Main Streamlit app
- ├── train.csv # Kaggle training dataset (add manually)
- ├── test.csv # Kaggle test dataset (add manually)
- ├── requirements.txt # Python package requirements
- ├── README.md # You're reading this!

---

## 📦 Installation

### 1. Clone this repo

```bash
git clone https://github.com/shivcoderss/house-price-preprocessing-app.git
cd house-price-preprocessing-app
