import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew

st.set_page_config(page_title="House Prices Preprocessing", layout="wide")

@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

train, test = load_data()
st.title("🏠 House Prices - Data Preprocessing & Feature Engineering")

with st.expander("🔍 View Raw Dataset"):
    st.write("Train Data", train.head())
    st.write("Test Data", test.head())

train_ID = train['Id']
test_ID = test['Id']
y = train['SalePrice']
train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
data = pd.concat([train, test], axis=0)

st.subheader("📉 Missing Values")
missing = data.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
st.dataframe(missing)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=missing.values, y=missing.index, ax=ax)
plt.title("Missing Value Counts")
st.pyplot(fig)

st.subheader("🧼 Filling Missing Values")

for col in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']:
    data[col] = data[col].fillna('None')

for col in ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']:
    data[col] = data[col].fillna(data[col].mode()[0])

num_cols = data.select_dtypes(include=[np.number]).columns
for col in num_cols:
    data[col] = data[col].fillna(data[col].median())

st.success("✅ Missing values handled.")

st.subheader("🛠️ Feature Engineering")

data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data['TotalBath'] = (data['FullBath'] + 0.5 * data['HalfBath'] +
                     data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath'])
data['Age'] = 2025 - data['YearBuilt']
data['RemodAge'] = 2025 - data['YearRemodAdd']
data['IsRemodeled'] = (data['YearBuilt'] != data['YearRemodAdd']).astype(int)
data['SoldInYearBuilt'] = (data['YrSold'] == data['YearBuilt']).astype(int)

st.success("✅ Feature engineering done.")

st.subheader("🔁 Handling Skewed Features")
numeric_feats = data.select_dtypes(include=[np.number])
skewed_feats = numeric_feats.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = skewed_feats[abs(skewed_feats) > 0.75]

for feat in skewness.index:
    data[feat] = np.log1p(data[feat])

st.write(f"Applied log1p to {len(skewness)} features with skew > 0.75")

st.subheader("🧬 Encoding Categorical Features")
data = pd.get_dummies(data)
st.success("✅ One-hot encoding applied.")

st.subheader("📤 Final Shape of Preprocessed Data")
n_train = train.shape[0]
X_train = data[:n_train]
X_test = data[n_train:]
st.write("Train shape:", X_train.shape)
st.write("Test shape:", X_test.shape)