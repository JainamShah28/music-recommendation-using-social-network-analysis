import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
import xgboost as xgb
import shap

# Load data
df = pd.read_csv('scores2.csv', error_bad_lines=False)

# Remove any NaN
df = df.dropna()

# Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(df[['Bias_U', 'Bias_V', 'Bias_U/Bias_V', 'Jaccard Similarity', 'Normalised_Weight']])
y = df['Actual_Val'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create and train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, max_depth=3, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# Generate explainable AI using fast tree explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=['Bias_U', 'Bias_V', 'Bias_U/Bias_V', 'Jaccard Similarity', 'Normalised_Weight'], plot_type='bar')

# Compute R-squared score for neural network model
y_pred_nn = xgb_model.predict(X_test).flatten()
r2_nn = r2_score(y_test, y_pred_nn)
print(f"R-squared score for neural network model: {r2_nn:.4f}")

# Compute R-squared score for XGBoost model
y_pred_xgb = xgb_model.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"R-squared score for XGBoost model: {r2_xgb:.4f}")