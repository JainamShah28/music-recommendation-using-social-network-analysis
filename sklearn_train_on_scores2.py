import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('scores2.csv', error_bad_lines=False)
df = df.dropna()

# Preprocessing
X = df['Jaccard Similarity'] 
y = df['Actual_Val']
print(X)
# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X.values.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)


# Compute R^2 score
r2 = r2_score(y_test, y_pred)
print(f'R^2 score: {r2:.4f}')