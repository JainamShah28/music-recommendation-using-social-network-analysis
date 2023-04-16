import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import tensorflow_addons as tfa
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tf.config.list_physical_devices('GPU'))


# Load the data into a pandas dataframe
df = pd.read_csv('scores2.csv', error_bad_lines=False)

# Remove any NaN
df = df.dropna()

# Split the data into training and testing sets
X = df[['Bias_U', 'Bias_V', 'Bias_U/Bias_V', 'Jaccard Similarity', 'Normalised_Weight']]
y = df['Actual_Val']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""
model = keras.Sequential([
    keras.layers.Dense(1, activation='relu', input_shape=[5], 
    use_bias=True)
])
BATCH_SIZE = 64
EPOCHS = 100
INIT_LR = 1e-4
MAX_LR = 1e-2
steps_per_epoch = len(X_train_scaled) // BATCH_SIZE

clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=INIT_LR,
    maximal_learning_rate=MAX_LR,
    scale_fn=lambda x: 1/(2.**(x-1)),
    step_size=2 * steps_per_epoch
)

optimizer = tf.keras.optimizers.Adam(clr)
model.compile(loss='mse', optimizer=optimizer)

mcp_save =  tf.keras.callbacks.ModelCheckpoint('simple_best_train_on_scores2.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
model.fit(X_train_scaled, y_train, epochs=100, callbacks=[mcp_save], validation_split=0.2)
model.save('simple_train_on_scores2.h5')
"""

model = keras.models.load_model('train_on_scores2.h5', compile=False)
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

plt.scatter([i for i in range(len(y_test))], y_test, color="blue")
plt.scatter([i for i in range(len(y_test))], y_pred, color="orange")
plt.title('Actual vs Predicted Values')
plt.show()


# Generate explainable AI
import shap

# Create background data summary
background = shap.kmeans(X_train, 100)

# Generate explainable AI
explainer = shap.KernelExplainer(model.predict, background, link="identity")
shap_values = explainer.shap_values(X_test, nsamples=500)
shap.summary_plot(shap_values, X_test, feature_names=['Bias_U', 'Bias_V', 'Bias_U/Bias_V', 'Jaccard Similarity', 'Normalised_Weight'])