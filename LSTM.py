import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Assume your data is a DataFrame
data = pd.read_csv('/workspaces/SpatialDeformation_Ireland/Dataset/EGMS_L3_E32N34_100km_U_2018_2022_1_spatial.csv')

# Feature selection
features = data[['easting', 'northing', 'height', 'rmse', 'mean_velocity', 'mean_velocity_std',
                 'acceleration', 'acceleration_std', 'seasonality', 'seasonality_std']]
target = data['deformation']

# Data standardization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_dim=features_scaled.shape[1], activation='relu'))
model.add(Dropout(0.2))  # Add Dropout layer to prevent overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=100, batch_size=128,
                    validation_split=0.2, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# Make predictions
y_pred = model.predict(X_test)

# Calculate RMSE, MSE, R²
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Calculate absolute error between predictions and true values
error = np.abs(y_test.values - y_pred.flatten())

# Count number of samples where error is less than 1
correct_predictions = np.sum(error < 1)

# Calculate accuracy
accuracy = correct_predictions / len(y_test)

print(f'AccuracyTest: {accuracy * 100:.2f}%')

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R²: {r2}')

# Make predictions on the training set
y_train_pred = model.predict(X_train)

# Calculate absolute error between training predictions and true values
train_error = np.abs(y_train.values - y_train_pred.flatten())

# Count number of training samples where error is less than 1
train_correct_predictions = np.sum(train_error < 1)

# Calculate training accuracy
train_accuracy = train_correct_predictions / len(y_train)

print(f'Train Accuracy: {train_accuracy * 100:.2f}%')

# Plot training and validation loss changes

# If you need to plot training and validation loss changes, you can use the following code
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')
plt.xlabel('Epochs', fontsize=30)  # Adjust font size for x-label
plt.ylabel('Loss', fontsize=30)     # Adjust font size for y-label
plt.legend(fontsize=20)
plt.show()

# Plot comparison of predicted and true values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='True Values', alpha=0.6)
plt.plot(y_pred, label='Predictions', alpha=0.6)
plt.xlabel('Sample Index', fontsize=30)
plt.ylabel('Deformation', fontsize=30)
plt.legend(fontsize=20, loc='upper center')
plt.show()
