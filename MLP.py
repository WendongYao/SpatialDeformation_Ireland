import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm

data = pd.read_csv('/workspaces/SpatialDeformation_Ireland/Dataset/EGMS_L3_E32N34_100km_U_2018_2022_1_spatial.csv')


# Assuming 'data' is your dataset DataFrame
# Select features and target variable
features = data[['easting', 'northing', 'height', 'rmse', 'mean_velocity', 'mean_velocity_std', 'acceleration', 'acceleration_std', 'seasonality', 'seasonality_std', 'D-1', 'D-2', 'D-3']]
target = data['deformation']

# Data normalization
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Convert to Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 128  # Set batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the multilayer perceptron (MLP) model
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = MLP(input_size=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Record loss value of each epoch
train_losses = []
test_losses = []

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    for batch_features, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
        optimizer.zero_grad()
        output = model(batch_features)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    average_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(average_train_loss)

    # Compute loss on the test set
    model.eval()
    epoch_test_loss = 0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            output = model(batch_features)
            loss = criterion(output, batch_labels)
            epoch_test_loss += loss.item()
    average_test_loss = epoch_test_loss / len(test_loader)
    test_losses.append(average_test_loss)

    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {average_train_loss:.4f}, Test Loss: {average_test_loss:.4f}')

# Plot training and testing loss over epochs

plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Testing Loss')
plt.xlabel('Epochs', fontsize=30)  # Adjust font size for x-label
plt.ylabel('Loss', fontsize=30)    # Adjust font size for y-label
plt.title('')  # Adjust font size for title
plt.xticks(fontsize=24)  # Adjust font size for x-axis tick labels
plt.yticks(fontsize=24)  # Adjust font size for y-axis tick labels
plt.legend()
plt.show()



# Evaluate the model and record predicted and true values
model.eval()
test_loss = 0
train_loss = 0
train_predictions = []
train_true_values = []
test_predictions = []
test_true_values = []

# Compute predictions and true values on the training set
with torch.no_grad():
    for batch_features, batch_labels in train_loader:
        predictions = model(batch_features)
        loss = criterion(predictions, batch_labels)
        train_loss += loss.item()
        train_predictions.extend(predictions.numpy())
        train_true_values.extend(batch_labels.numpy())

average_train_loss = train_loss / len(train_loader)
print(f'Train Loss: {average_train_loss:.4f}')

# Compute predictions and true values on the test set
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        predictions = model(batch_features)
        loss = criterion(predictions, batch_labels)
        test_loss += loss.item()
        test_predictions.extend(predictions.numpy())
        test_true_values.extend(batch_labels.numpy())

average_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {average_test_loss:.4f}')

# Compute RMSE, MSE, R²
from sklearn.metrics import mean_squared_error, r2_score

# Compute MSE, RMSE, R² on the training set
train_mse = mean_squared_error(train_true_values, train_predictions)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(train_true_values, train_predictions)

# Compute MSE, RMSE, R² on the test set
test_mse = mean_squared_error(test_true_values, test_predictions)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(test_true_values, test_predictions)

# Output results
print(f'Train MSE: {train_mse:.4f}, Train RMSE: {train_rmse:.4f}, Train R²: {train_r2:.4f}')
print(f'Test MSE: {test_mse:.4f}, Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}')

# Compute accuracy
train_predictions = np.array(train_predictions)
train_true_values = np.array(train_true_values)
test_predictions = np.array(test_predictions)
test_true_values = np.array(test_true_values)

train_accuracy = np.mean(np.abs((train_predictions - train_true_values)) < 1)
test_accuracy = np.mean(np.abs((test_predictions - test_true_values)) < 1)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Plot comparison of predicted and true values on the training set
plt.figure(figsize=(10, 6))
plt.plot(train_true_values, label='True Values')
plt.plot(train_predictions, label='Predictions', alpha=0.7)
plt.xlabel('Sample')
plt.ylabel('Deformation')
plt.title(' ')
plt.legend()
plt.show()

# Plot comparison of predicted and true values on the test set
plt.figure(figsize=(10, 6))
plt.plot(test_true_values, label='True Values', alpha=0.6)
plt.plot(test_predictions, label='Predictions', alpha=0.6)
plt.xlabel('Sample Index', fontsize=30)
plt.ylabel('Deformation', fontsize=30)
plt.xticks(fontsize=24)  # Adjust font size for x-axis tick labels
plt.yticks(fontsize=24)  # Adjust font size for y-axis tick labels
plt.title(' ')
plt.legend()
plt.show()
