import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch

df = pd.read_csv('EGMS_L3_E32N34_100km_U_2018_2022_1.csv')

# The target variable is 'deformation'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Feature variables include ID, longitude, latitude, height, rmse, mean_velocity, mean_velocity_std,
# acceleration, acceleration_std, seasonality, seasonality_std
X = df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]

y = df.iloc[:, 14]
# if only values not in STD form are needed, change X into this:
# X = df.iloc[:, [1, 2, 3, 4, 5, 7, 9, 11, 12, 13]]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Calculate Mean Squared Error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

importance = rf.feature_importances_
feature_names = X.columns

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importance, y=feature_names)
plt.title('Feature Importance', fontsize=30)
plt.ylabel('Proportion', fontsize=20)
plt.xlabel('Importance', fontsize=20)  # Add x-label if needed
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
