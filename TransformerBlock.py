import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    LayerNormalization,
    MultiHeadAttention,
    Dropout,
    Flatten,
    Reshape,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('/workspaces/SpatialDeformation_Ireland/Dataset/EGMS_L3_E32N34_100km_U_2018_2022_1_spatial.csv')

# Feature selection
features = data[
    [
        'easting',
        'northing',
        'height',
        'rmse',
        'mean_velocity',
        'mean_velocity_std',
        'acceleration',
        'acceleration_std',
        'seasonality',
        'seasonality_std',
        'D-1',
        'D-2',
        'D-3',
    ]
]
target = data['deformation']

# Data standardization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.2, random_state=42
)

# Transformer model
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="relu", kernel_regularizer=l2(0.01)),
                Dense(embed_dim, kernel_regularizer=l2(0.01)),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_model(
    input_shape, embed_dim, num_heads, ff_dim, num_transformer_blocks, rate=0.1
):
    inputs = Input(shape=input_shape)
    x = Dense(embed_dim)(inputs)
    x = Reshape((1, embed_dim))(x)

    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, rate)(x, training=True)

    x = Flatten()(x)
    x = Dropout(rate)(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(rate)(x)
    outputs = Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Hyperparameters
embed_dim = 64  # Increase embedding size
num_heads = 8   # Increase number of attention heads
ff_dim = 128    # Increase hidden layer size of the feed-forward network
num_transformer_blocks = 2  # Increase number of Transformer blocks
dropout_rate = 0.3  # Increase dropout rate

model = build_model(
    input_shape=(features_scaled.shape[1],),
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_transformer_blocks=num_transformer_blocks,
    rate=dropout_rate,
)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    verbose=1,
)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# Predict
y_pred = model.predict(X_test)

# Calculate RMSE, MSE, R²
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R²: {r2}')

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')
plt.xlabel('Epochs', fontsize=30)
plt.ylabel('Loss', fontsize=30)
plt.legend(fontsize=20)
plt.show()

# Plot comparison of predicted and true values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='True Values', alpha=0.6)
plt.plot(y_pred, label='Predictions', alpha=0.6)
plt.xlabel('Sample Index', fontsize=30)
plt.ylabel('Deformation', fontsize=30)
plt.legend(fontsize=20)
plt.show()
