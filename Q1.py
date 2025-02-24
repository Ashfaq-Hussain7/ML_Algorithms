from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load MNIST from OpenML
mnist = fetch_openml('mnist_784', version=1)

# Extract images and labels
X = mnist.data.astype(np.float32) / 255.0  # Normalize pixel values
y = mnist.target.astype(np.int32)

print("Dataset shape:", X.shape, y.shape)

# One-hot encode labels
y = to_categorical(y, 10)

# Split dataset into train, validation, and test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Train shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)
print("Test shape:", X_test.shape, y_test.shape)
