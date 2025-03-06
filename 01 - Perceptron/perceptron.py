import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

np.random.seed(42)


def forward_propagation(data, weights, bias):
    forward = np.dot(data, weights) + bias
    sig_op = sigmoid(forward)
    return sig_op


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Binary Cross-Entropy loss
def bce_loss(y_true, y_pred):
    m = len(y_true)

    loss = -(1 / m) * np.sum(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )
    return loss


# Backward propagation (gradient calculation)
def backward_propagation(X, y, y_hat):
    m = len(X)
    dl_w = (1 / m) * np.dot(X.T, (y_hat - y.reshape(-1, 1)))
    dl_b = (1 / m) * np.sum(y_hat - y)
    return dl_w, dl_b


# Initial parameters
num_feature = 10
epochs = 100
learning_rate = 0.01
data, label = make_classification(n_samples=10000, n_features=num_feature)

label = label.reshape(-1, 1)

# Initialize weights randomly
weights = np.random.randn(num_feature, 1)
bias = 1

# Training loop
for i in range(epochs):
    # Forward propagation
    forward = forward_propagation(data, weights, bias)

    forward = np.clip(forward, 1e-10, 1 - 1e-10)  # Clip to avoid log(0)

    argmax_pred = (forward > 0.5).astype(int)  # Binary classification (0 or 1)

    # Compute the loss
    loss = bce_loss(y_true=label, y_pred=forward)
    print(
        f"Epoch {i + 1}, Loss: {loss}, Accuracy: {accuracy_score(y_true=label,y_pred=argmax_pred)}"
    )

    # Backward propagation
    dl_w, dl_b = backward_propagation(X=data, y=label, y_hat=forward)

    # Update weights and bias
    weights -= learning_rate * dl_w
    bias -= learning_rate * dl_b

    # learning rate scheduling
    # learning_rate *= 0.1
