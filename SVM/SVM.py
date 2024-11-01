import numpy as np

# Define a small sample dataset (X) and labels (y)
X_sample = np.array([
    [2, 3],
    [1, 5],
    [2, 8],
    [5, 2],
    [8, 1],
    [7, 2]
])

y_sample = np.array([1, 1, 1, 0, 0, 0])  # Labels: 1 for one class, 0 for the other

# Updated SVM class
class SVM:
    def __init__(self, learning_rate=0.001, iterations=1000, regularization_strength=0.01, batch_size=64):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization_strength = regularization_strength
        self.batch_size = batch_size
        self.w = None
        self.b = None

    def stochastic_gradient_descent(self, X, y):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0
        y_ = np.where(y == 0, -1, 1)  # Convert 0 to -1 for SVM

        for _ in range(self.iterations):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y_[indices]

            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                X_batch = X_shuffled[start:end]  # No need for toarray()
                y_batch = y_shuffled[start:end]

                decision = y_batch * (X_batch.dot(self.w) + self.b)
                mask = decision < 1  # Only update where the margin is violated

                dw = (2 * self.regularization_strength * self.w) - np.mean((mask * y_batch)[:, None] * X_batch, axis=0)
                db = -np.mean(mask * y_batch)

                # Update weights and bias
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

    def fit(self, X, y):
        self.stochastic_gradient_descent(X, y)

    def predict(self, X):
        decision_function = X.dot(self.w) + self.b
        predictions = np.where(decision_function >= 0, 1, 0)  # Change -1 back to 0
        return predictions

# Initialize and train the SVM
svm = SVM(learning_rate=0.001, iterations=10, regularization_strength=0.01, batch_size=2)
svm.fit(X_sample, y_sample)

# Predict on the sample data
predictions = svm.predict(X_sample)

# Output the predictions and accuracy
print("Predictions:", predictions)
print("True Labels:", y_sample)
accuracy = np.mean(predictions == y_sample)
print(f"Accuracy: {accuracy:.2f}")
