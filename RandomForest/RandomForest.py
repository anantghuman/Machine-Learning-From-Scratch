import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def split_data(self, X, y, feature, threshold):
        feature_index = int(feature)  # Convert feature name to index
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]
        return left_indices, right_indices

    def entropy(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        if np.all(probabilities == 0):
            return 0
        return -np.sum(probabilities * np.log2(probabilities))

    def information_gain(self, X, y, feature, threshold):
        left_indices, right_indices = self.split_data(X, y, feature, threshold)
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        p_left = len(left_indices) / len(X)
        p_right = len(right_indices) / len(X)
        return self.entropy(y) - (p_left * self.entropy(y[left_indices]) + p_right * self.entropy(y[right_indices]))

    def find_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gain = 0
        n_features = X.shape[1]
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self.information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or depth >= self.max_depth:
            return node(value=np.unique(y)[0], feature=None, threshold=None, gain=0, left=None, right=None)

        feature, threshold = self.find_best_split(X, y)
        if feature is None:
            return node(value=np.unique(y)[0], feature=None, threshold=None, gain=0, left=None, right=None)

        left_indices, right_indices = self.split_data(X, y, feature, threshold)
        left_node = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_node = self.build_tree(X[right_indices], y[right_indices], depth + 1)
        return node(feature=feature, threshold=threshold, gain=0, value=None, left=left_node, right=right_node)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict(self, X):
        return [self.make_prediction(x, self.root) for x in X]

    def make_prediction(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.make_prediction(x, node.left)
        else:
            return self.make_prediction(x, node.right)

class node():
      def __init__(self, feature, threshold, gain, value, left, right, ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

def train_test_split(x, y, random_state=1311, test_size=0.3):
    np.random.seed(random_state)
    size = len(x)
    shuffle = np.random.permutation(np.arange(size))
    test_size = int(math.floor(test_size * size))
    test_indices = shuffle[:test_size]
    train_indices = shuffle[test_size:]
    x_train = x[train_indices]  # Keep as DataFrame
    y_train = y[train_indices]  # y remains a NumPy array
    x_test = x[test_indices]  # Keep as DataFrame
    y_test = y[test_indices]  # y remains a NumPy array
    return x_train, x_test, y_train, y_test

class RandomForest:
    def __init__(self, n_trees=7, max_depth=7, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def bootstrap(self, x, y):
        n_samples = x.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return x[indices], y[indices]  # Return NumPy arrays

    def predict(self, x):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        preds = np.swapaxes(predictions, 0, 1)
        return np.array([self.most_common_label(pred) for pred in preds])

    def most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def fit(self, x, y):
        for i in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            sample_x, sample_y = self.bootstrap(x, y)  # Keep as NumPy arrays
            tree.fit(sample_x, sample_y)  # Fit tree with sampled data
            self.trees.append(tree)

def train_test_split(x, y, random_state=1312, test_size=0.9):
    np.random.seed(random_state)
    size = len(x)
    shuffle = np.random.permutation(size)
    test_size = int(np.floor(test_size * size))
    test_indices = shuffle[:test_size]
    train_indices = shuffle[test_size:]

    x_train = x[train_indices]
    y_train = y[train_indices]  # Use NumPy array indexing
    x_test = x[test_indices]
    y_test = y[test_indices]

    return x_train, x_test, y_train, y_test

data = pd.read_csv('Iris.csv')
X = data.iloc[:, :-1].values  # Features as NumPy array
y = data.iloc[:, -1].values    # Labels as NumPy array

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create and train model
model = RandomForest(n_trees=7, max_depth=7, min_samples_split=2)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)  # Calculate accuracy
print("Accuracy:", accuracy)
