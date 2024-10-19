import numpy as np
import pandas as pd

class NeuralNetwork:
  def __init__(self, input_size, hidden_size, output_size):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.weights1 = np.random.randn(self.input_size, self.hidden_size)
    self.weights2 = np.random.randn(self.hidden_size, self.output_size)

  @staticmethod
  def sigmoid(x):
    return 1/(1+np.exp(-x))

  @staticmethod
  def sigmoid_derivative(x):
    return x * (1 - x)


  def forward_pass(self, inputs):
    self.hidden_output = self.sigmoid(np.dot(inputs, self.weights1))
    self.output = self.sigmoid(np.dot(self.hidden_output, self.weights2))
    return self.output

  

  def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

  def stochastic_gradient_descent():
    gradient = error * sigmoid_derivative(output)

  def back_prop(self, inputs, y_true, learning_rate):
    error = y_true - self.output
    gradient_output = error * self.sigmoid_derivative(self.output)

    self.weights2 += learning_rate * np.dot(self.hidden_output.T, gradient_output)

    gradient_hidden = np.dot(gradient_output, self.weights2.T) * self.sigmoid_derivative(self.hidden_output)

    self.weights1 += learning_rate * np.dot(inputs.T, gradient_hidden)

  def train(self, inputs, y_true, learning_rate, epochs):
      for epoch in range(epochs):
        output = self.forward_pass(inputs)
        self.back_prop(inputs, y_true, learning_rate)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('breast-cancer.csv')

data.drop(data.columns[0], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0}) 



X = data.drop(columns=['diagnosis']).values
y = data['diagnosis'].values.reshape(-1, 1)  # Reshape for the neural network

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_size = X_train.shape[1]  
hidden_size = 10  
output_size = 1  

nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

nn.train(X_train, y_train, learning_rate=0.1, epochs=1000)

predictions = nn.forward_pass(X_test)

binary_predictions = (predictions > 0.5).astype(int)

accuracy = np.mean(binary_predictions == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
