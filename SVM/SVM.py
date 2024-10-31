import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re

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
        y_ = np.where(y <= 0, -1, 1)  # Convert labels to -1 and 1

        for _ in range(self.iterations):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y_[indices]

            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                X_batch = X_shuffled[start:end].toarray()  # Convert sparse to dense array
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
        return np.where(decision_function >= 0, 1, -1)

# Data Preparation Functions
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df['review'], df['sentiment']  # Assuming 'review' and 'sentiment' columns

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabetical characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize each token
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Rejoin tokens to form the cleaned text
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def preprocess_data(reviews):
    return reviews.apply(clean_text)

def prepare_data(file_path):
    reviews, sentiments = load_data(file_path)
    cleaned_reviews = preprocess_data(reviews)

    # Vectorization (using sparse matrix)
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for efficiency
    X = vectorizer.fit_transform(cleaned_reviews)

    # Convert sentiments to binary (1 for positive, 0 for negative)
    y = np.where(sentiments == 'positive', 1, 0)

    return X, y

# Load and prepare data
file_path = 'IMDB Dataset.csv'  # Path to dataset
X, y = prepare_data(file_path)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
svm = SVM(learning_rate=0.001, iterations=1000, regularization_strength=0.01, batch_size=64)
svm.fit(X_train, y_train)

# Make predictions and calculate accuracy
predictions = svm.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy:.2f}')
