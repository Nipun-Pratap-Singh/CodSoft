# Step 1: Data Loading
import pandas as pd

# Load the dataset from CSV file
data = pd.read_csv('iris.csv')

# Step 2: Data Exploration
print(data.head())

# Step 3: Data Preprocessing
X = data.drop('species', axis=1)  # Features
y = data['species']  # Target variable

# Step 4: Model Selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 5: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)  # Using Logistic Regression as the classifier
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 7: Prediction
# Example prediction
example_data = [[5.1, 3.5, 1.4, 0.2]]  # Example sepal and petal measurements
predicted_species = model.predict(example_data)
print("Predicted species:", predicted_species[0])
