import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the Iris dataset
df = pd.read_csv("Iris.csv")
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Display dataset information
print(df.head())  # View first 5 rows
print(df.info())  # Data types and missing values
print(df.describe())  # Summary statistics

# Pairplot to visualize relationships between features
sns.pairplot(df, hue="Species", markers=["o", "s", "D"])
plt.show()

# Boxplot to check distributions of sepal length across species
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Species", y="SepalLengthCm")
plt.show()

# Split Features (X) and Target Variable (y)
X = df.drop(columns=['Species'])  # Features
y = df['Species']  # Target variable

# Convert Labels to Numerical Values
encoder = LabelEncoder()
y = encoder.fit_transform(y)  # 'setosa' -> 0, 'versicolor' -> 1, 'virginica' -> 2

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Get class labels from encoder
class_labels = encoder.classes_

# Plot confusion matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save the trained model
joblib.dump(model, "iris_model.pkl")
# Load the saved model
loaded_model = joblib.load("iris_model.pkl")
# Example input for prediction
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_df = pd.DataFrame(sample, columns=X.columns)

# Make a prediction
prediction = loaded_model.predict(sample_df)

# Convert numerical prediction back to species name
predicted_species = encoder.inverse_transform(prediction)[0]
print(f"Predicted species: {predicted_species}")



