import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
file_path = "car data.csv"
df = pd.read_csv(file_path)
# Strip column names to remove any leading/trailing spaces
df.columns = df.columns.str.strip()
# Display dataset columns
print("\nColumns in dataset:", df.columns)
# Check for missing values
print("\nMissing values in dataset:")
print(df.isnull().sum())
# Rename columns to maintain consistency
df.rename(columns={'Kms_Driven': 'Driven_kms'}, inplace=True)
# Encode categorical variables (Convert text to numbers)
encoder = LabelEncoder()
categorical_columns = ['Fuel_Type', 'Selling_type', 'Transmission', 'Owner']
for col in categorical_columns:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col])
# Feature selection: Remove unnecessary columns
X = df.drop(columns=['Car_Name', 'Selling_Price'])  # Independent variables
y = df['Selling_Price']  # Target variable
# Print selected features
print("\nSelected Features:")
print(X.head())
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
# Print evaluation metrics
print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared Score (R²): {r2}")

# Save the trained model
joblib.dump(model, "car_price_model.pkl")
# Load the trained model
loaded_model = joblib.load("car_price_model.pkl")
# Create a sample input matching the correct feature names
sample_input = pd.DataFrame([{
    'Year': 2014,
    'Present_Price': 5.59,
    'Driven_kms': 27000,  # Ensure correct column name
    'Fuel_Type': 1,  # Adjust based on encoding
    'Selling_type': 1,  # Adjust based on encoding
    'Transmission': 0,  # Adjust based on encoding
    'Owner': 0  # Adjust based on encoding
}])
sample_input = sample_input[X.columns]
# Predict the car price for the sample input
sample_pred = loaded_model.predict(sample_input)
print(f"\nPredicted Selling Price: {sample_pred[0]:.2f} Lakhs")

# Data visualization
plt.figure(figsize=(8, 5))
sns.histplot(df['Selling_Price'], bins=30, kde=True, color="blue")
plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price (in Lakhs)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred, color="red")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.show()







