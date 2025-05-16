# Shunimport pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("nm.air.csv")

# Filter only for Nitrogen Dioxide (NO2) data
no2_df = df[df["Name"] == "Nitrogen dioxide (NO2)"].copy()

# Convert Start_Date to datetime
no2_df["Start_Date"] = pd.to_datetime(no2_df["Start_Date"])
no2_df["Year"] = no2_df["Start_Date"].dt.year
no2_df["Month"] = no2_df["Start_Date"].dt.month

# Encode the location (Geo Place Name)
le = LabelEncoder()
no2_df["Location_Code"] = le.fit_transform(no2_df["Geo Place Name"])

# Define features and target
X = no2_df[["Year", "Month", "Location_Code"]]
y = no2_df["Data Value"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual NO2 Levels")
plt.ylabel("Predicted NO2 Levels")
plt.title("Actual vs Predicted NO2 Levels")
plt.grid(True)
plt.show()
