import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
data = pd.read_csv("price_data.csv")

# Convert text → numbers
data = pd.get_dummies(data)

# Split
X = data.drop("price", axis=1)
y = data["price"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save model
pickle.dump(model, open("price_model.pkl", "wb"))

print("Price model trained!")