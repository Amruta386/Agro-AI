import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# =========================
# 1. LOAD DATASET
# =========================
data = pd.read_csv("crop_data.csv")

print("Dataset loaded:", data.shape)

# =========================
# 2. SPLIT FEATURES & LABEL
# =========================
X = data.drop("label", axis=1)
y = data["label"]

# =========================
# 3. TRAIN / TEST SPLIT (IMPORTANT)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# 4. TRAIN MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 5. TEST MODEL ACCURACY
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# =========================
# 6. SAVE MODEL
# =========================
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully!")