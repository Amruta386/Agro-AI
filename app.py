import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import json
import os
import pickle

app = Flask(__name__)

FILE = "users.json"

# 🔥 Load ML model
price_model = pickle.load(open("price_model.pkl", "rb"))

# ---------------- USER FUNCTIONS ---------------- #

def load_users():
    if not os.path.exists(FILE):
        return {}
    with open(FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(FILE, "w") as f:
        json.dump(users, f)

# ---------------- AUTH ---------------- #

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/register_page")
def register_page():
    return render_template("register.html")

@app.route("/register", methods=["POST"])
def do_register():
    users = load_users()

    username = request.form["username"]
    password = request.form["password"]

    if username in users:
        return "User already exists!"

    users[username] = password
    save_users(users)

    return redirect(url_for("login"))

@app.route("/login_check", methods=["POST"])
def login_check():
    users = load_users()

    username = request.form["username"]
    password = request.form["password"]

    if username in users and users[username] == password:
        return redirect(url_for("dashboard"))
    else:
        return "Invalid credentials!"

# ---------------- DASHBOARD ---------------- #

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# ---------------- CROP ---------------- #

@app.route("/crop")
def crop():
    return render_template("crop.html")

@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    n = int(request.form["n"])
    p = int(request.form["p"])
    k = int(request.form["k"])
    temp = float(request.form["temp"])
    hum = float(request.form["hum"])
    ph = float(request.form["ph"])
    rain = float(request.form["rain"])

    if n > 50 and p > 40 and rain > 100:
        crop = "🌾 Rice"
    elif temp > 30:
        crop = "🌽 Maize"
    else:
        crop = "🌱 Wheat"

    return render_template("crop_result.html", crop=crop)

# ---------------- PRICE ---------------- #

@app.route("/price")
def price():
    return render_template("price.html")

@app.route("/predict_price", methods=["POST"])
def predict_price():
    crop = request.form["crop"]
    month = int(request.form["month"])
    market = request.form["market"]

    input_data = pd.DataFrame({
        "crop": [crop],
        "month": [month],
        "market": [market]
    })

    input_data = pd.get_dummies(input_data)

    model_columns = price_model.feature_names_in_
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    prediction = price_model.predict(input_data)

    trend = "📈 Increasing" if prediction[0] > 2300 else "📉 Stable/Low"

    return render_template(
        "price_result.html",
        price=round(prediction[0], 2),
        trend=trend
    )

# ---------------- LOGOUT ---------------- #

@app.route("/logout")
def logout():
    return redirect(url_for("login"))

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)