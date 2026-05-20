import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import json
import os
import pickle
import requests
import webbrowser

# 🔥 NEW IMPORTS
from tensorflow.keras.models import load_model 
import numpy as np 
from PIL import Image

app = Flask(__name__)

FILE = "users.json"
CROPS_FILE = "crops.json"
ORDERS_FILE = "orders.json"
CART_FILE = "cart.json"

# 🔥 Load ML models
price_model = pickle.load(open("price_model.pkl", "rb"))

fertilizers = {
    "🌾 Rice": "Urea + DAP fertilizer",
    "🌽 Maize": "NPK 20-20-20 fertilizer",
    "🌱 Wheat": "Nitrogen-rich fertilizer"
}

# 🔥 Disease Model
disease_model = load_model("plant_disease_mobilenet.h5")

disease_classes = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus"
]
treatments = {
    "Pepper__bell___Bacterial_spot": "Apply copper-based bactericide and remove infected leaves.",
    "Pepper__bell___healthy": "Plant is healthy. No treatment needed.",

    "Potato___Early_blight": "Use fungicide and avoid overhead watering.",
    "Potato___Late_blight": "Apply chlorothalonil or mancozeb fungicide immediately.",
    "Potato___healthy": "Plant is healthy. No treatment needed.",

    "Tomato_Bacterial_spot": "Use copper sprays and disease-free seeds.",
    "Tomato_Early_blight": "Apply fungicide and prune lower infected leaves.",
    "Tomato_Late_blight": "Use fungicide and improve air circulation.",
    "Tomato_Leaf_Mold": "Reduce humidity and apply fungicide.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves and apply fungicide.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray neem oil or insecticidal soap.",
    "Tomato__Target_Spot": "Apply fungicide and avoid wet foliage.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies and remove infected plants.",
    "Tomato__Tomato_mosaic_virus": "Remove infected plants and disinfect tools.",
    "Tomato_healthy": "Plant is healthy. No treatment needed."
}

# ✅ Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ✅ OpenWeather API
API_KEY = "34c18588bd8a6c54e4d7c8bda9ce5713"


# ---------------- USER FUNCTIONS ---------------- #

def load_users():
    if not os.path.exists(FILE):
        return {}
    with open(FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(FILE, "w") as f:
        json.dump(users, f)


# ---------------- CROP STORAGE ---------------- #

def load_crops():
    if not os.path.exists(CROPS_FILE):
        return []
    with open(CROPS_FILE, "r") as f:
        return json.load(f)

def save_crops(crops):
    with open(CROPS_FILE, "w") as f:
        json.dump(crops, f)


def load_orders():
    if not os.path.exists(ORDERS_FILE):
        return []
    with open(ORDERS_FILE, "r") as f:
        return json.load(f)

def save_orders(orders):
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f)


# ---------------- CART STORAGE ---------------- #

def load_cart():
    if not os.path.exists(CART_FILE):
        return []
    with open(CART_FILE, "r") as f:
        return json.load(f)

def save_cart(cart):
    with open(CART_FILE, "w") as f:
        json.dump(cart, f)


# ---------------- WEATHER FUNCTION ---------------- #

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        return None

    return data["main"]["temp"], data["main"]["humidity"]


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


# ---------------- CROP PREDICTION ---------------- #

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

    fertilizer = fertilizers.get(crop, "General fertilizer")

    return render_template(
        "crop_result.html",
        crop=crop,
        fertilizer=fertilizer
    )

# ---------------- WEATHER CROP ---------------- #

@app.route("/weather_crop")
def weather_crop():
    return render_template("weather_crop.html")

@app.route("/predict_weather_crop", methods=["POST"])
def predict_weather_crop():
    city = request.form["city"]
    rain = float(request.form["rain"])
    n = int(request.form["n"])
    p = int(request.form["p"])
    k = int(request.form["k"])
    ph = float(request.form["ph"])

    weather = get_weather(city)

    if weather is None:
        return render_template("weather_crop.html", error="Invalid city!")

    temp, hum = weather

    if n > 50 and p > 40 and rain > 100:
        crop = "🌾 Rice"
    elif temp > 30:
        crop = "🌽 Maize"
    else:
        crop = "🌱 Wheat"

    return render_template("weather_crop_result.html",
                           city=city, temp=temp, hum=hum,
                           crop=crop)


# ---------------- PRICE PREDICTION ---------------- #

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
    input_data = input_data.reindex(columns=price_model.feature_names_in_, fill_value=0)

    prediction = price_model.predict(input_data)
    trend = "📈 Increasing" if prediction[0] > 2300 else "📉 Stable/Low"

    return render_template("price_result.html",
                           price=round(prediction[0], 2),
                           trend=trend)


# ---------------- 🌿 PLANT DISEASE DETECTION ---------------- #

@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    file = request.files["image"]

    if file.filename == "":
        return "No file selected!"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Image preprocessing
    img = Image.open(filepath).convert("RGB")
    img = img.resize((128, 128))

    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = disease_model.predict(img_array)

    class_index = np.argmax(prediction[0])

    if class_index >= len(disease_classes):
        return "Model output mismatch error"

    result = disease_classes[class_index]
    display_name = result.replace("_", " ")

    confidence = float(np.max(prediction[0]) * 100)
    confidence = round(confidence, 2)

    treatment = treatments.get(result, "No suggestion available")

    return render_template(
        "result.html",
        result=result,
        display_name=display_name,
        image=filepath,
        confidence=confidence,
        treatment=treatment
    )


# ---------------- RUN ---------------- #
@app.route("/logout")
def logout():
    return redirect(url_for("login"))

@app.route("/disease")
def disease():
    return render_template("upload.html")

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
