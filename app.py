import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import json
import os
import pickle
import requests

app = Flask(__name__)

FILE = "users.json"
CROPS_FILE = "crops.json"
ORDERS_FILE = "orders.json"
CART_FILE = "cart.json"   # ✅ NEW

# 🔥 Load ML model
price_model = pickle.load(open("price_model.pkl", "rb"))

# ✅ OpenWeather API Key
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


# ---------------- CART STORAGE (NEW) ---------------- #

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

    return render_template("crop_result.html", crop=crop)


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
                           rain=rain, n=n, p=p, k=k, ph=ph,
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


# ---------------- E-COMMERCE ---------------- #

@app.route("/add_crop", methods=["GET", "POST"])
def add_crop():
    if request.method == "POST":
        crops = load_crops()

        new_crop = {
            "name": request.form["name"],
            "price": request.form["price"],
            "quantity": request.form["quantity"]
        }

        crops.append(new_crop)
        save_crops(crops)

        return redirect("/farmer_dashboard")

    return render_template("add_crop.html")


@app.route("/farmer_dashboard")
def farmer_dashboard():
    crops = load_crops()
    return render_template("farmer_dashboard.html", crops=crops)


@app.route("/update_crop/<int:index>", methods=["POST"])
def update_crop(index):
    crops = load_crops()
    crops[index]["price"] = request.form["price"]
    crops[index]["quantity"] = request.form["quantity"]
    save_crops(crops)
    return redirect("/farmer_dashboard")


@app.route("/delete_crop/<int:index>")
def delete_crop(index):
    crops = load_crops()
    crops.pop(index)
    save_crops(crops)
    return redirect("/farmer_dashboard")


@app.route("/crops")
def view_crops():
    crops = load_crops()
    return render_template("view_crop.html", crops=crops)


# ---------------- CART SYSTEM ---------------- #

@app.route("/add_to_cart/<int:index>", methods=["POST"])
def add_to_cart(index):
    crops = load_crops()
    cart = load_cart()

    quantity = int(request.form["quantity"])
    crop = crops[index]

    if quantity > int(crop["quantity"]):
        return "Not enough stock!"

    cart_item = {
        "name": crop["name"],
        "price": crop["price"],
        "quantity": quantity
    }

    cart.append(cart_item)
    save_cart(cart)

    return redirect("/cart")


@app.route("/cart")
def view_cart():
    cart = load_cart()
    return render_template("cart.html", cart=cart)


@app.route("/place_order_from_cart")
def place_order_from_cart():
    cart = load_cart()
    orders = load_orders()
    crops = load_crops()

    total_price = 0

    for item in cart:
        total_price += int(item["price"]) * int(item["quantity"])

        # Reduce stock
        for crop in crops:
            if crop["name"] == item["name"]:
                crop["quantity"] = str(int(crop["quantity"]) - int(item["quantity"]))

    order = {
        "items": cart,
        "total_price": total_price
    }

    orders.append(order)

    save_orders(orders)
    save_crops(crops)
    save_cart([])   # clear cart

    return redirect("/payment")


# ---------------- PAYMENT ---------------- #

@app.route("/payment")
def payment():
    return render_template("payment.html")


# ---------------- LOGOUT ---------------- #

@app.route("/logout")
def logout():
    return redirect(url_for("login"))


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)