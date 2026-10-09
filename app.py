import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
import json
import os
import pickle
import requests
import webbrowser

from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
app.secret_key = "agro_ai_secret_key"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

FILE = "users.json"
CROPS_FILE = "crops.json"
ORDERS_FILE = "orders.json"
CART_FILE = "cart.json"

# Load ML models
price_model = pickle.load(open("price_model.pkl", "rb"))

fertilizers = {
    "🌾 Rice": "Urea + DAP fertilizer",
    "🌽 Maize": "NPK 20-20-20 fertilizer",
    "🌱 Wheat": "Nitrogen-rich fertilizer"
}

# Disease Model
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

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# OpenWeather API
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


# ---------------- ORDER STORAGE ---------------- #

def load_orders():
    if not os.path.exists(ORDERS_FILE):
        return []

    with open(ORDERS_FILE, "r") as f:
        return json.load(f)


def save_orders(orders):
    file_path = os.path.abspath(ORDERS_FILE)

    print("====================================")
    print("SAVING ORDERS TO:")
    print(file_path)
    print("====================================")

    with open(file_path, "w") as f:
        json.dump(orders, f, indent=4)


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

        session["username"] = username

        return redirect(url_for("dashboard"))

    return "Invalid credentials!"


# ---------------- ADMIN LOGIN ---------------- #

@app.route("/admin_login")
def admin_login():

    return render_template("admin_login.html")


@app.route("/admin_login_check", methods=["POST"])
def admin_login_check():

    username = request.form["username"]
    password = request.form["password"]

    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:

        session["is_admin"] = True

        return redirect(url_for("farmer_dashboard"))

    return render_template(
        "admin_login.html",
        error="Invalid admin username or password!"
    )


@app.route("/admin_logout")
def admin_logout():

    session.pop("is_admin", None)

    return redirect(url_for("dashboard"))


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
        return render_template(
            "weather_crop.html",
            error="Invalid city!"
        )

    temp, hum = weather

    if n > 50 and p > 40 and rain > 100:
        crop = "🌾 Rice"

    elif temp > 30:
        crop = "🌽 Maize"

    else:
        crop = "🌱 Wheat"

    return render_template(
        "weather_crop_result.html",
        city=city,
        temp=temp,
        hum=hum,
        crop=crop
    )


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

    input_data = input_data.reindex(
        columns=price_model.feature_names_in_,
        fill_value=0
    )

    prediction = price_model.predict(input_data)

    trend = "📈 Increasing" if prediction[0] > 2300 else "📉 Stable/Low"

    return render_template(
        "price_result.html",
        price=round(prediction[0], 2),
        trend=trend
    )


# ---------------- PLANT DISEASE DETECTION ---------------- #

@app.route("/predict_disease", methods=["POST"])
def predict_disease():

    file = request.files["image"]

    if file.filename == "":
        return "No file selected!"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    img = Image.open(filepath).convert("RGB")
    img = img.resize((128, 128))

    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = disease_model.predict(img_array)

    class_index = np.argmax(prediction[0])

    if class_index >= len(disease_classes):
        return "Model output mismatch error"

    result = disease_classes[class_index]

    display_name = result.replace("_", " ")

    confidence = float(np.max(prediction[0]) * 100)
    confidence = round(confidence, 2)

    treatment = treatments.get(
        result,
        "No suggestion available"
    )

    return render_template(
        "result.html",
        result=result,
        display_name=display_name,
        image=filepath,
        confidence=confidence,
        treatment=treatment
    )


# ==========================================================
#                    E-COMMERCE MODULE
# ==========================================================


# ---------------- ADD CROP ---------------- #

@app.route("/add_crop", methods=["GET", "POST"])
def add_crop():

    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))

    if request.method == "POST":

        crops = load_crops()

        new_crop = {
            "name": request.form["name"],
            "price": request.form["price"],
            "quantity": request.form["quantity"]
        }

        crops.append(new_crop)

        save_crops(crops)

        return redirect(url_for("farmer_dashboard"))

    return render_template("add_crop.html")


# ---------------- FARMER DASHBOARD ---------------- #

@app.route("/farmer_dashboard")
def farmer_dashboard():

    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))

    crops = load_crops()

    return render_template(
        "farmer_dashboard.html",
        crops=crops
    )


# ---------------- UPDATE CROP ---------------- #

@app.route("/update_crop/<int:index>", methods=["POST"])
def update_crop(index):

    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))

    crops = load_crops()

    if index >= len(crops):
        return "Crop not found!"

    crops[index]["price"] = request.form["price"]
    crops[index]["quantity"] = request.form["quantity"]

    save_crops(crops)

    return redirect(url_for("farmer_dashboard"))


# ---------------- DELETE CROP ---------------- #

@app.route("/delete_crop/<int:index>")
def delete_crop(index):

    if not session.get("is_admin"):
        return redirect(url_for("admin_login"))

    crops = load_crops()

    if index >= len(crops):
        return "Crop not found!"

    crops.pop(index)

    save_crops(crops)

    return redirect(url_for("farmer_dashboard"))


# ---------------- MARKETPLACE HOME ---------------- #

@app.route("/marketplace")
def marketplace():

    crops = load_crops()

    return render_template(
        "marketplace.html",
        crops=crops
    )


@app.route("/categories")
def categories():

    crops = load_crops()

    selected_category = request.args.get(
        "category",
        "All Crops"
    )

    category_map = {
        "Millets": [
            "ragi",
            "millet",
            "foxtail millet",
            "little millet",
            "pearl millet",
            "bajra",
            "jowar"
        ],
        "Cereals": [
            "rice",
            "wheat",
            "maize",
            "corn",
            "barley",
            "oats"
        ],
        "Pulses": [
            "dal",
            "pulse",
            "pulses",
            "toor",
            "tur",
            "moong",
            "green gram",
            "urad",
            "black gram",
            "chana",
            "gram",
            "lentil"
        ],
        "Vegetables": [
            "tomato",
            "potato",
            "onion",
            "carrot",
            "beans",
            "brinjal",
            "cabbage",
            "cauliflower",
            "spinach",
            "peas"
        ],
        "Fruits": [
            "apple",
            "banana",
            "mango",
            "orange",
            "grapes",
            "papaya",
            "guava",
            "watermelon",
            "pineapple"
        ]
    }

    if selected_category != "All Crops":

        filtered_crops = []

        for crop in crops:

            crop_name = crop["name"].lower()

            for keyword in category_map.get(
                selected_category,
                []
            ):

                if keyword in crop_name:

                    filtered_crops.append(crop)

                    break

    else:

        filtered_crops = crops

    return render_template(
        "categories.html",
        crops=filtered_crops,
        selected_category=selected_category
    )


# ---------------- VIEW MARKETPLACE ---------------- #

@app.route("/crops")
def view_crops():

    crops = load_crops()

    return render_template(
        "view_crop.html",
        crops=crops
    )


# ---------------- ADD TO CART ---------------- #

@app.route("/add_to_cart/<int:index>", methods=["POST"])
def add_to_cart(index):

    crops = load_crops()
    cart = load_cart()

    if index >= len(crops):
        return "Crop not found!"

    crop = crops[index]

    quantity = int(request.form["quantity"])

    available_quantity = int(
        str(crop["quantity"]).split()[0]
    )

    if quantity <= 0:
        return "Quantity must be greater than zero!"

    if quantity > available_quantity:
        return "Not enough stock!"

    cart_item = {
        "name": crop["name"],
        "price": crop["price"],
        "quantity": quantity
    }

    cart.append(cart_item)

    save_cart(cart)

    return redirect(url_for("view_cart"))


# ---------------- INCREASE CART QUANTITY ---------------- #

@app.route("/increase_cart/<int:index>")
def increase_cart(index):

    cart = load_cart()

    if index >= len(cart):
        return "Cart item not found!"

    item = cart[index]

    crops = load_crops()

    available_quantity = 0

    for crop in crops:

        if (
            crop["name"] == item["name"]
            and int(crop["price"]) == int(item["price"])
        ):

            available_quantity = int(
                str(crop["quantity"]).split()[0]
            )

            break

    current_quantity = int(item["quantity"])

    if current_quantity >= available_quantity:
        return "Not enough stock!"

    cart[index]["quantity"] = current_quantity + 1

    save_cart(cart)

    return redirect(url_for("view_cart"))


# ---------------- DECREASE CART QUANTITY ---------------- #

@app.route("/decrease_cart/<int:index>")
def decrease_cart(index):

    cart = load_cart()

    if index >= len(cart):
        return "Cart item not found!"

    current_quantity = int(
        cart[index]["quantity"]
    )

    if current_quantity > 1:

        cart[index]["quantity"] = current_quantity - 1

        save_cart(cart)

    return redirect(url_for("view_cart"))


# ---------------- REMOVE FROM CART ---------------- #

@app.route("/remove_from_cart/<int:index>")
def remove_from_cart(index):

    cart = load_cart()

    if index >= len(cart):
        return "Cart item not found!"

    cart.pop(index)

    save_cart(cart)

    return redirect(url_for("view_cart"))


# ---------------- VIEW CART ---------------- #

@app.route("/cart")
def view_cart():

    cart = load_cart()

    total_price = 0

    for item in cart:

        total_price += (
            int(item["price"]) *
            int(item["quantity"])
        )

    return render_template(
        "cart.html",
        cart=cart,
        total_price=total_price
    )


# ---------------- PLACE ORDER FROM CART ---------------- #

@app.route("/place_order_from_cart")
def place_order_from_cart():

    cart = load_cart()

    if not cart:

        return render_template(
            "order_summary.html",
            cart=[],
            total_price=0
        )

    crops = load_crops()

    total_price = 0

    for item in cart:

        quantity = int(item["quantity"])

        total_price += (
            int(item["price"]) *
            quantity
        )

        for crop in crops:

            if crop["name"] == item["name"]:

                available_quantity = int(
                    str(crop["quantity"]).split()[0]
                )

                if quantity > available_quantity:

                    return (
                        "Not enough stock for "
                        + crop["name"]
                        + ". Available stock: "
                        + str(available_quantity)
                    )

    return render_template(
        "order_summary.html",
        cart=cart,
        total_price=total_price
    )


# ---------------- PAYMENT ---------------- #

@app.route("/payment")
def payment():

    cart = load_cart()

    total_price = 0

    for item in cart:

        total_price += (
            int(item["price"]) *
            int(item["quantity"])
        )

    return render_template(
        "payment.html",
        total_price=total_price
    )


@app.route("/payment_success", methods=["POST"])
def payment_success():

    payment_method = request.form.get(
        "payment_method"
    )

    cart = load_cart()

    if not cart:
        return "Your cart is empty."

    crops = load_crops()
    orders = load_orders()

    total_price = 0

    for item in cart:

        quantity = int(item["quantity"])
        price = int(item["price"])

        total_price += price * quantity

        crop_found = False

        for crop in crops:

            if (
                crop["name"] == item["name"]
                and int(crop["price"]) == price
            ):

                available_quantity = int(
                    str(crop["quantity"]).split()[0]
                )

                if quantity > available_quantity:

                    return (
                        "Not enough stock for "
                        + crop["name"]
                        + ". Available stock: "
                        + str(available_quantity)
                    )

                crop_found = True

                break

        if not crop_found:
            return "Crop not found!"

    if orders:

        order_id = max(
            int(order.get("order_id", 0))
            for order in orders
        ) + 1

    else:

        order_id = 1

    username = session.get("username")

    order = {
        "order_id": order_id,
        "username": username,
        "items": cart,
        "total_price": total_price,
        "payment_method": payment_method
    }

    for item in cart:

        quantity = int(item["quantity"])
        price = int(item["price"])

        for crop in crops:

            if (
                crop["name"] == item["name"]
                and int(crop["price"]) == price
            ):

                original_quantity = str(
                    crop["quantity"]
                )

                available_quantity = int(
                    original_quantity.split()[0]
                )

                remaining_quantity = (
                    available_quantity - quantity
                )

                parts = original_quantity.split()

                if len(parts) > 1:

                    crop["quantity"] = (
                        str(remaining_quantity)
                        + " "
                        + " ".join(parts[1:])
                    )

                else:

                    crop["quantity"] = str(
                        remaining_quantity
                    )

                break

    save_crops(crops)

    orders.append(order)

    save_orders(orders)

    save_cart([])

    return render_template(
        "payment_success.html",
        order=order
    )


# ---------------- ACCOUNT ---------------- #

@app.route("/account")
def account():

    username = session.get("username")

    if not username:
        return redirect(url_for("login"))

    orders = load_orders()

    customer_orders = []

    for order in orders:

        if order.get("username") == username:

            customer_orders.append(order)

    return render_template(
        "account.html",
        username=username,
        orders=customer_orders
    )


# ---------------- MY ORDERS ---------------- #

@app.route("/my_orders")
def my_orders():

    orders = load_orders()

    return render_template(
        "my_orders.html",
        orders=orders
    )


# ---------------- LOGOUT ---------------- #

@app.route("/logout")
def logout():

    return redirect(url_for("login"))


# ---------------- DISEASE PAGE ---------------- #

@app.route("/disease")
def disease():

    return render_template("upload.html")


# ---------------- RUN ---------------- #

if __name__ == "__main__":

    webbrowser.open("http://127.0.0.1:5000")

    app.run(debug=True)