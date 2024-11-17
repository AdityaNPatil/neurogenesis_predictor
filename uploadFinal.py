import os
import random
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from PIL import Image
import tensorflow as tf
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# Import the implications and treatments from the external file
from implications_treatments import implications_and_treatments

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load your trained model
model = tf.keras.models.load_model("./neurogenesis_model_custom.h5")

# Define the class names
class_names = [
    "Medium Neurogenesis (25-50%)",
    "Low Neurogenesis (0-25%)",
    "Very High Neurogenesis (75-90%)",
    "High Neurogenesis (50-75%)",
]

# User Management - Dummy users for demo purposes
users = {
    "admin": {"password": "admin"}
}

# Define the User class for Flask-Login
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Load the user for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Image preprocessing
def prepare_image(image):
    image = image.resize((128, 128))
    image = image.convert("RGB")
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            flash("Logged in successfully!", "success")
            return redirect(url_for('upload_file'))
        else:
            flash("Invalid credentials. Please try again.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
@login_required  # Only authenticated users can access this route
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            img = Image.open(file.stream)
            img = prepare_image(img)  # prepare_image is a preprocessing function
            predictions = model.predict(img)
            predicted_class = class_names[np.argmax(predictions)]
            
            # Get a random implication and treatment from the list for the predicted class
            selected_implication_and_treatment = random.choice(implications_and_treatments[predicted_class])
            implication = selected_implication_and_treatment["implications"]
            treatment = selected_implication_and_treatment["treatments"]
            
            return render_template("result.html", 
                                   predicted_class=predicted_class,
                                   implication=implication, 
                                   treatment=treatment)
    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)
