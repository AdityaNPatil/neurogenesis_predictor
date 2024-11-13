import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from PIL import Image
import tensorflow as tf
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load your trained model
model = tf.keras.models.load_model("./neurogenesis_model_custom.h5")

# Define the class names and detailed implications and treatments
class_names = [
    "Medium Neurogenesis (25-50%)",
    "Low Neurogenesis (0-25%)",
    "Very High Neurogenesis (75-90%)",
    "High Neurogenesis (50-75%)",
]

implications_and_treatments = {
    "Medium Neurogenesis (25-50%)": {
        "implications": """
            <strong>Implications of Medium Neurogenesis:</strong><br>
            • Moderate neurogenesis, sufficient for standard brain functions but with potential for enhancement.<br>
            • May support everyday memory and emotional resilience but could be improved for optimal brain plasticity.<br>
            • This level often results from balanced lifestyle habits but may benefit from specific enhancements.<br>
        """,
        "treatments": """
            <strong>Recommended Suggestions:</strong><br>
            • <strong>Exercise:</strong> Engage in moderate aerobic exercises (e.g., walking, swimming) to stimulate neurogenesis.<br>
            • <strong>Nutrition:</strong> Eat a nutrient-rich diet with antioxidants, omega-3s, and vitamins (e.g., B12, D) to promote neurogenesis.<br>
            • <strong>Mindfulness Practices:</strong> Incorporate meditation, yoga, or other stress-relieving practices.<br>
            • <strong>Sleep:</strong> Ensure 7-8 hours of restful sleep each night to allow for neural repair and regeneration.<br>
        """,
    },
    "Low Neurogenesis (0-25%)": {
        "implications": """
            <strong>Implications of Low Neurogenesis:</strong><br>
            • Low neurogenesis could indicate limited cognitive resilience, potentially impacting memory and learning ability.<br>
            • This level may lead to mood challenges, increased anxiety, or a reduced ability to adapt to stress.<br>
            • It may be a signal of lifestyle factors, age-related changes, or other health conditions that need addressing.<br>
        """,
        "treatments": """
            <strong>Recommended Suggestions:</strong><br>
            • <strong>High-Intensity Interval Training (HIIT):</strong> Incorporate HIIT exercises to strongly stimulate neurogenesis.<br>
            • <strong>Supplementation:</strong> Consider omega-3 fatty acids, magnesium, and possibly curcumin for neurogenic support.<br>
            • <strong>Cognitive Therapy:</strong> Engage in cognitive therapy or mentally stimulating activities (e.g., puzzles, learning a new skill).<br>
            • <strong>Medical Consultation:</strong> Consult a neurologist to explore potential medical treatments if symptoms are prominent.<br>
        """,
    },
    "Very High Neurogenesis (75-90%)": {
        "implications": """
            <strong>Implications of Very High Neurogenesis:</strong><br>
            • Strong neurogenesis level, supporting enhanced memory, learning ability, and emotional resilience.<br>
            • This level provides excellent adaptability to stress and a high capacity for brain plasticity.<br>
            • Likely results from optimal lifestyle practices and a low-stress environment that naturally promotes neurogenesis.<br>
        """,
        "treatments": """
            <strong>Recommended Maintenance:</strong><br>
            • <strong>Continue Current Lifestyle:</strong> Maintain current exercise, diet, and mental health routines to support neurogenesis.<br>
            • <strong>Mindful Challenges:</strong> Engage in mentally stimulating challenges to keep neurogenesis levels high.<br>
            • <strong>Reduce Stress:</strong> Continue stress-reducing activities and consider new forms of mindfulness (e.g., tai chi).<br>
        """,
    },
    "High Neurogenesis (50-75%)": {
        "implications": """
            <strong>Implications of High Neurogenesis:</strong><br>
            • Above-average neurogenesis level, supporting good cognitive function, adaptability, and emotional resilience.<br>
            • This level is indicative of a brain that is responsive to growth stimuli, making it generally healthy and adaptable.<br>
            • Indicates a generally healthy lifestyle with potential for even greater neurogenesis if targeted improvements are applied.<br>
        """,
        "treatments": """
            <strong>Recommended Suggestions:</strong><br>
            • <strong>Varied Exercise Routine:</strong> Incorporate a mix of aerobic and anaerobic exercises (e.g., cycling, weight training) for optimal benefits.<br>
            • <strong>Anti-inflammatory Diet:</strong> Focus on foods high in antioxidants and anti-inflammatory properties (e.g., berries, leafy greens).<br>
            • <strong>Learning Activities:</strong> Explore new and challenging mental activities (e.g., language learning) to stimulate the brain.<br>
            • <strong>Stress Management:</strong> Maintain and deepen relaxation practices, such as meditation and progressive muscle relaxation.<br>
        """,
    },
}


def prepare_image(image):
    image = image.resize((128, 128))
    image = image.convert("RGB")
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Dummy user demonstration (Replace with actual user data in production)
users = {"testuser": {"password": "testpass"}}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

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
            implication = implications_and_treatments[predicted_class]["implications"]
            treatment = implications_and_treatments[predicted_class]["treatments"]
            
            return render_template("result.html", 
                                   predicted_class=predicted_class,
                                   implication=implication, 
                                   treatment=treatment)
    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)
