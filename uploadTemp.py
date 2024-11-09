import os
import numpy as np
from flask import Flask, request, render_template_string, redirect, url_for
from PIL import Image
import tensorflow as tf
import time

app = Flask(__name__)

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
            <strong>Recommended Treatments:</strong><br>
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
            <strong>Recommended Treatments:</strong><br>
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
            <strong>Recommended Treatments:</strong><br>
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


@app.route("/", methods=["GET"])
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Home - Neurogenesis Predictor</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <style>
            body {
                background-color: #f8f9fa;
                font-family: 'Montserrat', sans-serif;
            }

            .hero-section {
                background-image: url('https://source.unsplash.com/1600x900/?brain,neuroscience');
                background-size: cover;
                background-position: center;
                color: #fff;
                text-align: center;
                padding: 100px 0;
            }

            .hero-section h1 {
                font-size: 4rem;
            }

            .section {
                padding: 50px 0;
            }

            .footer {
                background-color: #343a40;
                color: #fff;
                text-align: center;
                padding: 20px;
            }

            .btn {
                background-color: #007bff;
                color: #fff;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                text-decoration: none;
            }

            .btn:hover {
                background-color: #0062cc;
            }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container">
                <a class="navbar-brand" href="/">Neurogenesis Predictor</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav ms-auto">
                        <li class="nav-item">
                            <a class="nav-link active" aria-current="page" href="/">Home</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/predict">Prediction</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>

        <section class="hero-section">
            <div class="container" style="color:black;">
                <h1>Unlock Your Brain's Potential</h1>
                <p>Discover your neurogenesis level and take steps to improve your cognitive health.</p>
                <a href="/predict" class="btn btn-lg">Predict Now</a>
            </div>
        </section>

        <section class="section">
            <div class="container">
                <h2>What is Neurogenesis?</h2>
                <p>Neurogenesis is the process of generating new neurons in the brain. It's crucial for cognitive health, learning, and emotional resilience. Higher neurogenesis levels support better memory, mood, and adaptability to stress.</p>
            </div>
        </section>

        <section class="section">
            <div class="container">
                <h2>How Does This Predictor Help?</h2>
                <p>This tool offers a quick and easy way to assess neurogenesis levels and provides personalized suggestions on lifestyle changes to improve brain health. Whether for general health or recovery support, knowing your neurogenesis level can be beneficial.</p>
            </div>
        </section>

        <footer class="footer">
            &copy; 2024 Neurogenesis Predictor | All Rights Reserved
        </footer>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """


@app.route("/predict", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            img = Image.open(file.stream)
            img = prepare_image(img)
            time.sleep(2)  # Optional delay for processing simulation
            predictions = model.predict(img)
            predicted_class = class_names[np.argmax(predictions)]
            implication = implications_and_treatments[predicted_class]["implications"]
            treatment = implications_and_treatments[predicted_class]["treatments"]
            return f"""
            <div style="background-color: black; padding: 10px; color: white;">
                <a href="/" style="color: white; margin: 0 10px; text-decoration: none; font-weight: bold;">Home</a>
                <a href="/predict" style="color: white; margin: 0 10px; text-decoration: none; font-weight: bold;">Prediction Page</a>
            </div>
            <div style="display: flex; flex-direction: column; align-items: center; margin-top: 100px; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);">
                <h1 style="color: #007bff; text-align: center; font-size: 32px; margin-bottom: 30px;">Neurogenesis Prediction Result</h1>
                <div style="text-align: center; font-size: 20px; color: #28a745; margin-bottom: 40px;">
                    Predicted Neurogenesis Level: <strong style="font-size:25px; color:red">{predicted_class}</strong>
                </div>
                <div style="display:flex; flex-direction:column; gap:5px; justify-content:center; align-items:center; margin-bottom: 20px;">
                    <h2 style="color: #333; font-size: 24px; margin-bottom: 10px;">Implications</h2>
                    <p style="font-size: 16px;">{implication}</p>
                </div>
                <div style="display:flex; flex-direction:column; gap:5px; justify-content:center; align-items:center;">
                    <h2 style="color: #333; font-size: 24px; margin-bottom: 10px;">Suggested Treatment</h2>
                    <p style="font-size: 16px;">{treatment}</p>
                </div>
            </div>
            <div style="background-color: #333; color: white; padding: 10px; position: fixed; bottom: 0; width: 100%; text-align: center;">© 2024 Neurogenesis Predictor | All Rights Reserved</div>
            """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Neurogenesis Predictor</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <style>
            body {
                background-color: #f8f9fa;
                font-family: 'Montserrat', sans-serif;
            }

            .container {
                max-width: 600px;
                margin: 0 auto;
                padding: 50px;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            }

            .form-group {
                margin-bottom: 20px;
            }

            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
            }

            .footer {
                background-color: #343a40;
                color: #fff;
                text-align: center;
                padding: 20px;
                position:fixed;
                bottom:0;
                width:100vw;
            }

            .btn {
                background-color: #007bff;
                color: #fff;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                text-decoration: none;
            }

            .btn:hover {
                background-color: #0062cc;
            }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container">
                <a class="navbar-brand" href="/">Neurogenesis Predictor</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav" style="height:fit-content">
                    <ul class="navbar-nav ms-auto">
                        <li class="nav-item">
                            <a class="nav-link" href="/">Home</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link active" aria-current="page" href="#">Prediction</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>

        <section class="container">
            <h1>Neurogenesis Predictor</h1>
            <form method="post" enctype="multipart/form-data" class="form-group" onsubmit="showLoading()">
                <div class="mb-3">
                    <label for="fileInput" class="form-label">Upload MRI Image</label>
                    <input type="file" class="form-control" id="fileInput" name="file" accept="image/*" required>
                </div>
                <button type="submit" class="btn btn-primary">Predict Neurogenesis Level</button>
            </form>
            <div class="loading">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p>Processing your image...</p>
            </div>
        </section>

        <footer class="footer">
            &copy; 2024 Neurogenesis Predictor | All Rights Reserved
        </footer>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            function showLoading() {
                document.querySelector('.loading').style.display = 'block';
            }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(debug=True)
