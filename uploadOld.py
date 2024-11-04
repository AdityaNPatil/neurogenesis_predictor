import os
import numpy as np
from flask import Flask, request, render_template
from PIL import Image
import tensorflow as tf
import time

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('./neurogenesis_model_custom.h5')  # Adjust this path according to model location

# Define the class names based on your dataset
class_names = ['Medium Neurogenesis(25-50%)', 'Low Neurogenesis(0-25%)', 'Very High Neurogenesis(75-90%)', 'High Neurogenesis(50-75%)']

def prepare_image(image):
    image = image.resize((128, 128))  # Resize to match model input
    image = image.convert('RGB')        # Ensure image has 3 channels
    image = np.array(image) / 255.0     # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            img = Image.open(file.stream)
            img = prepare_image(img)
            # Simulate processing time for better UX
            time.sleep(2)  # Optional: Add delay to simulate processing
            predictions = model.predict(img)
            predicted_class = class_names[np.argmax(predictions)]
            return f'''
            <div class="result animated fadeIn" style="display:flex; justify-content:center; font-size:2rem; margin-top: 20px; color:#28a745;">
                Predicted class: {predicted_class}
            </div>
            '''
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Neurogenesis Predictor</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f4f4f9;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
                padding: 40px;
                text-align: center;
                max-width: 400px;
                width: 100%;
            }
            h1 {
                color: #333;
            }
            .form-group {
                margin-bottom: 20px;
            }
            input[type="file"] {
                display: block;
                margin: 0 auto;
                padding: 10px;
            }
            input[type="submit"] {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            input[type="submit"]:hover {
                background-color: #0056b3;
            }
            .result {
                margin-top: 20px;
                font-size: 18px;
                color: #28a745;
                opacity: 0;
                animation: fadeIn 1s forwards;
            }
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            .loading {
                display: none;
                justify-content: center;
                align-items: center;
                font-size: 1.5rem;
                color: #007bff;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0%{
                    transform: scale(0.8);
                }
                100% {
                    transform: scale(1);
                }
            }
        </style>
        <script>
            function showLoading() {
                document.querySelector('.loading').style.display = 'flex';
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Neurogenesis Predictor</h1>
            <form method="post" enctype="multipart/form-data" class="form-group" onsubmit="showLoading()">
                <input type="file" name="file" class="form-control" required>
                <input type="submit" value="Upload and Predict">
            </form>
            <div class="loading">Predicting...</div>
        </div>
    </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(debug=True)
