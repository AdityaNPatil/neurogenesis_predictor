import os
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = tf.keras.models.load_model("neurogenesis_model.h5")


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.reshape((1, 128, 128, 3))  # Reshape for the model
    return img_array


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            img = preprocess_image(file_path)
            prediction = model.predict(img)
            neurogenesis_rate = prediction[0]
            return f"Predicted Neurogenesis Rate: {neurogenesis_rate}"
    return """
    <!doctype html>
    <title>Upload MRI Scan</title>
    <h1>Upload an MRI scan to predict Neurogenesis rate</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    """


if __name__ == "__main__":
    app.run(debug=True)
