import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

def visualize_model_architecture(model_path):
    """
    Print the summary of the neurogenesis model instead of visualizing the architecture.
    """
    model = tf.keras.models.load_model(model_path)
    model.summary()  # Print the summary instead of plotting

def predict_image(model_path, img_path):
    """
    Load an image and predict the neurogenesis rate using the trained model.
    """
    model = tf.keras.models.load_model(model_path)
    
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=-1)
    
    # Neurogenesis rate classes 
    # (Mild dementia = Medium neuro | moderate dementia = low neuro | non dem = very high neuro | very mild dem = high neuro)
    classes = {0: 'Medium Neurogenesis', 1: 'Low Neurogenesis', 2: 'Very High Neurogenesis', 3: 'High Neurogenesis'}
    return classes[predicted_class[0]]

# Use the functions
visualize_model_architecture('neurogenesis_model.h5')
print(predict_image('neurogenesis_model.h5', 'C:/IMPORTANT STUFF/VIT/Sem 7/Project Sem 7/Project sample code/New Codes 3/Alzheimer_s Dataset/train/NonDemented/nonDem0.jpg'))
