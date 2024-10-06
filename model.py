import tensorflow as tf
from tensorflow.keras import layers, models

def create_neurogenesis_model(input_shape=(128, 128, 3), num_classes=4):
    """
    This model is for predicting neurogenesis rates based on MRI scans.
    Input shape is (128, 128, 3) and the output is a softmax for num_classes rates.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # 4 neurogenesis rate classes
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
