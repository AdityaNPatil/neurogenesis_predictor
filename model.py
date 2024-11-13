import tensorflow as tf
from tensorflow.keras import layers, models

def create_neurogenesis_model(input_shape=(128, 128, 3), num_classes=4):
    """
    This model is for predicting neurogenesis rates based on MRI scans.
    Input shape is (128, 128, 3) and the output is a softmax for num_classes rates.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),  # Feature extraction
        layers.MaxPooling2D((2, 2)),                                            # Spatial dimension reduction
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),                                                       # Convert to 1D Vector
        layers.Dense(128, activation='relu'),                                   # High level reasoning
        layers.Dense(num_classes, activation='softmax')  # 4 neurogenesis rate classes # Multi class classification - softmax
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Adam optimizer for dynamic learning rate , categorical cross entropy for measure the difference between the predicted probabilities and the actual labels for a multi-class classification task
    return model
