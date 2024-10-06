import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_neurogenesis_model

data_dir = './Alzheimer_s Dataset/train'  # Keeping the original dataset
batch_size = 32
image_size = (128, 128)

# Image augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, 
    validation_split=0.1,           # Validation split (90/30 - 90 train , 10 - test)
    # color_mode = 'grayscale'
)

# Load the training and validation data
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Create the model
model = create_neurogenesis_model()

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Save the model
model.save('neurogenesis_model.h5')
