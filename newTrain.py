import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model import create_neurogenesis_model

data_dir = './Alzheimer_s Dataset/train'
batch_size = 32
image_size = (128, 128)

# Image augmentation and normalization with additional augmentations
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # fill_mode='nearest',
    validation_split=0.2
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

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)   # Early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3) # Reduce Learning Rate

# Train the model with increased epochs and callbacks for monitoring
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,  # Increased epochs for better learning
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
model.save('neurogenesis_model_custom2.h5')