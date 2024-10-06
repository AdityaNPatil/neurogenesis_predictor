import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Reduce Dropout: Decrease the dropout rate to avoid underfitting.
# Increase Model Complexity: Add more convolutional layers to increase the model's capacity.
# Tune the Learning Rate: Use a smaller learning rate to prevent the model from missing optimal weights.
# Adjust Data Augmentation: Make data augmentation less aggressive to allow the model to learn core patterns from the data.

# Data Augmentation (Less aggressive)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,        # Reduced rotation
    width_shift_range=0.1,    # Reduced width shift
    height_shift_range=0.1,   # Reduced height shift
    shear_range=0.1,          # Reduced shear
    zoom_range=0.1,           # Reduced zoom
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'Alzheimer_s Dataset/train',
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'Alzheimer_s Dataset/test',
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

# Updated Model Architecture (Increased complexity)
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(4, activation='softmax')  # 4 classes for neurogenesis rates
])

# Compile Model (Lower learning rate)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Reduced learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Learning Rate Adjustment and Early Stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=[reduce_lr, early_stopping]  # Added early stopping to avoid overfitting
)

model.save('neurogenesis_model_v2.h5')
