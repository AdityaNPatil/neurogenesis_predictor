import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = tf.keras.models.load_model('neurogenesis_modelTest.h5')

# Generate test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'Alzheimer_s Dataset/test',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate model
results = model.evaluate(test_generator)
print(f"Test Loss: {results[0]}")
print(f"Test Accuracy: {results[1]}")

# Get true labels and predictions
true_labels = test_generator.classes
predictions = model.predict(test_generator)
predicted_classes = predictions.argmax(axis=1)

# Plot confusion matrix
cm = confusion_matrix(true_labels, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Calculate and print classification report
report = classification_report(true_labels, predicted_classes, target_names=test_generator.class_indices.keys())
print("Classification Report:\n", report)

# Print precision, recall, and F1-score
precision = precision_score(true_labels, predicted_classes, average='weighted')
recall = recall_score(true_labels, predicted_classes, average='weighted')
f1 = f1_score(true_labels, predicted_classes, average='weighted')

print(f"Weighted Precision: {precision}")
print(f"Weighted Recall: {recall}")
print(f"Weighted F1-Score: {f1}")

# Print accuracy score for confirmation
print(f"Accuracy Score: {accuracy_score(true_labels, predicted_classes)}")
