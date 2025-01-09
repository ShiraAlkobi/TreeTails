import tensorflow as tf
from model1 import parse_tfrecord_fn
# Load the saved model
model = tf.keras.models.load_model('saved_model.keras')

# Prepare the test dataset
def load_tfrecord(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file, buffer_size=8 * 1024 * 1024)  # 8 MB
    dataset = dataset.map(parse_tfrecord_fn)
    return dataset

test_dataset = load_tfrecord(
    '../TFRecords/test.tfrecord').batch(16).prefetch(tf.data.experimental.AUTOTUNE)

# Evaluate the model on the test data
results = model.evaluate(test_dataset)

# Print the evaluation results
print("Evaluation Results:")
print(f"Bounding Box Loss (MSE): {results[1]}")  # Bounding box loss
print(f"Bounding Box Metric (MAE): {results[3]}")  # Bounding box metric
print(f"Class Label Loss (Sparse Categorical Crossentropy): {results[2]}")  # Class label loss
print(f"Class Label Metric (Accuracy): {results[4]}")  # Class label metric

# Optionally, make predictions on the test dataset
for images, targets in test_dataset.take(1):  # Take 1 batch
    predictions = model.predict(images)

    print("Predicted Bounding Boxes:")
    print(predictions[0])  # Bounding box predictions

    print("Predicted Class Labels:")
    print(predictions[1])  # Class label predictions

    print("True Bounding Boxes:")
    print(targets['bbox_output_reshape'])

    print("True Class Labels:")
    print(targets['class_output_reshape'])
