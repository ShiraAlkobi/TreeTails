import tensorflow as tf
from model1 import parse_tfrecord_fn
# Load the saved model
model = tf.keras.models.load_model('saved_model.keras')

# Prepare the test dataset
def load_tfrecord(tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file, buffer_size=8 * 1024 * 1024)  # 8 MB
    dataset = dataset.map(parse_tfrecord_fn)
    return dataset

test_dataset = load_tfrecord('./Data Processing/TFRecords/test.tfrecord').batch(16).prefetch(tf.data.experimental.AUTOTUNE)

# Evaluate the model on the test data
test_loss, test_metrics = model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}')
print(f'Test Metrics: {test_metrics}')

# Make predictions (optional)
for images, labels in test_dataset.take(1):  # Take 1 batch
    predictions = model.predict(images)
    print(f'Predictions: {predictions}')
    print(f'Labels: {labels}')
