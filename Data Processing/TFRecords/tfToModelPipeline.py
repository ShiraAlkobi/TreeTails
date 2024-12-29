import tensorflow as tf
from model1 import parse_tfrecord_fn, model
# Load the trained model


# Load the test dataset
# Assume `test_data` is a tf.data.Dataset object or similar
# Replace with your test dataset code
test_data = tf.data.TFRecordDataset('./Data Processing/TFRecords/test.tfrecord')  # Update with your test TFRecord file
test_data = test_data.map(parse_tfrecord_fn)  # Apply the same parsing function used for training
test_data = test_data.batch(32)  # Use the same batch size as during training

# Evaluate the model
results = model.evaluate(test_data)

# Print evaluation results
print("Test Evaluation Results:")
for metric, value in zip(model.metrics_names, results):
    print(f"{metric}: {value:.4f}")
