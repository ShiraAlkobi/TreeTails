import tensorflow as tf


# Define the training loop
def train_model(model, train_dataset, valid_dataset, epochs=10):
    # Specify the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)

    # Specify the loss function
    def compute_loss(model, images, groundtruth):
        loss_dict = model(images, groundtruth, training=True)
        total_loss = tf.reduce_sum(loss_dict['total_loss'])
        return total_loss

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for image, labels in train_dataset:
            with tf.GradientTape() as tape:
                total_loss = compute_loss(model, image, labels)
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print(f"Step loss: {total_loss.numpy()}")

        # Validation after each epoch
        if epoch % 1 == 0:
            validate_model(model, valid_dataset)

        print(f"Finished Epoch {epoch + 1}")


# Validation step
def validate_model(model, valid_dataset):
    total_loss = 0
    for image, labels in valid_dataset:
        loss = compute_loss(model, image, labels)
        total_loss += loss.numpy()
    print(f"Validation Loss: {total_loss / len(valid_dataset)}")


# Start training
train_model(model, train_dataset, valid_dataset, epochs=10)
model.save('trained_model/')
