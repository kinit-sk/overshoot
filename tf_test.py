import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model with SGD + Momentum
# model.compile(optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.9),
model.compile(optimizer=optimizers.AdamW(beta_1=0.0),
# model.compile(optimizer=optimizers.AdamW(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
# Train the model
history = model.fit(train_images, train_labels, epochs=50, batch_size=128,
                    validation_data=(test_images, test_labels))