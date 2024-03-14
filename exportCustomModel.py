import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Load MobileNetV2 pre-trained on ImageNet data without the top (classification) layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for classification and localization
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)

num_classes = 2

# For classification
output_class = layers.Dense(num_classes, activation='softmax', name='output_class')(x)

# For localization (four coordinates)
output_bbox = layers.Dense(4, name='output_bbox')(x)

# Combine both outputs
model = models.Model(inputs=base_model.input, outputs=[output_class, output_bbox])

# Compile the model (you may need to adjust the loss functions and metrics based on your task)
model.compile(optimizer='adam',
              loss={'output_class': 'sparse_categorical_crossentropy', 'output_bbox': 'mse'},
              metrics={'output_class': 'accuracy', 'output_bbox': 'mae'})

model.summary()

model.save('CustomMobileNetV2')