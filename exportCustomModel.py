import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def findNumberOfClasses() -> int:
    pass

def main() -> None:
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)

    num_classes = findNumberOfClasses()

    output_class = layers.Dense(num_classes, activation='softmax', name='output_class')(x)

    output_bbox = layers.Dense(4, name='output_bbox')(x)

    model = models.Model(inputs=base_model.input, outputs=[output_class, output_bbox])

    model.compile(optimizer='adam',
                loss={'output_class': 'sparse_categorical_crossentropy', 'output_bbox': 'mse'},
                metrics={'output_class': 'accuracy', 'output_bbox': 'mae'})

    model.summary()

    model.save('CustomMobileNetV2')

if __name__ == '__main__':
    main()