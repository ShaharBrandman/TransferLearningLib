'''
exportCustomModel.py
© Author: ShaharBrandman (2024)
'''
import os
import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

#find the number of classes of the training dataset
def findNumberOfClasses(datasetPath) -> int:
    dataset = os.listdir(datasetPath)
    
    foundClasses = []

    for d in dataset:
        d = d.split('-')
        if d[0]:
            if d[0] not in foundClasses:
                foundClasses.append(d[0])

    return len(foundClasses)

def exportModel(preTrainedModelPath: str = None, datasetPath: str = 'data/images') -> None:

    if preTrainedModelPath:
        base_model = tf.saved_model.load(preTrainedModelPath)
    else:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)

    output_class = layers.Dense(
        findNumberOfClasses(datasetPath),
        activation='softmax',
        name='output_class'
    )(x)

    output_bbox = layers.Dense(4, name='output_bbox')(x)

    model = models.Model(inputs=base_model.input, outputs=[output_class, output_bbox])

    model.compile(optimizer='adam',
                loss={'output_class': 'sparse_categorical_crossentropy', 'output_bbox': 'mse'},
                metrics={'output_class': 'accuracy', 'output_bbox': 'mae'})

    model.summary()

    model.save('CustomMobileNetV2')

def argsMain() -> None:
    parser = argparse.ArgumentParser(description='Export a custom object recognition model for localization and classification')
    
    parser.add_argument(
        '--preTrainedModel',
        type=str,
        help='Path to the pre trained model file'
    )

    parser.add_argument(
        '--datasetPath',
        type=str,
        help='Path to the dataset folder'
    )

    args = parser.parse_args()

    exportModel(
        args.preTrainedModel,
        args.datasetPath
    )

def main() -> None:
    exportModel()

if __name__ == '__main__':
    argsMain()