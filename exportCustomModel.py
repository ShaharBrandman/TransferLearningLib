'''
exportCustomModel.py
© Author: ShaharBrandman (2024)
'''
import os
import argparse
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy, MeanAbsoluteError


#find the number of classes of the training dataset
def findNumberOfClasses(datasetPath) -> int:
    dataset = os.listdir(datasetPath)
    
    foundClasses = []

    for d in dataset:
        d = d.split('-')
        if d[1]:
            if d[1] not in foundClasses:
                foundClasses.append(d[1])
    return len(foundClasses)

def exportModel(datasetPath: str = 'data/annotations/') -> None:

    base_model = EfficientNetV2S(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    
    base_model = Sequential([base_model,
                       GlobalAveragePooling2D(),
                       Dense(64, activation='relu')])

    output_class = Dense(
        128,
        activation='relu'
    )(base_model.output)

    output_class = Dense(
        4,
        activation='softmax', 
        name='output_class'
    )(output_class)

    output_bbox = Dense(
        32,
        activation='relu'
    )(base_model.output)

    output_bbox = Dense(
        4,
        name='output_bbox'
    )(output_bbox)

    model = Model(
        inputs = base_model.input,
        outputs = [
            output_class, 
            output_bbox
        ]
    )

    model.compile(
        loss = {
            'output_class': CategoricalCrossentropy(),
            'output_bbox': MeanSquaredError()
        }, 
        optimizer = Adam(learning_rate=0.001),
        metrics = {
            'output_class': ['accuracy'],
            'output_bbox': [MeanAbsoluteError()]
        },
        loss_weights={'output_class':1, 'output_bbox':100})

    model.summary()

    model.save('CustomMobileNetV2')

def argsMain() -> None:
    parser = argparse.ArgumentParser(description='Export a custom object recognition model for localization and classification')
    
    parser.add_argument('--preTrainedModel', type=str, help='Path to the pre trained model file', required=True)

    parser.add_argument('--datasetPath', type=str, help='Path to the dataset folder', required=True)

    args = parser.parse_args()

    exportModel(
        args.preTrainedModel,
        args.datasetPath
    )

def main() -> None:
    exportModel()

if __name__ == '__main__':
    #argsMain()
    main()