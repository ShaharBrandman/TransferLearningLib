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
        if d[1]:
            if d[1] not in foundClasses:
                foundClasses.append(d[1])
    return len(foundClasses)

def exportModel(preTrainedModelPath: str = None, datasetPath: str = 'data/annotations/') -> None:

    if preTrainedModelPath:
        base_model = tf.saved_model.load(preTrainedModelPath)
    else:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    '''
    X which stands for our extension of the base_model neural network
    which takes the output of the original architecure as input
    and use GlobalAveragePooling2D algorithm for fine tuning in our new model
    '''
    # x = base_model.output                       #new model input
    # x = layers.GlobalAveragePooling2D()(x)      #Fine tuning using GlobalAveragePooling2D
    # x = layers.Dense(256, activation='relu')(x) #Hidden layer

    x = base_model.output 
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.5)(x)


    output_class = layers.Dense(
        findNumberOfClasses(datasetPath),
        activation='softmax',
        name='output_class'
    )(x)

    output_bbox = layers.Dense(4, name='output_bbox')(x)

    '''
    initite a new model with the base_model neural network as our input
    and our new 5 neurons for classification and localization tasks
    which will output will be a single neuron for label prediction
    and 4 neurons which correspond to coordinates
    Example: (xmin, ymin, xmax, ymax)
    '''
    model = models.Model(inputs=base_model.input, outputs=[output_class, output_bbox])

    model.compile(
        optimizer='adam',                                       #Adaptive Moment Estimation for best general optimization
        loss = {
            'output_class': 'categorical_crossentropy',  #classification loss functions
            'output_bbox': 'mse'                                #localization loss functions
        },
        metrics = {
            'output_class': 'accuracy',                         #classification metrics
            'output_bbox': 'mae'                                #localization metrics
        }
    )

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