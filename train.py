'''
train.py
© Author: ShaharBrandman (2024)
'''
import os
import numpy as np
import tensorflow as tf
import argparse

from tensorflow.keras.callbacks import EarlyStopping

from exportCustomModel import findNumberOfClasses

def initModel():
    return tf.keras.models.load_model('CustomMobileNetV2')

def parseSingleTFRecord(record):
    featureDescription = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32)
    }

    record = tf.io.parse_single_example(record, featureDescription)

    image = tf.image.decode_jpeg(record['image/encoded'], channels=3)
    
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0

    labels = tf.sparse.to_dense(record['image/object/class/label'])
    bbox_xmin = tf.sparse.to_dense(record['image/object/bbox/xmin'])
    bbox_xmax = tf.sparse.to_dense(record['image/object/bbox/xmax'])
    bbox_ymin = tf.sparse.to_dense(record['image/object/bbox/ymin'])
    bbox_ymax = tf.sparse.to_dense(record['image/object/bbox/ymax'])
    
    bbox = tf.stack([bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax], axis=1)

    return image, bbox, labels

def getDataset(tfrecordPath):
    dataset = tf.data.TFRecordDataset(tfrecordPath)

    return dataset.map(parseSingleTFRecord)

def train(dataset, epochs=100, batchSize=1):
    dataset = dataset.batch(batchSize)
    
    es = EarlyStopping(patience=1, monitor='val_loss')
    
    model = initModel()

    if dataset is None:
        return print("Dataset is empty")

    os.makedirs('CustomMobileNetV2/checkpoint', exist_ok=True)

    num_classes = findNumberOfClasses('data/annotations/') # Debug: Print the number of classes
    print("Number of classes:", num_classes)

    images_list = []
    output_class_list = []
    output_bbox_list = []

    for image, bbox, labels in dataset:
        bbox = np.reshape(bbox, (-1, 4))
        #labels = np.reshape(labels, (-1, num_classes))

        print(f'image shape: {image.shape}')

        # Assuming labels are already one-hot encoded
        print(f'bbox shape: {bbox.shape}')
        print(f'label shape: {labels.shape}')

        model.fit(image, {"output_bbox": bbox, "output_class": labels}, epochs=epochs, callbacks=[es])
    model.save(f'CustomMobileNetV2/trainedModel.h5')

def argsMain() -> None:
    parser = argparse.ArgumentParser(description='transfer learn a custom object recognition model for localization and classification')
    
    parser.add_argument(
        '--datasetPath',
        type=str,
        help='Path to the training dataset path',
        required=True
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help='number of times to train the model (epochs)',
        required=True
    )

    parser.add_argument(
        '--batchSize',
        type=int,
        help='number of batch size to use for validation during training',
        required=True
    )

    args = parser.parse_args()

    train(
        getDataset(args.datasetPath),
        args.epochs,
        args.batchSize
    )

def main() -> None:
    train(
        getDataset('data/train.tfrecord')
    )

if __name__ == '__main__':
    #argsMain()
    main()
