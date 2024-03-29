'''
train.py
© Author: ShaharBrandman (2024)
'''
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

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

    return image, (
        tf.sparse.to_dense(record['image/object/bbox/xmin']),
        tf.sparse.to_dense(record['image/object/bbox/xmax']),
        tf.sparse.to_dense(record['image/object/bbox/ymin']),
        tf.sparse.to_dense(record['image/object/bbox/ymax'])
    ), tf.sparse.to_dense(record['image/object/class/label'])

def getDataset(tfrecordPath):
    dataset = tf.data.TFRecordDataset(tfrecordPath)
    
    return dataset.map(parseSingleTFRecord)

def train(dataset, epochs=10, batchSize=32):
    dataset = dataset.batch(batchSize)
    
    model = initModel()

    for image, bbox, labels in dataset:
        model.fit(image, tf.argmax(labels, axis=1), epochs=epochs, batch_size=batchSize)

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
        args.datasetPath,
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