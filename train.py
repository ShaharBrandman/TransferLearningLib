'''
train.py
© Author: ShaharBrandman (2024)
'''
import tensorflow as tf
import argparse

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

    num_classes = findNumberOfClasses('data/images')
    labels = tf.one_hot(tf.sparse.to_dense(record['image/object/class/label']), depth=num_classes)
    labels = tf.reshape(labels, (-1, num_classes))  # Reshape to (batch_size, num_classes)
    
    return image, (
        tf.sparse.to_dense(record['image/object/bbox/xmin']),
        tf.sparse.to_dense(record['image/object/bbox/xmax']),
        tf.sparse.to_dense(record['image/object/bbox/ymin']),
        tf.sparse.to_dense(record['image/object/bbox/ymax'])
    ), labels

def getDataset(tfrecordPath):
    dataset = tf.data.TFRecordDataset(tfrecordPath)

    return dataset.map(parseSingleTFRecord)

def train(dataset, epochs=50, batchSize=1):
    dataset = dataset.batch(batchSize)
    
    model = initModel()

    if dataset is None:
        return print("Dataset is empty")

    for image, bbox, labels in dataset:
        if image.shape[0] == 0:
            continue
        print("Image shape:", image.shape)
        print("Labels shape:", labels.shape)
        model.fit(image, labels, epochs=epochs, batch_size=batchSize)

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
