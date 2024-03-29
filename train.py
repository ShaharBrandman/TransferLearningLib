import tensorflow as tf
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

    for image, bbox, labels in dataset:
        model.fit(image, tf.argmax(labels, axis=1), epochs=epochs, batch_size=batchSize)

# def visPrediction(imagePath, label, bbox):
#     img = plt.imread(imagePath)
#     plt.imshow(img)
#     plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[2]), bbox[1] - bbox[0], bbox[3] - bbox[2], fill=False, edgecolor='r', linewidth=2))
#     plt.text(bbox[0], bbox[2] - 5, label, color='red')
#     plt.axis('off')
#     plt.show()

# def inference(imagePath, labelMap):
#     img = plt.imread(imagePath)
#     original_shape = img.shape[:2]

#     img = tf.image.resize(img, (224, 224))
#     img = img / 255.0

#     classes, bbox = model.predict(tf.expand_dims(img, axis=0))

#     classId = tf.argmax(classes, axis=1).numpy()[0]
#     label = labelMap[classId]

#     xmin, xmax, ymin, ymax = bbox[0]

#     # Scaling bounding box back to original image size
#     xmin = int(xmin * original_shape[1])
#     xmax = int(xmax * original_shape[1])
#     ymin = int(ymin * original_shape[0])
#     ymax = int(ymax * original_shape[0])

#     print(f'predicted label: {label}')
#     print(f'predicted boxes: {xmin, xmax, ymin, ymax}')

#     visPrediction(imagePath, label, (xmin, xmax, ymin, ymax))

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
    argsMain()

# def loadLabelMap(labelMapPath):
#     t = {}
#     with open(labelMapPath, 'r') as file:
#         for line in file:
#             if "id:" in line:
#                 currentId = int(line.split(":")[1].strip())
#             elif "name:" in line:
#                 t[currentId] = line.split(":")[1].strip().strip("'")
#     return t
# labelMap = loadLabelMap('data/train_label_map.pbtxt')
# inference('data/images/1.jpg', labelMap)