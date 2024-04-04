import tensorflow as tf
import matplotlib.pyplot as plt
import os
from exportCustomModel import exportModel

def initModel() -> None:
    return  tf.keras.models.load_model('CustomMobileNetV2')

def parse_tfrecord_fn(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32)
    }

    example = tf.io.parse_single_example(example, feature_description)

    # Decode JPEG image from 'image/encoded' feature
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    
    image = tf.image.resize(image, (224, 224))  # Resize image to match model input shape
    
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    
    label = tf.one_hot(example['image/object/class/label'].values, depth=num_classes)  # Convert class to one-hot encoding
    
    # Convert bounding box coordinates from SparseTensor to dense tensor
    bbox = tf.stack([
        tf.sparse.to_dense(example['image/object/bbox/xmin']),
        tf.sparse.to_dense(example['image/object/bbox/ymin']),
        tf.sparse.to_dense(example['image/object/bbox/xmax']),
        tf.sparse.to_dense(example['image/object/bbox/ymax'])
    ], axis=-1)
    
    return image, {'output_class': label, 'output_bbox': bbox}

# Define training parameters
batch_size = 1
num_epochs = 10
num_classes = 1  # Number of classes in your dataset

# Load TFRecord files
train_files = tf.io.gfile.glob('train.tfrecord')
train_dataset = tf.data.TFRecordDataset(train_files)

# Parse TFRecord examples and preprocess images
train_dataset = train_dataset.map(parse_tfrecord_fn)

# Shuffle, batch, and prefetch the dataset
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Train the model
model = initModel()
model.fit(train_dataset, epochs=num_epochs)
