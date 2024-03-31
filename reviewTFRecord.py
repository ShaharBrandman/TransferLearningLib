import tensorflow as tf
import matplotlib.pyplot as plt

raw_image_dataset = tf.data.TFRecordDataset('data/train.tfrecord')

image_feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32)
}

#Parse function to extract features
def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

for image_features in parsed_image_dataset: 
    image_raw = image_features['image/encoded'].numpy() #extract raw image bytes

    # Convert the raw bytes into a Tensor
    image = tf.image.decode_jpeg(image_raw)

    # Plot the image
    plt.figure()
    plt.imshow(image)
    
    # Extract bounding box coordinates
    xmin = tf.sparse.to_dense(image_features['image/object/bbox/xmin']).numpy()
    xmax = tf.sparse.to_dense(image_features['image/object/bbox/xmax']).numpy()
    ymin = tf.sparse.to_dense(image_features['image/object/bbox/ymin']).numpy()
    ymax = tf.sparse.to_dense(image_features['image/object/bbox/ymax']).numpy()
    
    labels = tf.sparse.to_dense(image_features['image/object/class/label']).numpy()
    classNames = tf.sparse.to_dense(image_features['image/object/class/text']).numpy()

    print(f'labels: {labels}, className: {classNames}')
    print(f'labels shape: {labels.shape}, className shape: {classNames.shape}')
    print(f'xmin shape {xmin.shape}, xmax: {xmax.shape}, ymin: {ymin.shape}, ymax: {ymax.shape}')

    # Plot bounding boxes
    for i in range(len(xmin)):
        print('bounding boxes: ', xmin[i], xmax[i], ymin[i],ymax[i], ' for ', classNames[i])
        plt.plot([xmin[i], xmax[i], xmax[i], xmin[i], xmin[i]],
                 [ymin[i], ymin[i], ymax[i], ymax[i], ymin[i]], 'r-')
    
    plt.title('Image with Bounding Boxes')
    plt.axis('off')
    plt.show()
