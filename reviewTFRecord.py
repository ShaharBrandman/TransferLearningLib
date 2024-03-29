import tensorflow as tf
import matplotlib.pyplot as plt

# Load the TFRecord dataset
raw_image_dataset = tf.data.TFRecordDataset('data/train.tfrecord')

# Create a dictionary describing the features.
image_feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32)
}

# Parse function to extract features
def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

# Apply parsing function to the dataset
parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

# Visualize the images and bounding boxes
for image_features in parsed_image_dataset.take(5): # Visualize the first 5 images
    image_raw = image_features['image/encoded'].numpy() # Extract raw image bytes

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
    
    # Plot bounding boxes
    for i in range(len(xmin)):
        print(xmin[i], xmax[i], ymin[i],ymax[i])
        plt.plot([xmin[i], xmax[i], xmax[i], xmin[i], xmin[i]],
                 [ymin[i], ymin[i], ymax[i], ymax[i], ymin[i]], 'r-')
    
    plt.title('Image with Bounding Boxes')
    plt.axis('off')
    plt.show()
