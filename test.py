import tensorflow as tf
import matplotlib.pyplot as plt

def initModel():
    return tf.keras.models.load_model('CustomMobileNetV2')

def loadLabelMap(labelMapPath):
    t = {}
    with open(labelMapPath, 'r') as file:
        for line in file:
            if "id:" in line:
                currentId = int(line.split(":")[1].strip())
            elif "name:" in line:
                t[currentId] = line.split(":")[1].strip().strip("'")
    return t

def visPrediction(imagePath, label, bbox):
    img = plt.imread(imagePath)
    plt.imshow(img)
    plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[2]), bbox[1] - bbox[0], bbox[3] - bbox[2], fill=False, edgecolor='r', linewidth=2))
    plt.text(bbox[0], bbox[2] - 5, label, color='red')
    plt.axis('off')
    plt.show()

def inference(imagePath, labelMap):
    model = initModel()

    img = plt.imread(imagePath)
    original_shape = img.shape[:2]

    img = tf.image.resize(img, (224, 224))
    img = img / 255.0

    classes, bbox = model.predict(tf.expand_dims(img, axis=0))

    classId = tf.argmax(classes, axis=1).numpy()[0]
    label = labelMap[classId]

    xmin, xmax, ymin, ymax = bbox[0]

    # Scaling bounding box back to original image size
    xmin = int(xmin * original_shape[1])
    xmax = int(xmax * original_shape[1])
    ymin = int(ymin * original_shape[0])
    ymax = int(ymax * original_shape[0])

    print(f'predicted label: {label}')
    print(f'predicted boxes: {xmin, xmax, ymin, ymax}')

    visPrediction(imagePath, label, (xmin, xmax, ymin, ymax))

def main() -> None:
    labelMap = loadLabelMap('data/train_label_map.pbtxt')
    inference('data/images/person-shahar/16.jpeg', labelMap)

if __name__ == '__main__':
    main()