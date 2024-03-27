import numpy as np
import cv2
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import xml.etree.ElementTree as ET

DEFAULT_MODEL_PATH = "ssd_mobilenet_v2_coco_2018_03_29/saved_model"

def initModel(modelPath = DEFAULT_MODEL_PATH):
    return tf.saved_model.load(modelPath)

def preproccess(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.uint8)

def saveAsCOCOXML(boxes, classes, scores, imageShape, imagePath):
    global category_index
    root = ET.Element("annotation")

    filename = ET.SubElement(root, "filename")
    filename.text = imagePath.split("/")[-1]

    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    height = ET.SubElement(size, "height")
    depth = ET.SubElement(size, "depth")

    width.text = str(imageShape[1])
    height.text = str(imageShape[2])
    depth.text = str(imageShape[0])

    for box, class_id, score in zip(boxes, classes, scores):
        ymin, xmin, ymax, xmax = box
        if (ymin and xmin and ymax and xmax) == 0:
            continue
        object_elem = ET.SubElement(root, "object")
        name = ET.SubElement(object_elem, "name")
        if category_index[class_id]:
            name.text = category_index[class_id]['name']
        else:
            name.text = str(class_id)

        pose = ET.SubElement(object_elem, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(object_elem, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(object_elem, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(object_elem, "bndbox")
        xmin_elem = ET.SubElement(bndbox, "xmin")
        xmin_elem.text = str(float(xmin))
        ymin_elem = ET.SubElement(bndbox, "ymin")
        ymin_elem.text = str(float(ymin))
        xmax_elem = ET.SubElement(bndbox, "xmax")
        xmax_elem.text = str(float(xmax))
        ymax_elem = ET.SubElement(bndbox, "ymax")
        ymax_elem.text = str(float(ymax))

    tree = ET.ElementTree(root)
    tree.write(imagePath.replace('.jpg', '.xml'))

category_index = None

def predict(imagePath, label_map_path = 'labels.pbtxt'):
    global category_index
    model = initModel()

    image = preproccess(imagePath)
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
    
    infer = model.signatures["serving_default"]

    detections = infer(image)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.uint8)
    scores = detections['detection_scores'][0].numpy()

    saveAsCOCOXML(boxes, classes, scores, image.shape, path)

predict('1.jpg')