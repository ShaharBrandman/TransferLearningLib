import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import tensorflow as tf
from object_detection.utils import label_map_util

def initModel(): 
    model = tf.saved_model.load("ssd_mobilenet_v2_coco_2018_03_29/saved_model")
    return model

def preproccess(path):
    image = cv2.imread(path)
    print(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.uint8)

def create_coco_xml(image_path, boxes, classes, scores, output_path):
    root = ET.Element("annotation")

    folder = ET.SubElement(root, "folder")
    folder.text = os.path.dirname(image_path)

    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(image_path)

    size = ET.SubElement(root, "size")
    height, width, channels = cv2.imread(image_path).shape
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(channels)

    for box, class_id, score in zip(boxes, classes, scores):
        object_elem = ET.SubElement(root, "object")

        ET.SubElement(object_elem, "name").text = str(class_id)
        ET.SubElement(object_elem, "pose").text = "Unspecified"
        ET.SubElement(object_elem, "truncated").text = "0"
        ET.SubElement(object_elem, "difficult").text = "0"

        bndbox = ET.SubElement(object_elem, "bndbox")
        ymin, xmin, ymax, xmax = box
        ET.SubElement(bndbox, "xmin").text = str(int(xmin * width))
        ET.SubElement(bndbox, "ymin").text = str(int(ymin * height))
        ET.SubElement(bndbox, "xmax").text = str(int(xmax * width))
        ET.SubElement(bndbox, "ymax").text = str(int(ymax * height))

    tree = ET.ElementTree(root)
    tree.write(output_path)

def predict(image, output_path):
    model = initModel()

    image = preproccess(image)

    label_map_path = 'labels.pbtxt' 
    category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

    infer = model.signatures["serving_default"]

    detections = infer(image)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.uint8)
    scores = detections['detection_scores'][0].numpy()

    create_coco_xml(image_path, boxes, classes, scores, output_path)

predict('1.jpeg', 'output.xml')