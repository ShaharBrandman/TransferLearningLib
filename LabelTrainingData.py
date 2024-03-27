'''
LabelTrainingData.py
© Author: ShaharBrandman (2024)

A script to recognize objects within images from a folder

using mobilenetv2 coco ssd architecure we can recognize objects
and save predictions in coco xml formatted annotations file
for later use
'''
import numpy as np
import os
import argparse
import cv2
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import xml.etree.ElementTree as ET

model = None
category_index = None

#load the desired keras model from file
def initModel(modelPath: str):
    return tf.saved_model.load(modelPath)

#preproccess an image from file by converting it to tensorflow tensor type
#optional: resizing the image to a different resolution for better model inference performence
def preproccess(path: str, resolution: tuple = None) -> tf.Tensor:
    image = cv2.imread(path)

    if resolution:
        image = cv2.resize((224, 224))
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.uint8)

def saveAsCOCOXML(boxes, classes, scores, imageShape, fileDetails: tuple) -> None:
    global category_index
    
    originalImagePath, outputPath = fileDetails

    outputDir = outputPath.split('/')
    outputDir = os.path.join(outputDir[0], outputDir[1], outputDir[2])
    
    os.makedirs(outputDir, exist_ok=True)

    root = ET.Element("annotation")

    filename = ET.SubElement(root, "filename")
    filename.text = originalImagePath.split("/")[-1]

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
        
        # if category_index[class_id]:
        #     name.text = category_index[class_id]['name']
        # else:
        #     name.text = str(class_id)
        name.text = category_index[class_id]['name']

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

    tree.write(outputPath)

def predictFile(
    imagePath: str,
    annotationsPath: str,
    expectedType: str = None,
    resolution: tuple = None
) -> None:
    global category_index
    global model

    image = preproccess(imagePath, resolution)
    
    infer = model.signatures["serving_default"]

    detections = infer(image)

    classes = detections['detection_classes'][0].numpy().astype(np.uint8)

    classes = [key for key, value in category_index.items() if value['name'] == expectedType]
    
    #make sure to only allow expected types only
    if len(classes) <= 0:
        return

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()

    saveAsCOCOXML(boxes, classes, scores, image.shape, (imagePath, annotationsPath))

def predictDirectory(
    folderPath: str,
    labelMapPath: str,
    annotationsPath: str,
    modelPath: str = 'ssd_mobilenet_v2_coco_2018_03_29/saved_model',
    datasetImageType: str = '.jpeg',
    resolution: tuple = None
) -> None:
    global category_index
    global model

    model = initModel(modelPath)

    category_index = label_map_util.create_category_index_from_labelmap(labelMapPath, use_display_name=True)

    for dir in os.listdir(folderPath):
        for img in os.listdir(f'{folderPath}{dir}'):
            expectedType = dir.split('-')[0]
            
            predictFile(
                f'{folderPath}{dir}/{img}', #the image in the folder to inference
                f'{annotationsPath}{dir}/{img.replace(datasetImageType, ".xml")}',
                expectedType,
                resolution
            )

def tupleType(arg):
    try:
        return tuple(map(int, arg.strip('()').split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid tuple format. Please provide tuples in the format, Example: (224,224).")

def argsMain() -> None:
    parser = argparse.ArgumentParser(description='Label a dataset in a COCO XML format using MobileNetV2')
    
    parser.add_argument(
        '--datasetPath',
        type=str,
        help='Path to the dataset folder',
        required=True
    )

    parser.add_argument(
        '--labelmap',
        type=str,
        help='Path to the label map file',
        required=True
    )

    parser.add_argument(
        'annotationsPath',
        type=str,
        help='Path to the Annotations output Folder',
        required=True
    )

    parser.add_argument(
        '--modelPath',
        type=str,
        help='Path to the pre-trained-model'
    )

    parser.add_argument(
        '--datasetType',
        type=str,
        help='Dataset Images type (jpeg, png, webp...)'
    )

    parser.add_argument(
        '--resolution',
        type=tupleType,
        nargs='+',
        help='desired dataset Resolutions'
    )

    parser.add_argument

    args = parser.parse_args()

    predictDirectory(
        folderPath = args.datasetPath,
        labelMapPath = args.labelMap, 
        annotationsPath = args.annotationsPath,
        datasetImageType = args.datasetType,
        modelPath = args.modelPath,
        resolution = args.resolution
    )

def main() -> None:
    predictDirectory('data/images/', 'data/default_mobilnetv2_labels.pbtxt', 'data/annotations/')

if __name__ == '__main__':
    main()