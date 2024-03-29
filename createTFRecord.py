import os
import argparse
import io
import xml.etree.ElementTree as ET

import tensorflow as tf

from PIL import Image

from object_detection.utils import dataset_util

#create a tfrecord example file from xml and images paths and a labelMap dictionary
def createTFExample(xmlPath, imagePath, labelMapDict) -> tf.train.Example:
    #read content from xml file
    with tf.io.gfile.GFile(xmlPath, 'r') as fid:
        xmlFileContent = fid.read()
        fid.close()

    xml = ET.fromstring(xmlFileContent)

    #read image and convert it to bytes
    with tf.io.gfile.GFile(imagePath, 'rb') as fid:
        encodedJpg = fid.read()
        fid.close()

    #ininitlize image and extracts its dimensions
    image = Image.open(io.BytesIO(encodedJpg))
    width, height = image.size

    filename = os.path.basename(imagePath).encode('utf8')
    
    #assign an image format, feel free to change it to whatever you wish
    image_format = b'jpeg'

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classesText, classes = [], []

    for obj in xml.findall('object'):
        xmin = float(obj.find('bndbox/xmin').text)
        ymin = float(obj.find('bndbox/ymin').text)
        xmax = float(obj.find('bndbox/xmax').text)
        ymax = float(obj.find('bndbox/ymax').text)

        xmins.append(max(0, xmin / float(width)))
        xmaxs.append(min(1, xmax / float(width)))
        ymins.append(max(0, ymin / float(height)))
        ymaxs.append(min(1, ymax / float(height)))

        className = obj.find('name').text.replace("'", '')
        classesText.append(className.encode('utf8'))
        classes.append(labelMapDict[className])

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encodedJpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classesText),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

#convert coco xml formatted dataset into tfrecord and labelmap files
def xml_to_tfrecord(xmlDir, imageDir, recordPath, labelMapPath):
    labelMapDict = {}
    usedIds = set()

    for xmlFolder in os.listdir(xmlDir):
        for xmlFile in os.listdir(os.path.join(xmlDir, xmlFolder)):
            if xmlFile.endswith('.xml'):
                xmlPath = os.path.join(xmlDir, xmlFolder, xmlFile)
                image_file = os.path.join(imageDir, xmlFile.replace('.xml', ''))

                updateLabelMap(labelMapDict, xmlPath, usedIds)
                
    with open(os.path.join(labelMapPath, 'train_label_map.pbtxt'), 'w', encoding='utf-8') as f:
        for idx, (className, classId) in enumerate(sorted(labelMapDict.items())):
            f.write('item {\n')
            f.write(f'  id: {idx + 1}\n')
            f.write(f'  name: \'{className}\'\n')
            f.write('}\n')

        f.close()

    record: str = os.path.join(recordPath, 'train.tfrecord')
    with tf.io.TFRecordWriter(record) as f:
        for xmlFolder in os.listdir(xmlDir):
            for xmlFile in os.listdir(os.path.join(xmlDir, xmlFolder)):
                if xmlFile.endswith('.xml'):
                    xmlPath = os.path.join(xmlDir, xmlFolder, xmlFile)
                    
                    imagePath = os.path.join(os.path.join(imageDir, xmlFolder), xmlFile.replace('.xml', '.jpeg'))
                    
                    tf_example = createTFExample(xmlPath, imagePath, labelMapDict)
                    
                    f.write(tf_example.SerializeToString())
        f.close()

    print(f'TFRecord written to: {record}')
    print(f'Label map written to: {os.path.join(labelMapPath, "label_map.pbtxt")}')

def updateLabelMap(labelMapDict, xmlPath, usedIds):
    #read xml file contents
    with tf.io.gfile.GFile(xmlPath, 'r') as fid:
        xmlContent = fid.read()

    xml = ET.fromstring(xmlContent)

    #find all subelements named object
    for obj in xml.findall('object'):
        #extract the object name
        name = obj.find('name').text
        if name not in labelMapDict:
            labelId = findUniqueID(usedIds)
            labelMapDict[name.replace("'", '')] = labelId
            usedIds.add(labelId)

def findUniqueID(usedIds):
    id = 1
    while id in usedIds:
        id += 1
    return id

def argsMain() -> None:
    parser = argparse.ArgumentParser(description='Convert COCO-formatted XML annotations to TFRecord.')
    
    parser.add_argument(
        '--annDir',
        type=str,
        help='Path to the Annotations directory',
        required=True
    )
    
    parser.add_argument(
        '--imageDir',
        type=str,
        help='Path to the images directory',
        required=True
    )
    
    parser.add_argument(
        '--recordOutput',
        type=str,
        help='Output Path for TFRecord',
        required=True
    )
    
    parser.add_argument(
        '--labelOutput',
        type=str,
        help='Output Path for label map',
        required=True
    )

    args = parser.parse_args()

    xml_to_tfrecord(args.annDir, args.ImageDir, args.recordOutput, args.labelOutput)

def main() -> None:
    xml_to_tfrecord('data/annotations', 'data/images', 'data/', 'data/')

if __name__ == "__main__":
    main()