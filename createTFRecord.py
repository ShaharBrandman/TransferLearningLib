import os
import argparse
import io
import tensorflow as tf
import xml.etree.ElementTree as ET

from PIL import Image
from object_detection.utils import dataset_util

def createTFExample(xmlPath, imagePath, labelMapDict):
    with tf.io.gfile.GFile(xmlPath, 'r') as fid:
        xml_str = fid.read()
        fid.close()
    xml = ET.fromstring(xml_str)

    with tf.io.gfile.GFile(imagePath, 'rb') as fid:
        encoded_jpg = fid.read()
        fid.close()
        
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = os.path.basename(imagePath).encode('utf8')
    image_format = b'jpeg'

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    for obj in xml.findall('object'):
        xmin = float(obj.find('bndbox/xmin').text)
        ymin = float(obj.find('bndbox/ymin').text)
        xmax = float(obj.find('bndbox/xmax').text)
        ymax = float(obj.find('bndbox/ymax').text)

        xmins.append(max(0, xmin / float(width)))
        xmaxs.append(min(1, xmax / float(width)))
        ymins.append(max(0, ymin / float(height)))
        ymaxs.append(min(1, ymax / float(height)))

        class_name = obj.find('name').text.replace("'", '')
        classes_text.append(class_name.encode('utf8'))
        classes.append(labelMapDict[class_name])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example

def xml_to_tfrecord(xmlDir, imageDir, record_output, label_output):
    label_map_dict = {}
    used_ids = set()

    for xmlFile in os.listdir(xmlDir):
        if xmlFile.endswith('.xml'):
            xml_path = os.path.join(xmlDir, xmlFile)
            image_file = os.path.join(imageDir, xmlFile.replace('.xml', ''))

            extracted_label_map = extract_label_map(xml_path, used_ids)
            label_map_dict.update(extracted_label_map)

    with open(os.path.join(label_output, 'train_label_map.pbtxt'), 'w', encoding='utf-8') as f:
        for idx, (class_name, class_id) in enumerate(sorted(label_map_dict.items())):
            f.write('item {\n')
            f.write(f'  id: {idx + 1}\n')
            f.write(f'  name: \'{class_name}\'\n')
            f.write('}\n')

        f.close()

    record_file = os.path.join(record_output, 'train.tfrecord')
    with tf.io.TFRecordWriter(record_file) as f:
        for xmlFile in os.listdir(xmlDir):
            if xmlFile.endswith('.xml'):
                xml_path = os.path.join(xmlDir, xmlFile)
                
                image_file = os.path.join(imageDir, xmlFile.replace('.xml', '.jpg'))
                
                tf_example = createTFExample(xml_path, image_file, label_map_dict)

                f.write(tf_example.SerializeToString())
        f.close()

    print(f'TFRecord written to: {record_file}')
    print(f'Label map written to: {os.path.join(label_output, "label_map.pbtxt")}')

def extract_label_map(xml_path, used_ids):
    label_map_dict = {}
    with tf.io.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
    xml = ET.fromstring(xml_str)

    for obj in xml.findall('object'):
        class_name = obj.find('name').text
        if class_name not in label_map_dict:
            label_id = find_unique_id(used_ids)
            label_map_dict[class_name.replace("'", '')] = label_id
            used_ids.add(label_id)

    return label_map_dict

def find_unique_id(used_ids):
    label_id = 1
    while label_id in used_ids:
        label_id += 1
    return label_id

def argsMain() -> None:
    parser = argparse.ArgumentParser(description='Convert COCO-formatted XML annotations to TFRecord.')
    parser.add_argument('-annDir', type=str, help='Path to the Annotations directory')
    parser.add_argument('-imageDir', type=str, help='Path to the images directory')
    parser.add_argument('-recordOutput', type=str, help='Output Path for TFRecord')
    parser.add_argument('-labelOutput', type=str, help='Output Path for label map')

    args = parser.parse_args()

    xml_to_tfrecord(args.annDir, args.ImageDir, args.recordOutput, args.labelOutput)

def main() -> None:
    xml_to_tfrecord('data/annotations', 'data/images', 'data/', 'data/')

if __name__ == "__main__":
    main()