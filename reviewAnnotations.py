import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def draw_bounding_boxes(image_path, annotations):
    """
    Draws bounding boxes on the image.
    
    Parameters:
        image_path (str): Path to the image file.
        annotations (list of dict): List of annotations, where each annotation
                                    is a dictionary containing 'xmin', 'ymin', 'xmax', 'ymax'.
    """
    # Open the image file
    with Image.open(image_path) as img:
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        for annotation in annotations:
            xmin = float(annotation['xmin'])
            ymin = float(annotation['ymin'])
            width = float(annotation['xmax']) - xmin
            height = float(annotation['ymax']) - ymin
            
            print(xmin, ymin, width, height)

            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=5, edgecolor='r', facecolor='none')

            ax.add_patch(rect)

        plt.show()

def parse_xml(annotation_path):
    """
    Parses the PASCAL VOC format XML file to extract bounding box information.
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    annotations = []

    for member in root.findall('object'):
        bndbox = member.find('bndbox')
        annotations.append({
            'xmin': bndbox.find('xmin').text,
            'ymin': bndbox.find('ymin').text,
            'xmax': bndbox.find('xmax').text,
            'ymax': bndbox.find('ymax').text
        })
    
    return annotations

def main(annotations_dir='data/annotations', images_dir='data/images'):
    for category in os.listdir(annotations_dir):
        print(category)
        for annotation_file in os.listdir(f'{annotations_dir}/{category}'):
            print(annotation_file)
            if annotation_file.endswith('.xml'):
                annotations = parse_xml(f'{annotations_dir}/{category}/{annotation_file}')
                
                draw_bounding_boxes(f'{images_dir}/{category}/{annotation_file.replace(".xml", ".jpeg")}', annotations)

if __name__ == '__main__':
    main()
