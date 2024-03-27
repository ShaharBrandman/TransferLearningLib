#!/bin/sh

pip install -r requirements.txt

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

tar -zxf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

python3 downloadFromFirebase.py

python3 LabelTrainingData.py

python3 createTFRecord.py

python3 exportCustomModel.py

python3 train.py

python3 testWebcam.py