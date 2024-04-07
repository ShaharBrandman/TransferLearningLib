#!/bin/sh

pip install -r requirements.txt

wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

tar -zxf ssd_mobilenet_v2_coco_2018_03_29.tar.gz