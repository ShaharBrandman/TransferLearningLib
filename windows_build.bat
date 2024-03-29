@echo off
REM Install requirements using pip
pip install -r requirements.txt

REM Download the file using PowerShell's Invoke-WebRequest
PowerShell -Command "Invoke-WebRequest -Uri http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz -OutFile ssd_mobilenet_v2_coco_2018_03_29.tar.gz"

REM Extract the downloaded tar.gz file using PowerShell
PowerShell -Command "Expand-Archive -Path ssd_mobilenet_v2_coco_2018_03_29.tar.gz -DestinationPath ."

REM Optional: Delete the tar.gz file after extraction
del ssd_mobilenet_v2_coco_2018_03_29.tar.gz

echo Done.