#!/bin/sh

pip install -r requirements.txt

python3 downloadFromFirebase.py

python3 LabelTrainingData.py

python3 createTFRecord.py

python3 exportCustomModel.py

python3 train.py

python3 testWebcam.py