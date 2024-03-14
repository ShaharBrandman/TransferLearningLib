#!/bin/sh

# pip install -r requirements.txt

# python3 downloadFromFirebase.py

# git clone https://www.github.com/tensorflow/models.git

python3 createTFRecord.py

# python3 exportCustomModel.py

python3 train.py
