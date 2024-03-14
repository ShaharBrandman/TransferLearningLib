'''
downloadFromFirebase.py

a simple script to download the training dataset from firebase storage

using firebase_admin to download from gs://sonicsteersystems.appspot.com
after authenticating as admin@sonicsteersystems.com which then and only then you can
read data

just for clearification these are the Authentication Policies:
    API KEY Access - (Write) => /testOutput, /trainingData for webapp access
    admin@sonicsteersystems.com - (Read, Write) => Everywhere

anyways this script signin to firebase as admin
and downloads /trainingData to File system in dataset/
'''