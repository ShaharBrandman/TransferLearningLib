'''
downloadFromFirebase.py

a simple script to download the training dataset from firebase storage
'''
import firebase_admin
from firebase_admin import credentials, storage
import argparse
import os

def getFirebaseBucket(storageBucket: str) -> storage.bucket:
    cred = credentials.Certificate("serviceAccountKey.json")

    firebase_admin.initialize_app(cred, {
        'storageBucket': storageBucket
    })

    return storage.bucket()

def downloadAllFrom(bucket: str, path: str = 'trainingData') -> None:
    bucket = getFirebaseBucket(bucket)

    trainingDataBlobs = bucket.list_blobs(prefix = "trainingData")

    for blob in trainingDataBlobs:
        n = blob.name.split('/')
        try:
            if n[1] != '' and n[2] != '':
                os.makedirs(f'data/images/{n[1]}', exist_ok=True)

                blob.download_to_filename(os.path.join(f'data/images/{n[1]}', n[2]))
        except Exception as e:
            print(e)
            d = input('continue?, quit: anykey / c: continue')
            if d != 'c':
                return

def main() -> None:
    downloadAllFrom('sonicsteersystems.appspot.com')

def argsMain() -> None:
    parser = argparse.ArgumentParser(description='Download all trainingData from firebase storage')
    
    parser.add_argument(
        '--dbURL',
        type=str,
        help='Database URL',
        required=True
    )

    parser.add_argument(
        '--dirPath',
        type=str,
        help='Path to the trainingData directory',
        required=True
    )

    args = parser.parse_args()

    downloadAllFrom(args.dbURL, args.dirPath)

if __name__ == '__main__':
    argsMain()