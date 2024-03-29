'''
run.py
© Author: ShaharBrandman (2024)
'''
import downloadFromFirebase as dff
import LabelTrainingData as ltd
import createTFRecord as ctf
import exportCustomModel as ecm
import train as t

if __name__ == '__main__':
    dff.main()
    print('Downloaded the training data from firebase')
    ltd.main()
    print('Finished Succesully labeling the training data and creating xml annotations')
    ctf.main()
    print('Successully created a tf record')
    ecm.main()
    print('Exported the custom model')
    t.main()
    print('Done training!')