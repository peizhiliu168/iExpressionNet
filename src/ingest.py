##################################################
# Data ingestation funtions. We can either ingest#
# live webcam video or fer13 data                #
##################################################
import cv2
import pandas as pd
import numpy as np
import glob
import os
import re
from sklearn.model_selection import train_test_split

# ingest live webcam video ingest to produce a video stream generator
def ingest_live_video():
    print('Video capture starting... Press \'q\' to stop')
    stream = cv2.VideoCapture(0)

    while True:
        b, img = stream.read()
        yield img, stream
        
        # press "q" to stop stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stream.release()
            cv2.destroyAllWindows()
            break

# ingest FER13 data given path to data
def ingest_fer13(path):
    def preprocessing(data):
        image_array = np.zeros(shape=(len(data), 48, 48))
        image_label = np.array(list(map(int, data['emotion'])))
        for i, row in enumerate(data.index):
            image = np.fromstring(data.loc[row, ' pixels'], dtype=int, sep=' ')
            image = np.reshape(image, (48, 48))
            image_array[i] = image
        return image_array, image_label


    data = pd.read_csv(path)

    train_data, train_labels = preprocessing(data[data[' Usage']=='Training'])
    publicTest_data, publicTest_labels = preprocessing(data[data[' Usage']=='PrivateTest'])
    privateTest_data, privateTest_labels = preprocessing(data[data[' Usage']=='PublicTest'])

    return train_data, train_labels, publicTest_data, publicTest_labels, privateTest_data, privateTest_labels

def ingest_specific(path, train_ratio):
    # find all file names
    # load all images
    # parse all file names for lables
    # split based on train ratio evenly

    files = glob.glob(os.path.join(path,"*.jpg"))

    all_data = []
    all_labels = []
    for f in files:
        # get image data
        img = cv2.imread(f, flags=cv2.IMREAD_ANYCOLOR)
        all_data.append(img)

        # parse the filename to get the class. Note the file
        # name should always be in the format {prefix}_sample={number}_class={label}.jpg
        # otherwise, this dataloader won't work
        basename = os.path.basename(f).split(".")[0]
        all_labels.append(int(basename.split("_")[2].split("=")[1]))

    # first split the train and test/validation sets
    train_data, eval_data, train_labels, eval_labels = train_test_split(all_data, all_labels, train_size=train_ratio, stratify=all_labels)
    valid_data, test_data, valid_labels, test_labels = train_test_split(eval_data, eval_labels, train_size=0.5, stratify=eval_labels)
    
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels



#ingest_specific("data/specific_dataset", 0.8)

    


