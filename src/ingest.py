##################################################
# Data ingestation funtions. We can either ingest#
# live webcam video or fer13 data                #
##################################################
import cv2
import pandas as pd
import numpy as np

# ingest live webcam video ingest to produce a video stream generator
def ingest_live_video():
    print('Video capture starting... Press \'q\' to stop')
    stream = cv2.VideoCapture(0)

    while True:
        b, img = stream.read()
        yield img
        
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
    


