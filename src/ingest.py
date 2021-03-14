##################################################
# Data ingestation funtions. We can either ingest#
# live webcam video or fer13 data                #
##################################################
import cv2
import pandas as pd

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

    train_image_array, train_image_label = preprocessing(data[data[' Usage']=='Training'])
    val_image_array, val_image_label = preprocessing(data[data[' Usage']=='PrivateTest'])
    test_image_array, test_image_label = preprocessing(data[data[' Usage']=='PublicTest'])

    return train_image_array, train_image_label, val_image_array, val_image_label, test_image_array, test_image_label
    

