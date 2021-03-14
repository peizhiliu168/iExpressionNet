import cv2


# the Ingest class is used to ingest data into the pipeline and its components
# there are different types of data the can be ingested including live video 
# feed, 
class Ingest:
    def __init__(self):
        pass

    def live_video(self):
        stream = cv2.VideoCapture(-1)
        while True:
            _, img = stream.read()
            print(img)
            cv2.imshow('img',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stream.release()
                cv2.destroyAllWindows()
                break


ingest = Ingest()
ingest.live_video()
