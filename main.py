#########################################################
# Mainly here to test different aspects of the pipeline #
#########################################################
from src.face_detection import face_detect
from src.ingest import ingest_live_video
import cv2

def main():    
    for frame in ingest_live_video():
        cropped = face_detect(frame, 'data\\haarcascade_frontalface_default.xml', grayscale=True)
        print(cropped['rectangles'])
        if cropped['rectangles']:
            rectangle = cropped['rectangles'][0]
            cv2.rectangle(frame, (rectangle[0],rectangle[1]), (rectangle[0]+rectangle[2],rectangle[1]+rectangle[3]), (255,0,0), 2)
        cv2.imshow('frame',frame)











# indeally, main will never get imported
if __name__ == "__main__":
    main()
