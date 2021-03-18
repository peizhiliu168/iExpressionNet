#########################################################
# Mainly here to test different aspects of the pipeline #
#########################################################
from src.face_detection import face_detect
from src.ingest import ingest_live_video
from src.pipeline import Pipeline

import cv2

def main():    
    '''p = Pipeline()
    p.train_general_model("data/icml_face_data.csv", "general_model.pt", learning_rate=1e-3, n_epochs=40, stop_thr=1e-4, use_valid=True, batch_size=128)
    print(p.test_loss, p.test_acc)'''
    '''for frame in ingest_live_video():
        rects, cropped = face_detect(frame, 'data\\haarcascade_frontalface_default.xml', grayscale=True, single=True)
        print(cropped)
        if cropped != []:
            cv2.imshow('t',cropped[0])
            rectangle = rects[0]
            cv2.rectangle(frame, (rectangle[0],rectangle[1]), (rectangle[0]+rectangle[2],rectangle[1]+rectangle[3]), (255,0,0), 2)
        cv2.imshow('frame',frame)'''


    p = Pipeline()
    #p.sample_specific_data(200, "data/specific_dataset")
    p.load_model("data/general_model.pt", mode="train")
    #p.ingest_specific_data("data/specific_dataset", train_ratio=0.8)
    p.ingest_fer13_data("data/icml_face_data.csv")
    #p.train_specific_model("data/specific_model.pt", learning_rate=1e-3, n_epochs=4, stop_thr=1e-5, use_valid=True, batch_size=32)
    #p.evaluate_specific_model()
    p.evaluate_general_model()
    #p.sample_specific_data(100, 'data/specific_dataset')
    #p.run()











# indeally, main will never get imported
if __name__ == "__main__":
    main()
