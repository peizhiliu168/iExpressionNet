###################################################
# The pipeline organizes and orchestrates the     #
# overall flow of data in this project. The object#
# have three major methods: train on general data,#
# train on specific data, and a forward pass. It  #
# may also have additional methods for collecting #
# data etc...                                     #
###################################################
import torch
import os
import time
import cv2

from .models import Emotion_Classifier_Conv, Emotion_Detector_Conv
from .run_model import run_model
from .fer13 import FER2013
from .ingest import ingest_live_video, ingest_fer13
from .face_detection import face_detect

class Pipeline:
    def __init__ (self, face_model_path='data/haarcascade_frontalface_default.xml'):
        # determine GPU or CPU to use
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu"  
        self.device = torch.device(dev)

        self.face_model_path = face_model_path

        # initialize pipline model
        #self.model = Emotion_Classifier_Conv()
        self.model = Emotion_Detector_Conv()

        # a list of expression categories in the FER13 dataset. The
        # labels and indices in array map one-to-one
        self.categories = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    # train the general model on FER13 dataset
    def train_general_model(self, fer13_path, output_path, learning_rate=0.01, batch_size=128, n_epochs=30, stop_thr=1e-4, use_valid=False):

        # t for train, v for validation, e for evaulation, f for features, l for labels
        self.fer13_tf, self.fer13_tl, self.fer13_vf, self.fer13_vl, _, _ = ingest_fer13(fer13_path)
        trainset = FER2013('Training', self.fer13_tf, self.fer13_tl, self.fer13_vf, self.fer13_vl, self.fer13_ef, self.fer13_el)
        validset = FER2013('PublicTest', self.fer13_tf, self.fer13_tl, self.fer13_vf, self.fer13_vl, self.fer13_ef, self.fer13_el)

        if (use_valid):
            self.model, self.general_train_info, self.general_valid_info = run_model(self.model, 
                                                                                    running_mode='train', 
                                                                                    train_set=trainset, 
                                                                                    valid_set=validset, 
                                                                                    batch_size=batch_size, 
                                                                                    learning_rate=learning_rate, 
                                                                                    n_epochs=n_epochs, 
                                                                                    device=self.device, 
                                                                                    stop_thr=stop_thr)
        else:
            self.model, self.general_train_info, _ = run_model(self.model, 
                                                                running_mode='train', 
                                                                train_set=trainset, 
                                                                batch_size=batch_size, 
                                                                learning_rate=learning_rate, 
                                                                n_epochs=n_epochs, 
                                                                device=self.device)

        # draw some pretty graphs
        
        # export model to output_path
        torch.save(self.model.state_dict(), output_path)
        return


    def evaluate_general_model(self, fer13_path):
        # t for train, v for validation, e for evaulation, f for features, l for labels
        _, _, _, _, self.fer13_ef, self.fer13_el = ingest_fer13(fer13_path)
        testset = FER2013('PrivateTest', self.fer13_tf, self.fer13_tl, self.fer13_vf, self.fer13_vl, self.fer13_ef, self.fer13_el)

        # get test loss and accuracy
        self.test_loss, self.test_acc = run_model(self.model, running_mode='test', test_set=testset, device=self.device)
        
        # draw some pretty graphs
        return


    # there is a pretrained general model, we can load it into 
    # self.model by specifying a path to the .pt file
    def load_general_model(self, general_model_path, mode="train"):
        self.model.load_state_dict(torch.load(general_model_path))
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()
        return

    
    # sample specific data for the user
    def sample_specific_data(self, n_samples, path, prefix="emotion"):
        self.specific_data_path = path
        if not os.path.exists(path):
            os.makedirs(path)

        for i, emotion in enumerate(self.categories):
            print("Please make and hold a/an {} expression.".format(emotion))
            time.sleep(3)
            print("Start sampling in 3")
            time.sleep(1)
            print("Start sampling in 2")
            time.sleep(1)
            print("Start sampling in 1")
            time.sleep(1)

            for samples, frame in enumerate(ingest_live_video()):
                rects, crops = face_detect(frame, self.face_model_path, single=True, grayscale=True)

                if crops != []:
                    cropped = crops[0]
                    cv2.imshow('cropped', cropped)
                    rectangle = rects[0]
                    cv2.rectangle(frame, (rectangle[0],rectangle[1]), (rectangle[0]+rectangle[2],rectangle[1]+rectangle[3]), (0,255,0), 2)
                    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                    
                    # resize the image to be 48x48, the same as the fer13 dataset
                    cropped = cv2.resize(cropped, (48, 48))

                    # finally write the image
                    image_path = os.path.join(self.specific_data_path, "{}_sample={}_class={}.jpg".format(prefix, samples, i))
                    cv2.imwrite(image_path, cropped)


                cv2.putText(frame, 'sampling {}/{}'.format(samples, n_samples), (10, 30), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)

                if samples >= n_samples:
                    break

        return



if __name__ == "__main__":
    p = Pipeline()
    #p.train_general_model("data/icml_face_data.csv", "general_model.pt")
    p.sample_specific_data(100, '../data/specific_dataset')
    print(p.test_acc)

    

