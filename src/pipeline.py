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
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt


from .models import Emotion_Classifier_Conv, Emotion_Detector_Conv
from .run_model import run_model
from .dataloaders import FER2013, SpecificDataset
from .ingest import ingest_live_video, ingest_fer13, ingest_specific
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
        self.classes = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])

        return


    def ingest_fer13_data(self, fer13_path):
        # t for train, v for validation, e for evaulation, f for features, l for labels
        self.fer13_tf, self.fer13_tl, self.fer13_vf, self.fer13_vl, self.fer13_ef, self.fer13_el = ingest_fer13(fer13_path)
        return

    def ingest_specific_data(self, specific_path, train_ratio=0.8):
        # t for train, v for validation, e for evaulation, f for features, l for labels
        self.specific_tf, self.specific_tl, self.specific_vf, self.specific_vl, self.specific_ef, self.specific_el = ingest_specific(specific_path, train_ratio=train_ratio)
        return


    # train the general model on FER13 dataset
    def train_general_model(self, output_path, learning_rate=0.01, batch_size=128, n_epochs=30, stop_thr=1e-4, use_valid=False):
        self.general_model_path = output_path
        
        # initialize dataloader
        trainset = FER2013('Training', self.fer13_tf, self.fer13_tl, self.fer13_vf, self.fer13_vl)
        validset = FER2013('PublicTest', self.fer13_tf, self.fer13_tl, self.fer13_vf, self.fer13_vl)

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
        plt.subplot(1,2,1)
        # loss
        plt.plot(np.arange(n_epochs), self.general_train_info.get("loss"), label='training')
        if use_valid: plt.plot(np.arange(n_epochs), self.general_valid_info.get("loss"), label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Loss vs epoch')

        plt.subplot(1,2,2)
        # accuracy
        plt.plot(np.arange(n_epochs), self.general_train_info.get("acc"), label='training')
        if use_valid: plt.plot(np.arange(n_epochs), self.general_valid_info.get("acc"), label='validation')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs epoch')

        plt.show()
        
        # export model to output_path
        torch.save(self.model.state_dict(), self.general_model_path)
        print("General training complete!")
        return


    # train the general model on FER13 dataset
    def train_specific_model(self, output_path, learning_rate=0.01, batch_size=128, n_epochs=30, stop_thr=1e-4, use_valid=False):
        self.specific_model_path = output_path

        # initialize dataloader
        trainset = SpecificDataset(self.specific_tf, self.specific_tl, split="Training")
        validset = SpecificDataset(self.specific_vf, self.specific_vl, split="Validating")

        # freeze portion of the model
        self.model.conv1_block1.weight.requires_grad = False
        self.model.conv1_block1.bias.requires_grad = False
        self.model.conv2_block1.weight.requires_grad = False
        self.model.conv2_block1.bias.requires_grad = False
        
        self.model.conv1_block2.weight.requires_grad = False
        self.model.conv1_block2.bias.requires_grad = False
        self.model.conv2_block2.weight.requires_grad = False
        self.model.conv2_block2.bias.requires_grad = False

        #self.model.fc1.weight.requires_grad = False
        #self.model.fc1.bias.requires_grad = False
        #self.model.fc2.weight.requires_grad = False
        #self.model.fc2.bias.requires_grad = False

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
        plt.subplot(1,2,1)
        # loss
        plt.plot(np.arange(n_epochs), self.general_train_info.get("loss"), label='training')
        if use_valid: plt.plot(np.arange(n_epochs), self.general_valid_info.get("loss"), label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Loss vs epoch')

        plt.subplot(1,2,2)
        # accuracy
        plt.plot(np.arange(n_epochs), self.general_train_info.get("acc"), label='training')
        if use_valid: plt.plot(np.arange(n_epochs), self.general_valid_info.get("acc"), label='validation')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs epoch')

        plt.show()
        
        # export model to output_path
        torch.save(self.model.state_dict(), self.specific_model_path)
        print("Specific training complete!")
        return


    def evaluate_general_model(self):
        # initialize dataloader
        testset = FER2013('PrivateTest', privateTest_data=self.fer13_ef, privateTest_labels=self.fer13_el)

        # get test loss and accuracy
        test_loss, test_acc, pred = run_model(self.model, running_mode='test', test_set=testset, device=self.device)
        print("test loss: {}, test accuracy: {}".format(test_loss, test_acc.item()))

        # calculate confusion matrix
        cm = confusion_matrix(pred[0], pred[1])
        print("Confusion matrix:")
        print(cm)

        # calculate f1 score
        f1 = f1_score(pred[0], pred[1], average=None)
        print("F1-score: {}".format(f1))

        # draw some pretty graphs

        return test_loss, test_acc, cm, f1

    def evaluate_specific_model(self):
        # initialize dataloader
        testset = SpecificDataset(self.specific_ef, self.specific_el, split="Testing")

        # get test loss and accuracy
        test_loss, test_acc, pred = run_model(self.model, running_mode='test', test_set=testset, device=self.device)
        print("test loss: {}, test accuracy: {}".format(test_loss, test_acc.item()))
        
        # calculate confusion matrix
        cm = confusion_matrix(pred[0], pred[1])
        print("Confusion matrix:")
        print(cm)

        # calculate f1 score
        f1 = f1_score(pred[0], pred[1], average=None)
        print("F1-score: {}".format(f1))

        # draw some pretty graphs

        return test_loss, test_acc, cm, f1


    # there is a pretrained model, we can load it into 
    # self.model by specifying a path to the .pt file
    def load_model(self, model_path, mode="train"):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
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

        for i, emotion in enumerate(self.classes):
            print("Please make and hold a/an {} expression.".format(emotion))
            time.sleep(3)
            print("Start sampling in 3")
            time.sleep(1)
            print("Start sampling in 2")
            time.sleep(1)
            print("Start sampling in 1")
            time.sleep(1)

            samples = 0
            for frame in ingest_live_video():
                rects, crops = face_detect(frame, self.face_model_path, single=True, grayscale=True)

                if crops != []:
                    samples += 1

                    cropped = crops[0]
                    cv2.imshow('cropped', cropped)
                    rectangle = rects[0]
                    cv2.rectangle(frame, (rectangle[0],rectangle[1]), (rectangle[0]+rectangle[2],rectangle[1]+rectangle[3]), (0,255,0), 2)
                    
                    # resize the image to be 48x48, the same as the fer13 dataset
                    cropped = cv2.resize(cropped, (48, 48))

                    # finally write the image
                    image_path = os.path.join(self.specific_data_path, "{}_sample={}_class={}.jpg".format(prefix, samples, i))
                    cv2.imwrite(image_path, cropped)

                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                cv2.putText(frame, 'sampling {}/{}'.format(samples + 1, n_samples), (10, 30), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)

                if samples >= n_samples:
                    break

        return

    # run the model with live camera feed
    def run(self):
        self.model.eval()

        tfms = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])

        for frame in ingest_live_video():
            rects, crops = face_detect(frame, self.face_model_path, grayscale=True, throwout=3)

            if crops != []:
                cropped_list = []
                for rectangle, cropped in zip(rects, crops):
                    cv2.rectangle(frame, (rectangle[0], rectangle[1]), (rectangle[0]+rectangle[2], rectangle[1]+rectangle[3]), (0,255,0), 2)

                    # resize the image to be 48x48, the same as the fer13 dataset
                    cropped = cv2.resize(cropped, (48, 48))

                    # transform the cropped image
                    cropped = Image.fromarray(cropped.astype('uint8'))
                    crop_transformed = tfms(cropped)

                    cropped_list.append(crop_transformed)

                # feed the image into the model to get predictions
                output = self.model(torch.stack(cropped_list))
                _, predicted = torch.max(output.data, 1)

                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                for p, pred in enumerate(predicted):
                    class_name = self.classes[pred]
                    cv2.putText(frame, '{}'.format(class_name), (rects[p][0], rects[p][1]), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('frame', frame)



            




if __name__ == "__main__":
    p = Pipeline()
    #p.train_general_model("data/icml_face_data.csv", "general_model.pt")
    p.sample_specific_data(100, '../data/specific_dataset')
    print(p.test_acc)

    

