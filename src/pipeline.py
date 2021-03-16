###################################################
# The pipeline organizes and orchestrates the     #
# overall flow of data in this project. The object#
# have three major methods: train on general data,#
# train on specific data, and a forward pass. It  #
# may also have additional methods for collecting #
# data etc...                                     #
###################################################
from models import Emotion_Classifier_Conv
from run_model import run_model
from fer13 import FER2013
from ingest import ingest_live_video, ingest_fer13

import torch

class Pipeline:
    def __init__ (self):
        self.model = Emotion_Classifier_Conv()
        
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu"  
        self.device = torch.device(dev)

        
    def train_general_model(self, fer13_path, output_path, learning_rate=0.01, batch_size=128, n_epochs=30, stop_thr=1e-4, use_valid=False):

        # t for train, v for validation, e for evaulation, f for features, l for labels
        self.fer13_tf, self.fer13_tl, self.fer13_vf, self.fer13_vl, self.fer13_ef, self.fer13_el = ingest_fer13(fer13_path)

        # preprocess data, augment data etc etc...

        trainset = FER2013('Training', self.fer13_tf, self.fer13_tl, self.fer13_vf, self.fer13_vl, self.fer13_ef, self.fer13_el)
        validset = FER2013('PublicTest', self.fer13_tf, self.fer13_tl, self.fer13_vf, self.fer13_vl, self.fer13_ef, self.fer13_el)
        testset = FER2013('PrivateTest', self.fer13_tf, self.fer13_tl, self.fer13_vf, self.fer13_vl, self.fer13_ef, self.fer13_el)

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


if __name__ == "__main__":
    p = Pipeline()
    p.train_general_model("data/icml_face_data.csv", "general_model.pt")

