## NOTE: requirements.txt does not contain pytorch installation, you will need to install pytorch separately

# iExpressionNet
## Introduction 
The proposed project will focus on utilizing deep learning to detect facial expressions in humans. While there have been many projects in the past that dealt with expression classification: (https://github.com/atulapra/Emotion-detection) and (https://tinyurl.com/1m4km78w), one motif we noticed throughout examining these projects is that the training accuracy on the expression classifier is often much higher than the validation accuracy. In other words, it may be the case that these models have a hard time generalizing expressions among different faces. This is perhaps due to the large variety of different faces present in the dataset.

While we will not attempt to improve the ability of deep learning models to generalize facial expression in this project, we will attempt to improve model performance for specific users. The overall idea is to train a model jointly, using both a general dataset of facial expressions as well as a dataset of a particular user’s facial expressions.

## Pipeline Overview
There will be three main parts to the pipline we want to create. Facial detection, general expression classification, and specific expression classification. Images containing (or not) human faces will be fed into the pipeline. Now, the first step in the pipeline will be to detect faces (or lack thereof). The most convenient way to do this is to use a pre-trained model provided by OpenCV. The facial detection step of the pipeline will output cropped images of faces, which is then fed into the next (and final) step of the pipeline to perform expression classification. Now, to train the expression classification step of the pipeline, we will first train a general CNN (same architecture as here: https://github.com/atulapra/Emotion-detection) with a general facial expression dataset (FER-13). Next, we will perform transfer learning on a specific CNN. We will freeze the convolutional layers of the previously trained general CNN and use that as the convolutional layers of the specific CNN. The fully-connected layers of the original CNN will then be trained by passing a specific dataset of user’s facial expressions.

## Dataset
The general dataset we are using is FER-13 with 7 classes (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set contains 28,709 examples. The public test set contains 3,589 examples, and the private test set contains another 3,589 examples. You can download the dataset at Kaggle: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

## References
- https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
- https://github.com/atulapra/Emotion-detection
