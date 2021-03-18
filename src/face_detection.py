###################################################
# face detection functions                        #
###################################################
import cv2
import numpy as np


# given an image, crop the face from the image. Since there can be
# multiple faces in an image, this function will return dictionaries
# in the format: {'rectangles':[[x,y,w,y],...], 'faces':[cropped faces,...]}
def face_detect(image, model_path, throwout=5, grayscale=False, single=False, **kwargs):
    model = cv2.CascadeClassifier(model_path)

    scaleFactor = 1.05
    minNeighbors = 7
    minSize = (48,48)

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if "scaleFactor" in kwargs:
        scaleFactor = kwargs['scaleFactor']
    if "minNeighbors" in kwargs:
        minNeighbors = kwargs['minNeighbors']
    if 'minSize' in kwargs:
        minSize = kwargs['minSize']

    rects, _, weights = model.detectMultiScale3(image, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize, outputRejectLevels=True)
    
    # check if there is face detected
    if rects == ():
        return [], []

    # if there is, determine the faces to throw out since it is not above a certain threshold
    rects = np.array(rects)
    weights = np.squeeze(np.array(weights), axis=1)
    indicator = weights > throwout
    rects = rects[indicator]
    weights = weights[indicator]
    
    if weights.size == 0:
        return [], []

    if single:
        idx = np.argmax(weights)
        rect = rects[idx]
        return np.array([rects[idx]]), np.array([image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]])
    
    crop = []
    for rect in rects:
        crop.append(image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])
    return rects, np.array(crop)

    

'''# detect and crop faces in a streaming manner
def face_detect_stream(images, model_path, grayscale=False, **kwargs):
    for image in images:
        yield face_detect(image, model_path, grayscale, **kwargs)

# detect and crop faces in a bluk manner
def face_detect_bulk(images, model_path, grayscale=False, **kwargs):
    results = []
    for image in images:
        results.append(face_detect(image, model_path, grayscale, **kwargs))
    return results'''