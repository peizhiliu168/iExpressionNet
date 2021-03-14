###################################################
# face detection functions                        #
###################################################
import cv2
import numpy as np


# given an image, crop the face from the image. Since there can be
# multiple faces in an image, this function will return dictionaries
# in the format: {'rectangles':[(x,y,w,y),...], 'faces':[cropped faces,...]}
def face_detect(image, model_path, grayscale=False, **kwargs):
    model = cv2.CascadeClassifier(model_path)

    scaleFactor = 1.05
    minNeighbors = 7
    minSize = (48,48)

    if not grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if "scaleFactor" in kwargs:
        scaleFactor = kwargs['scaleFactor']
    if "minNeighbors" in kwargs:
        minNeighbors = kwargs['minNeighbors']
    if 'minSize' in kwargs:
        minSize = kwargs['minSize']

    faces = model.detectMultiScale(image, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)

    rect = []
    crop = []
    for (x,y,w,h) in faces:
        rect.append((x,y,w,h))
        crop.append(np.array(image[y:y+h, x:x+w]))
        #cv2.imshow("crop", np.array(image[y:y+h, x:x+w]))
    return {'rectangles': rect, 'faces': crop}

    

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