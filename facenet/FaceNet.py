import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from facenet.model import inceptionmodel

def build_facenet() :
    model = inceptionmodel.load_inception(dimension=128)
    weighted_model = inceptionmodel.load_weights(model)
    return weighted_model

def embed_faces(faces) :
    model = build_facenet()
    embeddings = []
    for face in faces:
        embeddings.append(np.squeeze(model.predict(face)))
  
    return embeddings

def process_face(face, target = (160,160)) :
    factor_H = target[0] / face.shape[0]
    factor_W = target[1] / face.shape[1]
    factor = min(factor_H, factor_W)

    dsize = (int(face.shape[1] * factor), int(face.shape[0] * factor))
    face = cv2.resize(face, dsize)

    # Then pad the other side to the target size by adding black pixels
    diff_0 = target[0] - face.shape[0]
    diff_1 = target[1] - face.shape[1]
	
	# Put the base image in the middle of the padded image
    face = np.pad(face, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
	
	#double check: if target image is not still the same size with target.
    if face.shape[0:2] != target:
        face = cv2.resize(face, target)

	#---------------------------------------------------

	#normalizing the image pixels

    face_pixels = img_to_array(face) #what this line doing? must?
    face_pixels = np.expand_dims(face_pixels, axis = 0)
    face_pixels /= 255 #normalize input in [0, 1]

	#---------------------------------------------------
    return face_pixels