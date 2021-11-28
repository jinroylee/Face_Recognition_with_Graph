import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from retinaface import RetinaFace
from facenet import FaceNet

#encode each image in input directory as 128D vector and return the name of the faces
def face2nodevec(input_directory):
  faces = []
  names = []

  for filename in os.listdir(input_directory):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".PNG") or filename.endswith(".JPG"):
      face = np.squeeze(RetinaFace.extract_faces(img_path = input_directory+"/"+filename, align = True))
      faces.append(face)

      name = filename.split("_")[0]
      names.append(name)

  resizedfaces = []
  for face in faces:
    resizedface = FaceNet.process_face(face)
    resizedfaces.append(resizedface)
    

  node_reps = FaceNet.embed_faces(resizedfaces)
  
  return node_reps, names


#current = Path().resolve()
#print(current)



