# Face Recognition Project with Graph
Node classification with Face features and Graph Neural Network
> Jinho Lee (jinhohl2@illinois.edu)
>
> Seonggeun Cho (sc27@illinois.edu)

## Description
The purpose of this project is to utilize neural network and graph structure for face recognition. 
Two face detection methods; Retinaface and FaceNet are used to detect faces from images and extract their features.
We construct a graph structure where the vectors correspond to nodes, based on their visual resemblance.
Then, we utilized the node classification methods of Graph Neural Network in order to classify the identity of the face.

## File Structures
- **retinaface**: Detect face from am image
- **facenet**: Convert a face into a 128D vector
- **face2nodevec.py**: Import images from input_pictures directory and return a - list of feature vectors of the faces in input images. 
- **gnn_preprocess.py**: Generate data required for Graph Neural Network in matrix form.
- **main.inpynb**: Perform node classification 

## Environment Requirement
The code has been tested under Python 3.7.12. The required packages are as follows:

* tensorflow == 2.7.0
* pytorch == 1.3.1
* dgl == 0.6.1
* pytorch == 1.10.0+cu111
* scipy == 1.4.1
* numpy == 1.19.5
* sklearn == 1.0.1

## Setup
- Download weights of pre-trained models of Retinaface and Facenet to the root directory of the project
- Place face images in the input_pictures directory
- follow steps in the main.ipynb file

## Acknowledgement

**Facenet: A unified embedding for face recognition and clustering**. Florian Schroff, Dmitry Kalenichenko, James Philbin, CVPR, 2015.

**Retinaface: Single-stage dense face localisation in the wild**. Jiankang Deng, Jia Guo, Yuxiang Zhou, Jinke Yu, Irene Kotsia, Stefanos Zafeiriou, Imperial College London, InsightFace, Middlesex University London, FaceSoft, CVPR, 2019.

## Licence
MIT License
