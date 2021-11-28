from face2nodevec import *
import math
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import LabelEncoder

def load_gnn_data():
  #node features
  input_feats, input_names = face2nodevec("input_pictures")
  input_feats = np.array(input_feats) 
  input_names = np.array(input_names)
  #target_reps = np.asarray(face2nodevec("target_pictures"))

  #adjacency matrix
  edges = np.matmul(input_feats, input_feats.T)
  print(edges)
  edges_scaled = mat_minmax_scale(edges) 
  print(edges_scaled)
  links = edges_scaled * (edges_scaled >= 0.4).astype(int)
  A = (links != 0).astype(int)
  print(A)
  #labels and number of classes
  le = LabelEncoder()
  input_labels = le.fit(input_names).transform(input_names)
  num_classes = len(le.classes_)

  #train and test masks
  n = len(input_labels)
  
  train_mask = np.random.choice([0, 1], size=(n,), p=[2/10, 8/10])
  test_mask = (np.ones(n) - train_mask).astype(int)

  return A, input_feats, input_labels, num_classes, train_mask, test_mask

#Min-Max Scaling the entire 2-D matrix
##threshold: threshold value that determine whethere there is a link between nodes. 
##re-scaled all values in the matrix to values in between [0,1]
def mat_minmax_scale(input_mat):
  #if threshold == None:
  arr = np.reshape(input_mat, input_mat.shape[0]*input_mat.shape[1])
  arr_scaled = minmax_scale(arr)
  return np.reshape(arr_scaled, (input_mat.shape[0], input_mat.shape[1]))  
    #arr = np.reshape(input_mat, input_mat.shape[0]*input_mat.shape[1])
    #arr[np.where(arr == 0)] = threshold
    #arr_scaled = minmax_scale(arr)
  #  return np.reshape(arr_scaled, (input_mat.shape[0], input_mat.shape[1]))