
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


# In[84]:


def Load_TrainData(path):
    train = {}
    for root, dirs, files in os.walk(path):
        for filename in files:
            string_filename = str(filename)
            if string_filename.find("tiff") != -1:
                train[string_filename] = [[]]
    return train

def Load_Img(Img_Path='YM.SU3.60.tiff'):

  img = Image.open(Img_Path)
  img.show()
  return img


# In[85]:


def Crop_Img(Img, Dim_Tuple=(70,90,184,230)):

  cropped_img = Img.crop(Dim_Tuple)
  imgarray = np.array(cropped_img)

  print(imgarray.shape)
  print(imgarray)

  return imgarray


# In[86]:


def Display_Img(imgarray):
  img = Image.fromarray(imgarray)
  imgplot = plt.imshow(img)
  plt.show()


# In[87]:


def mean(Img, i,j, N):

  a=int((N-1)/2)
  u=0

  valid_cnt = 0
  for k in range(-a,a):
    for h in range(-a,a):
      if((k+i)>0 and (k+i)<Img.shape[0] and (h+j)>0 and (h+j)<Img.shape[1]):
        u=u+Img[k+i][h+j]
        valid_cnt = valid_cnt + 1

  u=np.float128(u/(valid_cnt*valid_cnt))
  return u


# In[88]:


def st_dev(Img, i,j, N):

  a=int((N-1)/2)
  sd=0

  valid_cnt = 0
  for k in range(-a,a):
    for h in range(-a,a):
      if((k+i)>0 and (k+i)<Img.shape[0] and (h+j)>0 and (h+j)<Img.shape[1]):
        m=mean(Img, i,j, N)
        sd=sd+np.square(Img[k+i][h+j]-m)
        valid_cnt = valid_cnt + 1

  sd=np.float128(sd/(valid_cnt*valid_cnt))
  sd=np.sqrt(sd)

  return sd


# In[89]:


def normalize(Img, N=11):

  Normalized_Img = np.zeros(Img.shape)

  for i in range(0,Img.shape[0]):
    for j in range(0,Img.shape[1]):

      print("index",i,j)
      m=mean(Img, i,j, N)
      sd=st_dev(Img, i,j, N)
      print("mean std",m, sd)

      Normalized_Img[i][j]=np.float128((Img[i][j]-m)/(6*sd))

  return Normalized_Img


# In[91]:
def Process_Train_Set(Train_DataSet, Normalisation_Window=11, FeatureDetection_Window=11):

  # Assuming Train_Img_DataSet is dictioanry with Key is Image name and value is 2_D aaray with pixel values
  Images_Data = {}

  for Train_Img in Train_DataSet.keys():

    # Load train image
    Img = Load_Img(Train_Img)

    # Crop the train image
    Cropped_Img = Crop_Img(Img, (70,90,184,230))

    # Normalize train image
    Normalized_Img = normalize(Cropped_Img, Normalisation_Window)

    # Extract image from normalized image
    Feature_Detection_Img = normalize(Normalized_Img, FeatureDetection_Window)

    # Store Feature_Detection_Img to an array of dimension Train_Set_len * No_Img_Rows * No_Img_Cols
    Images_Data[Train_Img] = Feature_Detection_Img

    return Images_Data

def Min_Max_Classifer(Train_DataSet, Test_Image_Path, Normalisation_Window=11, FeatureDetection_Window=11, alpha=3):
  
  # Pre-process train images
  Images_Data = Process_Train_Set(Train_DataSet, Normalisation_Window, FeatureDetection_Window)
  
  # Pre-Process Test Image Load -> Crop -> Normalize -> Feature Detection
  Img = Load_Img(Test_Image_Path)
  Cropped_Img = Crop_Img(Img, (70,90,184,230))
  Normalized_Img = normalize(Cropped_Img, Normalisation_Window)
  Feature_Detection_Img = normalize(Normalized_Img, FeatureDetection_Window)
  
  # Determine Min_Max Distance of test image from each train image
  
  Max_Similarity = 0
  Label_Image = ""
  
  for Train_Img in Train_DataSet.keys():
    
    Train_Img_Data = Train_DataSet[Train_Img]
    Test_Img_Data = Feature_Detection_Img
    
    # Calculate pixel wise minimum and maximum bet train and test image
    Pixel_Min = np.minimum(Train_Img_Data, Test_Img_Data)
    Pixel_Max = np.maximum(Train_Img_Data, Test_Img_Data)
    
    # Calculate pixel wise Min-Max Similarity
    Min_Max_ratio = np.divide(Pixel_Min, Pixel_Max)
    Min_Max_Sim = np.power(Min_Max_ratio, alpha)
    
    # Calculate total Min-Max Similarity
    Total_Similarity = np.sum(Min_Max_Sim)
    
    # Update Most similar train image and its similarity with test image
    if Total_Similarity > Max_Similarity:
      Max_Similarity = Total_Similarity 
      Label_Image = Train_Img
      
  # Extract emotion from image name
  Emotion_Label = Extract_Img_Label(Label_Image)
  return Emotion_Label

DatasetPath = os.getcwd() + "/ml-face-jaffe-dataset-master/dataset"
train = Load_TrainData(DatasetPath)

for ImagePath in train:
    path = DatasetPath + "/" + ImagePath
    Img = Load_Img(path)

imgplot = plt.imshow(Img)
plt.show()

Cropped_Img = Crop_Img(Img)
Display_Img(Cropped_Img)


# In[67]:


Normalized_Img = normalize(Cropped_Img, 11)


# In[68]:


print("Normalized Image array: ", Normalized_Img)
Display_Img(Normalized_Img)


# In[69]:


Feature_Detection_Img = normalize(Normalized_Img, 11)


# In[71]:


print("Features detected array: ", Feature_Detection_Img)
Display_Img(Feature_Detection_Img)
