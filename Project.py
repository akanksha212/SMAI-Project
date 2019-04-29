#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import pandas as pd
import os
import json
import cv2 as cv2

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[69]:


def Extract_Img_Label(Img_path):
  
  # Extract image emotion label from Img path name and return label
    
    tokens = Img_path.split('.')
    Emotion_id = tokens[1][0:2]

    if "HA" in Emotion_id:
        label = "HAPPY"
    elif "SA" in Emotion_id:
        label = "SAD"
    elif "SU" in Emotion_id:
        label = "SURPRISED"
    elif "DI" in Emotion_id:
        label = "DISGUSTED"
    elif "FE" in Emotion_id:
        label = "FEAR"
    elif "AN" in Emotion_id:
        label = "ANGRY"
    elif "NE" in Emotion_id:
        label = "NEAUTRAL"
    else:
        label = "UNKNOWN"

    return label


# In[70]:


def Crop_Img(Img, Dim_Tuple=(70,90,184,230)):
  
    cropped_img = Img.crop(Dim_Tuple)
    imgarray = np.array(cropped_img)

    return imgarray


# In[71]:


def Load_Img(Img_Path='YM.SU3.60.tiff', Crop_Dim=(70,90,184,230)):
  
    img = Image.open(Img_Path, 'r')
    Cropped_Img = Crop_Img(img, Crop_Dim)
    if len(Cropped_Img) > 2:
        img = img.crop(Crop_Dim)
        r, g, b = img.split()
        ra = np.array(r)
        ga = np.array(g)
        ba = np.array(b)
        Cropped_Img = np.array(0.299*ra + 0.587*ga + 0.114*ba)
        
    img.close()
    return Cropped_Img


# In[72]:


def Load_Data(path):
    train = {}
    for root, dirs, files in os.walk(path):
        for filename in files:
            string_filename = str(filename)
            if string_filename.find("tiff") != -1:
                Imagepath = path + "/" + string_filename
                train[string_filename] = Load_Img(Imagepath, (70,90,184,230))
                
    return train


# In[73]:


def Display_Img(imgarray):
    img = Image.fromarray(imgarray)
    imgplot = plt.imshow(img)
    plt.show()


# In[74]:


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


# In[8]:


def st_dev(Img, i,j, m, N):
  
    a=int((N-1)/2)
    sd=0

    valid_cnt = 0
    for k in range(-a,a):
        for h in range(-a,a):
            if((k+i)>0 and (k+i)<Img.shape[0] and (h+j)>0 and (h+j)<Img.shape[1]):
                #m=mean(Img, i,j, N)
                sd=sd+np.square(Img[k+i][h+j]-m)
                valid_cnt = valid_cnt + 1

    sd=np.float128(sd/(valid_cnt*valid_cnt))
    sd=np.sqrt(sd)

    return sd


# In[9]:


def normalize(Img, N=11):
  
    Normalized_Img = np.zeros(Img.shape)

    for i in range(0,Img.shape[0]):
        for j in range(0,Img.shape[1]):
            
            m=mean(Img, i,j, N) 
            sd=st_dev(Img, i,j, m, N)

            Normalized_Img[i][j]=np.float128((Img[i][j]-m)/(6*sd))

    return Normalized_Img


# In[10]:


def Feature_Detection(Img, N=11):
    
    Feature_Detection_Img = np.zeros(Img.shape)

    for i in range(0,Img.shape[0]):
        for j in range(0,Img.shape[1]):
            
            m=mean(Img, i,j, N) 
            sd=st_dev(Img, i,j, m, N)

            #Normalized_Img[i][j]=np.float128((Img[i][j]-m)/(6*sd))
            Feature_Detection_Img[i][j] = sd

    return Feature_Detection_Img


# In[11]:


def Process_Train_Set(Dataset_Path, Train_DataSet, Normalisation_Window=11, FeatureDetection_Window=11):
  
    # Assuming Train_Img_DataSet is dictioanry with Key is Image name and value is 2_D aaray with pixel values
    Images_Data = {}
    l = 0 
            
    for Train_Img in Train_DataSet.keys():
        
        print("Training Loop: ", l)
        l = l+1
    
        # Load and crop train image
        Cropped_Img = Load_Img(Dataset_Path + "/" + Train_Img, (70,90,184,230))

#         # Crop the train image
#         Cropped_Img = Crop_Img(Img, (70,90,184,230))

        # Normalize train image
        Normalized_Img = normalize(Cropped_Img, Normalisation_Window)

        # Extract image from normalized image
        Feature_Detection_Img = Feature_Detection(Normalized_Img, FeatureDetection_Window)

        # Store Feature_Detection_Img to an array of dimension Train_Set_len * No_Img_Rows * No_Img_Cols
        Images_Data[Train_Img] = Feature_Detection_Img
    
    return Images_Data


# In[75]:


def Min_Max_Classifer( Train_DataSet, Test_Image_Path, Crop_Dim=(70,90,184,230),  Normalisation_Window=11, FeatureDetection_Window=11, alpha=3):
  
    # Pre-Process Test Image Load -> Crop -> Normalize -> Feature Detection
    Cropped_Img = Load_Img(Test_Image_Path, Crop_Dim)
    
#     if len(Cropped_Img.shape) > 2:
#         #Cropped_Img = cv2.resize(Cropped_Img, (Cropped_Img.shape[1], Cropped_Img.shape[0]), interpolation = cv2.INTER_AREA)
#         Cropped_Img = cv2.cvtColor(Cropped_Img, Cropped_Img, cv2.COLOR_BGR2GRAY,1)
    
    Normalized_Img = normalize(Cropped_Img, Normalisation_Window)
    Feature_Detection_Img = Feature_Detection(Normalized_Img, FeatureDetection_Window)

    # Determine Min_Max Distance of test image from each train image

    Max_Similarity = 0
    Label_Image = ""

    for Train_Img in Train_DataSet.keys():
        
        print("FOR IMAGE ", Train_Img)
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
        print("Total_Similarity: ", Total_Similarity)

        # Update Most similar train image and its similarity with test image
        if Total_Similarity > Max_Similarity:
            Max_Similarity = Total_Similarity 
            Label_Image = Train_Img
      
    # Extract emotion from image name
    print("Max_Similarity: ", Max_Similarity)
    print("Label_Image: ", Label_Image)
    
    Emotion_Label = Extract_Img_Label(Label_Image)
    return Emotion_Label


# In[13]:


DatasetPath = os.getcwd() + "/ml-face-jaffe-dataset-master/dataset"
train = Load_Data(DatasetPath)

Images_Data = Process_Train_Set(DatasetPath, train, 11, 11)
print("Images Data: ", Images_Data)


# In[14]:


np.save("Trained_Data.npy", Images_Data)


# In[16]:


test_img_dir = os.getcwd() + "/ml-face-jaffe-dataset-master/test"
test = Load_Data(test_img_dir)

Correct_Predictions = 0
Wrong_Predictions = 0

for test_img in test.keys():
    
    test_img_path = test_img_dir + "/" + test_img
    Emotion_Label = Min_Max_Classifer(Images_Data, test_img_path, (70,90,184,230), 11, 11, 3)
    
    Actual = Extract_Img_Label(test_img)
    Predicted = Emotion_Label
    
    if Actual == Predicted:
        Correct_Predictions += 1
    else:
        Wrong_Predictions += 1
    
    print("Emotion for the image " +  test_img + "is :" + Emotion_Label)
    
print("Correct_Predictions: ", Correct_Predictions)
print("Wrong_Predictions: ", Wrong_Predictions)

Accuracy = (Correct_Predictions / (Correct_Predictions + Wrong_Predictions))*100
print("Accuracy: ", Accuracy)


# In[76]:


test_img_path = os.getcwd() + "/ml-face-jaffe-dataset-master/Test_Tiff/AK.SU1.11.tiff"
Cropped_Img = Load_Img(test_img_path, (65,85,179,225))
Display_Img(Cropped_Img)

Emotion_Label = Min_Max_Classifer(Images_Data, test_img_path, (65,85,179,225),  11, 11, 3)
print("Predicted Emotion: ", Emotion_Label)
print("Actual Emotion: ", Extract_Img_Label("AK.SU1.11.tiff"))

# test_img_path = os.getcwd() + "/ml-face-jaffe-dataset-master/test/YM.SU3.60.tiff"
# Cropped_Img = Load_Img(test_img_path)
# Display_Img(Cropped_Img)



# In[77]:


test_img_path = os.getcwd() + "/ml-face-jaffe-dataset-master/Test_Tiff/PR.AN1.12.tiff"
Cropped_Img = Load_Img(test_img_path, (65,85,179,225))
Display_Img(Cropped_Img)

Emotion_Label = Min_Max_Classifer(Images_Data, test_img_path, (65,85,179,225),  11, 11, 3)
print("Predicted Emotion: ", Emotion_Label)
print("Actual Emotion: ", Extract_Img_Label("PR.AN1.12.tiff"))

# test_img_path = os.getcwd() + "/ml-face-jaffe-dataset-master/dataset/KA.AN1.39.tiff"
# Cropped_Img = Load_Img(test_img_path)
# Display_Img(Cropped_Img)


# In[78]:


test_img_path = os.getcwd() + "/ml-face-jaffe-dataset-master/Test_Tiff/EK.HA1.14.tiff"
Cropped_Img = Load_Img(test_img_path, (68,88,182,228))
Display_Img(Cropped_Img)

# test_img_path = os.getcwd() + "/ml-face-jaffe-dataset-master/dataset/MK.HA3.118.tiff"
# Cropped_Img = Load_Img(test_img_path)
# Display_Img(Cropped_Img)

Emotion_Label = Min_Max_Classifer(Images_Data, test_img_path, (68,88,182,228),  11, 11, 3)
print("Predicted Emotion: ", Emotion_Label)
print("Actual Emotion: ", Extract_Img_Label("EK.HA1.14.tiff"))


# In[79]:


test_img_path = os.getcwd() + "/ml-face-jaffe-dataset-master/Test_Tiff/PR.NE1.13.tiff"
Cropped_Img = Load_Img(test_img_path, (68,88,182,228))
Display_Img(Cropped_Img)

# test_img_path = os.getcwd() + "/ml-face-jaffe-dataset-master/dataset/TM.NE3.179.tiff"
# Cropped_Img = Load_Img(test_img_path)
# Display_Img(Cropped_Img)

Emotion_Label = Min_Max_Classifer(Images_Data, test_img_path, (68,88,182,228),  11, 11, 3)
print("Predicted Emotion: ", Emotion_Label)
print("Actual Emotion: ", Extract_Img_Label("PR.NE1.13.tiff"))


# In[ ]:




