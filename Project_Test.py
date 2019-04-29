#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import json
import cv2 as cv2

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


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
        label = "NEUTRAL"
    else:
        label = "UNKNOWN"

    return label


def Crop_Img(Img, Dim_Tuple=(70,90,184,230)):

    cropped_img = Img.crop(Dim_Tuple)
    imgarray = np.array(cropped_img)

    return imgarray


def Load_Img(Img_Path='YM.SU3.60.tiff', Crop_Dim=(70,90,184,230)):

    img = Image.open(Img_Path, 'r')
    Cropped_Img = Crop_Img(img, Crop_Dim)

    # Convert 3-channel image to grayscale 1-channel image
    if len(Cropped_Img.shape) > 2:
        img = img.crop(Crop_Dim)
        r, g, b = img.split()
        ra = np.array(r)
        ga = np.array(g)
        ba = np.array(b)
        Cropped_Img = np.array(0.299*ra + 0.587*ga + 0.114*ba)

    img.close()
    return Cropped_Img


def Load_Data(path):
    train = {}
    for root, dirs, files in os.walk(path):
        for filename in files:
            string_filename = str(filename)
            if string_filename.find("tiff") != -1:
                Imagepath = path + "/" + string_filename
                train[string_filename] = Load_Img(Imagepath, (70,90,184,230))

    return train


def Display_Img(imgarray):
    img = Image.fromarray(imgarray)
    imgplot = plt.imshow(img)
    plt.show()


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


def st_dev(Img, i,j, m, N):

    a=int((N-1)/2)
    sd=0

    valid_cnt = 0
    for k in range(-a,a):
        for h in range(-a,a):
            if((k+i)>0 and (k+i)<Img.shape[0] and (h+j)>0 and (h+j)<Img.shape[1]):

                sd=sd+np.square(Img[k+i][h+j]-m)
                valid_cnt = valid_cnt + 1

    sd=np.float128(sd/(valid_cnt*valid_cnt))
    sd=np.sqrt(sd)

    return sd


def normalize(Img, N=11):

    Normalized_Img = np.zeros(Img.shape)

    for i in range(0,Img.shape[0]):
        for j in range(0,Img.shape[1]):

            m=mean(Img, i,j, N)
            sd=st_dev(Img, i,j, m, N)

            Normalized_Img[i][j]=np.float128((Img[i][j]-m)/(6*sd))

    return Normalized_Img


def Feature_Detection(Img, N=11):

    Feature_Detection_Img = np.zeros(Img.shape)

    for i in range(0,Img.shape[0]):
        for j in range(0,Img.shape[1]):

            m=mean(Img, i,j, N)
            sd=st_dev(Img, i,j, m, N)

            Feature_Detection_Img[i][j] = sd

    return Feature_Detection_Img


def Min_Max_Classifer( Train_DataSet, Test_Image_Path, Crop_Dim=(70,90,184,230),  Normalisation_Window=11, FeatureDetection_Window=11, alpha=3):

    # Pre-Process Test Image Load -> Crop -> Normalize -> Feature Detection
    Cropped_Img = Load_Img(Test_Image_Path, Crop_Dim)

    Normalized_Img = normalize(Cropped_Img, Normalisation_Window)
    Feature_Detection_Img = Feature_Detection(Normalized_Img, FeatureDetection_Window)

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
    print("Max_Similarity  " + str(Max_Similarity) + " found with train image " + str(Label_Image))

    Emotion_Label = Extract_Img_Label(Label_Image)
    return Emotion_Label


# Load pre-proceesed training image data
with open("Trained_Data.pckl", 'rb') as f:
    Images_Data = pickle.load(f)

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

print("\n Validation Testing results:")
print("Correct_Predictions: ", Correct_Predictions)
print("Wrong_Predictions: ", Wrong_Predictions)

Accuracy = (Correct_Predictions / (Correct_Predictions + Wrong_Predictions))*100
print("Accuracy: ", Accuracy)


print("\n Unknown Testing results:")

# TEST IMAGE 1
test_img_path = os.getcwd() + "/ml-face-jaffe-dataset-master/Test_Tiff/AK.SU1.11.tiff"
Cropped_Img = Load_Img(test_img_path, (65,85,179,225))
Display_Img(Cropped_Img)

Emotion_Label = Min_Max_Classifer(Images_Data, test_img_path, (65,85,179,225),  11, 11, 3)
print("\nResult for image AK.SU1.11.tiff " )
print("Predicted Emotion: ", Emotion_Label)
print("Actual Emotion: ", Extract_Img_Label("AK.SU1.11.tiff"))

# TEST IMAGE 2
test_img_path = os.getcwd() + "/ml-face-jaffe-dataset-master/Test_Tiff/PR.AN1.12.tiff"
Cropped_Img = Load_Img(test_img_path, (65,85,179,225))
Display_Img(Cropped_Img)

Emotion_Label = Min_Max_Classifer(Images_Data, test_img_path, (65,85,179,225),  11, 11, 3)
print("\nResult for image PR.AN1.12.tiff " )
print("Predicted Emotion: ", Emotion_Label)
print("Actual Emotion: ", Extract_Img_Label("PR.AN1.12.tiff"))

# TEST IMAGE 3
test_img_path = os.getcwd() + "/ml-face-jaffe-dataset-master/Test_Tiff/EK.HA1.14.tiff"
Cropped_Img = Load_Img(test_img_path, (68,88,182,228))
Display_Img(Cropped_Img)

Emotion_Label = Min_Max_Classifer(Images_Data, test_img_path, (68,88,182,228),  11, 11, 3)
print("\nResult for image EK.HA1.14.tiff " )
print("Predicted Emotion: ", Emotion_Label)
print("Actual Emotion: ", Extract_Img_Label("EK.HA1.14.tiff"))

# TEST IMAGE 4
test_img_path = os.getcwd() + "/ml-face-jaffe-dataset-master/Test_Tiff/PR.NE1.13.tiff"
Cropped_Img = Load_Img(test_img_path, (68,88,182,228))
Display_Img(Cropped_Img)

Emotion_Label = Min_Max_Classifer(Images_Data, test_img_path, (68,88,182,228),  11, 11, 3)
print("\nResult for image PR.NE1.13.tiff " )
print("Predicted Emotion: ", Emotion_Label)
print("Actual Emotion: ", Extract_Img_Label("PR.NE1.13.tiff"))

# TEST IMAGE 5
test_img_path = os.getcwd() + "/ml-face-jaffe-dataset-master/Test_Tiff/KR.SU1.15.tiff"
Cropped_Img = Load_Img(test_img_path, (68,88,182,228))
Display_Img(Cropped_Img)

Emotion_Label = Min_Max_Classifer(Images_Data, test_img_path, (68,88,182,228),  11, 11, 3)
print("\nResult for image KR.SU1.15.tiff " )
print("Predicted Emotion: ", Emotion_Label)
print("Actual Emotion: ", Extract_Img_Label("KR.SU1.15.tiff"))
