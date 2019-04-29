#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import pandas as pd
import os
import json
import cv2 as cv2

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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


def Process_Train_Set(Dataset_Path, Train_DataSet, Normalisation_Window=11, FeatureDetection_Window=11):

    # Assuming Train_Img_DataSet is dictioanry with Key is Image name and value is 2_D aaray with pixel values
    Images_Data = {}
    l = 0

    for Train_Img in Train_DataSet.keys():

        print("Training Loop: ", l)
        l = l+1

        # Load and crop train image
        Cropped_Img = Load_Img(Dataset_Path + "/" + Train_Img, (70,90,184,230))

        # Normalize train image
        Normalized_Img = normalize(Cropped_Img, Normalisation_Window)

        # Extract image from normalized image
        Feature_Detection_Img = Feature_Detection(Normalized_Img, FeatureDetection_Window)

        # Store Feature_Detection_Img to an array of dimension Train_Set_len * No_Img_Rows * No_Img_Cols
        Images_Data[Train_Img] = Feature_Detection_Img

    return Images_Data


DatasetPath = os.getcwd() + "/ml-face-jaffe-dataset-master/dataset"
train = Load_Data(DatasetPath)

Images_Data = Process_Train_Set(DatasetPath, train, 11, 11)
print("Images Data: ", Images_Data)

# Save pre-proceesed training image data
with open("Trained_Data.pckl", 'wb') as f:
    pickle.dump(Images_Data, f)

np.save("Trained_Data.npy", Images_Data)
