# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:34:32 2022

@author: srcdo
"""

import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as ms
import cv2
import sqlite3
import os
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers
    


root=tk.Tk()

root.title("Retinal Vessels Segmentation System")
w,h = root.winfo_screenwidth(),root.winfo_screenheight()

bg = Image.open(r"bh3.jpg")
#bg.resize((1366,500),Image.ANTIALIAS)
print(w,h)
bg_img = ImageTk.PhotoImage(bg)
bg_lbl = tk.Label(root,image=bg_img)
bg_lbl.place(x=0,y=93)
#, relwidth=1, relheight=1)



w = tk.Label(root, text="Retinal Vessels Segmentation System",width=65,background="#800517",foreground="white",height=2,font=("Times new roman",29,"bold"))
w.place(x=0,y=0)



w,h = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0"%(w,h))
root.configure(background="#800517")


from tkinter import messagebox as ms

import numpy as np
import cv2

def getJunctions(src):
    # the hit-and-miss kernels to locate 3-points junctions to be used in each directions
    # NOTE: float type is needed due to limitation/bug in warpAffine with signed char
    k1 = np.asarray([
        0,  1,  0,
        0,  1,  0,
        1,  0,  1], dtype=float).reshape((3, 3))
    k2 = np.asarray([
        1,  0,  0,
        0,  1,  0,
        1,  0,  1], dtype=float).reshape((3, 3))
    k3 = np.asarray([
        0, -1,  1,
        1,  1, -1,
        0,  1, 0], dtype=float).reshape((3, 3))

    # Some useful declarations
    tmp = np.zeros_like(src)
    ksize = k1.shape
    center = (ksize[1] / 2, ksize[0] / 2) # INVERTIRE 0 E 1??????
    # 90 degrees rotation matrix
    rotMat = cv2.getRotationMatrix2D(center, 90, 1)
    # dst accumulates all matches
    dst = np.zeros(src.shape, dtype=np.uint8)
    
    # Do hit & miss for all possible directions (0,90,180,270)
    for i in range(4):
        tmp = cv2.morphologyEx(src, cv2.MORPH_HITMISS, k1.astype(np.int8), tmp, (-1, -1), 1, cv2.BORDER_CONSTANT, 0)     
        dst = cv2.add(dst, tmp)
        tmp = cv2.morphologyEx(src, cv2.MORPH_HITMISS, k2.astype(np.int8), tmp, (-1, -1), 1, cv2.BORDER_CONSTANT, 0)
        dst = cv2.add(dst, tmp)
        tmp = cv2.morphologyEx(src, cv2.MORPH_HITMISS, k3.astype(np.int8), tmp, (-1, -1), 1, cv2.BORDER_CONSTANT, 0)
        dst = cv2.add(dst, tmp)
        # Rotate the kernels (90deg)
        k1 = cv2.warpAffine(k1, rotMat, ksize)
        k2 = cv2.warpAffine(k2, rotMat, ksize)
        k3 = cv2.warpAffine(k3, rotMat, ksize)
    
    return dst
def get_region(src):
    contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key = cv2.contourArea)
    black = np.zeros((src.shape[0], src.shape[1]), np.uint8)
    #black = np.zeros(src.shape, dtype=np.uint8)
    mask = cv2.drawContours(black,[c],0,255, -1)
    return mask
def train():
    basepath="E:/Palm Vein-Detection-in-real-time/Vein-Detection-in-real-time--PYTHON-master/21C9342_100%_Retinal_Vessels/dataset"
    
    # Initialing the CNN
    classifier = Sequential()
    
    # Step 1 - Convolution Layer 
    classifier.add(Convolution2D(32, 1,  1, input_shape = (64, 64, 3), activation = 'relu'))
    
    #step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size =(2,2)))
    
    # Adding second convolution layer
    classifier.add(Convolution2D(32, 1,  1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size =(2,2)))
    
    #Adding 3rd Concolution Layer
    classifier.add(Convolution2D(64, 1,  1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size =(2,2)))
    
    
    #Step 3 - Flattening
    classifier.add(Flatten())
    
    #Step 4 - Full Connection
    classifier.add(Dense(256, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(2, activation = 'softmax'))  #change class no.
    
    #Compiling The CNN
    classifier.compile(
                  optimizer = optimizers.SGD(lr = 0.01),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
    #Part 2 Fittting the CNN to the image
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory(
            basepath + '/images',
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory(
            basepath + '/imagestest',
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')
    
    steps_per_epoch = int( np.ceil(training_set.samples / 32) )
    val_steps = int( np.ceil(test_set.samples / 32) )
    
    model = classifier.fit_generator(
            training_set,
            steps_per_epoch=steps_per_epoch,
            epochs=200,
            validation_data = test_set,
            validation_steps =val_steps
          )
    
def Choose():
    global file
    file = askopenfilename(initialdir=r'', title='Select Image',
                                       filetypes=[("all files", "*.*")])
    
    # image3 =Image.open(file)
    # image3 =image3.resize((450,280), Image.ANTIALIAS)
    
    # choosen_image=ImageTk.PhotoImage(image3)
    
    # display = tk.Label(root, image=choosen_image)
    
    # display.image= choosen_image
    
    # display.place(x=10, y=100)
    src = cv2.imread(file)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src = cv2.resize(src, (600,400), interpolation = cv2.INTER_AREA)
    ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create(cv2.CV_32FC1, 1, 1 , 3, cv2.CV_8UC1, 1, 0 , cv2.BORDER_DEFAULT)
    ridges = ridge_filter.getRidgeFilteredImage(src)
    #cv2.imshow('Ridges', ridges)
    cv2.imwrite("ridge.jpg", ridges)

    blank_mask = np.zeros(src.shape, dtype=np.uint8)
    #kernel = np.ones((3,3),np.uint8)
    thresh = cv2.threshold(ridges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    only_road = get_region(thresh)
    subtract1 = cv2.subtract(only_road,thresh)
    #train()
    opening = cv2.dilate(thresh,None,iterations =2)
    #cv2.imshow("Adaptive Threshold", thresh)
    #cv2.imshow("subtract1 Threshold", subtract1)
    cv2.imwrite("threshold.jpg", thresh)
    cv2.imwrite("subtract.jpg", subtract1)
    thresh *= 255;
    # Morphology logic is: white objects on black foreground
    thresh = 255 - thresh;
    
    # Get junctions
    junctionsScore = getJunctions(thresh)
    
    # Draw markers where junction score is non zero
    dst = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    # find the list of location of non-zero pixels
    junctionsPoint = cv2.findNonZero(junctionsScore)
    
    for pt in junctionsPoint:
        pt = pt[0]
        dst[pt[1], pt[0], :] = [0, 0, junctionsScore[pt[1], pt[0]]]
    
    # show the result
    winDst = "Dst"
    winSrc = "Src"
    winJunc = "Junctions"
    
    #cv2.imshow(winSrc, src)
    cv2.imwrite("Src.jpg", src)
    #cv2.imshow(winJunc, junctionsScore)
    #cv2.imshow(winDst, dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
    image3 =Image.open("Src.jpg")
    
    image3 =image3.resize((300,300), Image.ANTIALIAS)
    
    choosen_image=ImageTk.PhotoImage(image3)
    
    display = tk.Label(root, image=choosen_image)
    
    display.image= choosen_image
    
    display.place(x=300, y=100)
    image3 =Image.open("threshold.jpg")
    
    image3 =image3.resize((300,300), Image.ANTIALIAS)
    
    choosen_image=ImageTk.PhotoImage(image3)
    
    display = tk.Label(root, image=choosen_image)
    
    display.image= choosen_image
    
    display.place(x=700, y=400)
    image3 =Image.open("subtract.jpg")
    
    image3 =image3.resize((300,300), Image.ANTIALIAS)
    
    choosen_image=ImageTk.PhotoImage(image3)
    
    display = tk.Label(root, image=choosen_image)
    
    display.image= choosen_image
    
    display.place(x=300, y=400)
    image3 =Image.open("ridge.jpg")
    
    image3 =image3.resize((300,300), Image.ANTIALIAS)
    
    choosen_image=ImageTk.PhotoImage(image3)
    
    display = tk.Label(root, image=choosen_image)
    
    display.image= choosen_image
    
    display.place(x=700, y=100)
    l10 = tk.Label(root, text="Result Image", width=13, font=("Times new roman", 15, "bold"), bg="snow")
    l10.place(x=750, y=720)

def Exit():
    root.destroy()
    
    
button1 = tk.Button(root,text='Choose Image',command=Choose,font=('Times New Roman',15,'bold'),width=14,bg='#FF8040',fg='black')
button1.place(x=50,y=250)

exit = tk.Button(root,text="Exit",command=Exit,font=('Times New Roman',15,'bold'),width=14,bg='red',fg='linen')
exit.place(x=50,y=350)





root.mainloop()