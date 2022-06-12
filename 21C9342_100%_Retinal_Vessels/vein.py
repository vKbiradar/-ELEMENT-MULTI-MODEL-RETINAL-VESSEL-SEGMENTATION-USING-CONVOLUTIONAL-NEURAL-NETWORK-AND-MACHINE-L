# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:34:32 2022

@author: srcdo
"""

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

src = cv2.imread("C:/Users/srcdo/Downloads/Vein-Detection-in-real-time--PYTHON-master/Vein-Detection-in-real-time--PYTHON-master/02_test.tif")
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src = cv2.resize(src, (600,400), interpolation = cv2.INTER_AREA)
ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create(cv2.CV_32FC1, 1, 1 , 3, cv2.CV_8UC1, 1, 0 , cv2.BORDER_DEFAULT)
ridges = ridge_filter.getRidgeFilteredImage(src)
cv2.imshow('Ridges', ridges)

blank_mask = np.zeros(src.shape, dtype=np.uint8)
#kernel = np.ones((3,3),np.uint8)
thresh = cv2.threshold(ridges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
only_road = get_region(thresh)
subtract1 = cv2.subtract(only_road,thresh)

opening = cv2.dilate(thresh,None,iterations =2)
cv2.imshow("Adaptive Threshold", thresh)
cv2.imshow("subtract1 Threshold", subtract1)

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

cv2.imshow(winSrc, src)
cv2.imshow(winJunc, junctionsScore)
cv2.imshow(winDst, dst)
cv2.waitKey()
cv2.destroyAllWindows()