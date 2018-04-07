#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 14:51:51 2017

@author: mu7ammad
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

kernel = np.ones((7,7),np.uint8)

class Segmentation(object):

    def __init__(self, Image):
        self.Image = Image
        
    def binarization(self):
        ret, thresh1 = cv2.threshold(self.Image,30,255,cv2.THRESH_BINARY)
        # convert to white
        closing1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

        # save to /tmp folder
        cv2.imwrite("./tmp/thresh1.jpg", thresh1)
        cv2.imwrite("./tmp/closing1.jpg", closing1)
        
        return closing1, thresh1

    # removing skul
    def removing_skul(self, closing1):
        erosion = cv2.erode(closing1,kernel,iterations = 6)
        NO_skull = self.Image * (-erosion)

        # save to /tmp folder
        cv2.imwrite("./tmp/NO_skull.jpg", NO_skull)
        cv2.imwrite("./tmp/erosion.jpg", erosion)
        
        return NO_skull, erosion
    
    # enhance image to segmentation
    def enhance_image_t_seg(self, NO_skull):
        median = cv2.medianBlur(NO_skull,5)
        blur = cv2.GaussianBlur(median,(5,5),0)

        # save to /tmp folder
        cv2.imwrite("./tmp/blur.jpg", blur)
        
        return blur
    
    def to_Segmentation(self, blur):
        ret, thresh2 = cv2.threshold(blur,120,255,cv2.THRESH_BINARY)
        img = thresh2
        # Remove noise
        no_noise = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        # save to /tmp folder
        cv2.imwrite("./tmp/no_noise.jpg", no_noise)
        cv2.imwrite("./tmp/thresh2.jpg", thresh2)
       
        return no_noise, thresh2
    
    # edge detection
    def Edge_Detection(self,thresh2):
        edges = cv2.Canny(thresh2,100,200)

        # save to /tmp folder
        cv2.imwrite("./tmp/edges.jpg", edges)
        
        return edges

