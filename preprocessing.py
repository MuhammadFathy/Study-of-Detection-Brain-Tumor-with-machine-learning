#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 15:42:10 2017

@author: mu7ammad
"""

import cv2
import numpy as np

from segmentaion import Segmentation

class Preprocessing(object):
    grayImage   = ''
    image       = ''
    closing1    = ''
    thresh1     = ''
    NO_skull    = ''
    erosion     = ''
    blur        = ''
    no_noise    = ''
    edge        = ''
    tumourImage = ''
    
    def preproces(self, originalImageUrl):
        image = cv2.imread(str(originalImageUrl));
        self.grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        self.image = Segmentation(self.grayImage)

    # binarization
    def binarization(self):
        self.closing1, self.thresh1 = self.image.binarization()
        #plt.imshow(closing1,'gray')

    # removing_skul
    def removingSkul(self):
        self.NO_skull, self.erosion = self.image.removing_skul(self.closing1)
        #plt.imshow(NO_skull,'gray')

    # enhance image to segmentation
    def enhanceImage(self):
        self.blur = self.image.enhance_image_t_seg(self.NO_skull)
        #plt.imshow(blur,'gray')

    # Segmentation
    def segmentation(self):
        self.no_noise, self.thresh2 = self.image.to_Segmentation(self.blur)
        #plt.imshow(no_noise,'gray')

        self.edge = self.image.Edge_Detection(self.thresh2)

        # Plot
        #segm.Show_plots(img,thresh1,closing1,erosion,blur,NO_skull,no_noise)

    def getInfectedRegion(self):
        col1 = 0
        col2 = 0
        row1 = 0
        row2 = 0
        
        for i in xrange(self.no_noise.shape[1]):
            for j in xrange(self.no_noise.shape[0]):
                if self.no_noise.item(j, i) > 0:
                    if col1 == 0 & col2 == 0:
                        col1 = j
                        row1 = i
                    else:
                        if col1 > j:
                            col1 = j
                        else:
                            if col2 < j:
                                col2 = j
                            
                        if row1 > i:
                            row1 = i
                        else:
                            if row2 < i:
                                row2 = i

        # draw rectangle to select tumour region                    
        cv2.rectangle(self.no_noise, (row1, col1), (row2, col2), (255,0,0), 2)

        # save again for good preview 
        cv2.imwrite("./tmp/no_noise.jpg", self.no_noise)

        # tumourImage
        self.tumourImage = self.no_noise[col1:col2, row1:row2]
        cv2.imwrite("./tmp/tumourImage.jpg", self.tumourImage)
        return col1, col2, row1, row2
        

  

preprocessing = Preprocessing()

preprocessing.preproces('/home/mu7ammad/workspace/Pythonwork/segmentation/images/FLAIR/img10.jpg')
preprocessing.binarization()
preprocessing.removingSkul()
preprocessing.enhanceImage()
preprocessing.segmentation()
preprocessing.getInfectedRegion()