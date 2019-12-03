# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:08:18 2019

@author: RAJVIR
"""

import cv2

def nothing(x):
    pass    

cap = cv2.VideoCapture(0)
cv2.namedWindow('temp')
cv2.createTrackbar('bl', 'temp', 0, 255, nothing)
cv2.createTrackbar('gl', 'temp', 0, 255, nothing)
cv2.createTrackbar('rl', 'temp', 0, 255, nothing)
cv2.createTrackbar('bh', 'temp', 255, 255, nothing)
cv2.createTrackbar('gh', 'temp', 255, 255, nothing)
cv2.createTrackbar('rh', 'temp', 255, 255, nothing)
while True:
        ret,img=cap.read()
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        bl_temp=cv2.getTrackbarPos('bl', 'temp')
        gl_temp=cv2.getTrackbarPos('gl', 'temp')
        rl_temp=cv2.getTrackbarPos('rl', 'temp')
        bh_temp=cv2.getTrackbarPos('bh', 'temp')
        gh_temp=cv2.getTrackbarPos('gh', 'temp')
        rh_temp=cv2.getTrackbarPos('rh', 'temp')
        thresh=cv2.inRange(hsv,(bl_temp,gl_temp,rl_temp),(bh_temp,gh_temp,rh_temp))
        if(cv2.waitKey(10) & 0xFF == ord('q')):
            break 
        cv2.imshow('Video', img)
        cv2.imshow('thresh', thresh)
        
cap.release()
cv2.destroyAllWindows()