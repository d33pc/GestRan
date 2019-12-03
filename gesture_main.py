# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:34:00 2019

@author: RAJVIR
"""

import cv2
import numpy as np
import pyautogui
import argparse
from keras.models import load_model
from utils import detector_utils as detector_utils


model = load_model('recon.h5')
detection_graph,sess = detector_utils.load_inference_graph()

if __name__ == '__main__':
    
    gest_class = ['palm','fist','fist_moved','ok','c','down']
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-sth',
            '--scorethreshold',
            dest='score_thresh',
            type=float,
            default=0.4,
            help='Score threshold for displaying bounding box')
    parser.add_argument(
            '-fps',
            '--fps',
            dest='fps',
            type=int,
            default=1,
            help='Show FPS on dispay visualization')
    parser.add_argument(
            '-src',
            '--source',
            dest='video_source',
            type=int,
            default=0,
            help='device index of the camera.')
    parser.add_argument(
            '-wd',
            '--width',
            dest='width',
            type=int,
            default=320,
            help='width of the frame in video stream')
    parser.add_argument(
            '-ht',
            '--height',
            dest='height',
            type=int,
            default=180,
            help='height of the frame in video stream')
    parser.add_argument(
            '-ds',
            '--display',
            dest='display',
            type=int,
            default=1,
            help='display the detected images using opencv, this reduces FPS')
    parser.add_argument(
            '-num-w',
            '--num-workers',
            dest='num-workers',
            type=int,
            default=4,
            help='number of workers')
    parser.add_argument(
            '-q-size',
            '--queue-size',
            dest='queue-size',
            type=int,
            default=5,
            help='size of the queue')
    
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
    cv2.namedWindow('capture',cv2.WINDOW_NORMAL)
    cv2.namedWindow('hand',cv2.WINDOW_AUTOSIZE)
    im_width,im_height = (cap.get(3),cap.get(4))
    print(im_width,im_height)
    num_hands_detect = 1
    score_thresh = args.score_thresh
    
    while True:
        ret,img = cap.read()
        
        try:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        except:
            print("image can not be converted to rgb")
        
        boxes,scores = detector_utils.detect_objects(img,detection_graph,sess)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        
        if scores[0]>score_thresh:
            
            left, right, top, bottom = boxes[0][1] * im_width, boxes[0][3] * im_width,boxes[0][0] * im_height, boxes[0][2] * im_height
            hand = img[int(top):int(bottom), int(left):int(right)]
            lab = cv2.cvtColor(hand, cv2.COLOR_BGR2LAB)
            mask2 = cv2.inRange(lab, np.array([17,128,136]), np.array([222,173,200]))
            kernel = np.ones((5, 5),np.uint8)
            filtered = cv2.GaussianBlur(mask2, (5,5), 0)
            morphed = cv2.morphologyEx(filtered,cv2.MORPH_OPEN,kernel)
            ret_o,otsu = cv2.threshold(morphed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            img_resized = cv2.resize(mask2,(320,120))
            img_resized = np.array(img_resized)
            img_final = np.reshape(img_resized,(1,120,320,1))
            
            try:    
                pred_ar = model.predict(img_final)
            except Exception as e:
                print("exception occured : ",e)
                
            for i in range(6):
                if pred_ar[0][i]>0.8:
                    cv2.putText(
                            img,gest_class[i],(10,170),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    
                    if i==0:
                        pyautogui.press('volumeup')
                    elif i==1:
                        pyautogui.press('volumedown')
                    elif i==3:
                        pyautogui.scroll(-50,pause=1)
                    break
                
            cv2.imshow('capture',img)
            cv2.imshow('hand',img_resized)
#            cv2.imshow("masked",morphed)
            
        else:
            cv2.putText(img,"hand not detected",(10,170),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('capture',img)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
            
            



