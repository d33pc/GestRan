# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:33:50 2019

@author: RAJVIR
"""

import os
import cv2
import numpy as np
from keras import layers,models
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Dropout




lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir(r'E:\hand recon\gestures\00'):
    if not j.startswith('.'):
        lookup[j] = count
        reverselookup[count] = j
        count+=1 


x_data = []
y_data = []
datacount = 0
for i in range(10):
    for j in os.listdir(r'E:\hand recon\gestures\0'+str(i)+'\\'):
        if not j.startswith('.'):
            count = 0
            for k in os.listdir(r'E:\hand recon\gestures\0'+str(i)+'\\'+j+'\\'):
                img = cv2.imread(r'E:\hand recon\gestures\0'+str(i)+'\\'+j+'\\'+k,0)
                img = cv2.resize(img,(320,120))
                _,img = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
                x_data.append(img)
                count+=1
            y_values = np.full((count,1),lookup[j])
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype='float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1)
y_data  = to_categorical(y_data)
x_data = x_data.reshape((datacount,120,320,1))
x_data /= 255

   
x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size=0.2)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size=0.5)

model = models.Sequential()
model.add(layers.Conv2D(32,(5,5),strides=(2,2),activation='relu', input_shape=(120,320,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),strides=(2,2),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(layers.Dense(6,activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64,verbose=1,validation_data=(x_validate,y_validate))

model.save('recon.h5')









