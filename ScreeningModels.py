import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import keras

def evaluate_model(preds,y,thresh):
  tn,fn,tp,fp = 0,0,0,0
  for i in range(len(preds)):
    if preds[i][0]  > thresh :
      if y[i][0] == 1:
        tn = tn + 1
      else:
        fn = fn + 1
    elif preds[i][0] < thresh:
      if y[i][0] == 0:
        tp = tp + 1
      else:
        fp = fp + 1
  sensitivity = tp/(tp+fn)
  specificity = tn/(tn+fp)
  return sensitivity, specificity

#CONV_GRAPH

model_conv = Sequential()
model_conv.add(Dense(1000, input_dim=4800, activation='relu'))
model_conv.add(Dense(250, activation='relu'))
model_conv.add(Dense(50, activation='relu'))
model_conv.add(Dense(2, activation='softmax'))
model_conv.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adam(lr=0.0001))

#FINGERPRINTS

model = Sequential()
model.add(Dense(500, input_dim=1024, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adam(lr=0.0001))

#DESCRIPTORS

model_desc = Sequential()
model_desc.add(Dense(50, input_dim=111, activation='relu'))
model_desc.add(Dense(15, activation='relu'))
model_desc.add(Dense(2, activation='softmax'))
model_desc.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.Adam(lr=0.0001))