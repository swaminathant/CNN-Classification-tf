#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: swaminathan t
"""
# Dog cat classifier so only two classes


import os
import numpy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, Flatten
from tensorflow.keras import models, optimizers, callbacks



def Preprocessing(Dim, ClassName, PreData):
    DataDir = ["train","validation"]
    BaseDir = os.path.dirname(__file__)
    trainDir = os.path.join(BaseDir, "cats_and_dogs", DataDir[0])
    ValidationDir = os.path.join(BaseDir, "cats_and_dogs", DataDir[1])
    if PreData == False:
        # This is for the train data loop
        TrainingData = []
        TrainingLabels = []
        for SingleClass in ClassName:
            for Root, Folder, File in os.walk(os.path.join(trainDir, SingleClass)):
                for Data in File:
                    if Data.split(".")[0] == "dog":
                        Img = Image.open(os.path.join(trainDir, SingleClass, Data))
                        Img = Img.resize(Dim, Image.ANTIALIAS)
                        TrainingData.append(np.array(Img))
                        TrainingLabels.append(0)
                    elif Data.split(".")[0] == "cat":
                        Img = Image.open(os.path.join(trainDir, SingleClass, Data))
                        Img = Img.resize(Dim, Image.ANTIALIAS)
                        TrainingData.append(np.array(Img))
                        TrainingLabels.append(1)
        # Convert the FeatureIndex to numpy array
        TrainingData = np.array(TrainingData)
        TrainingLabels = np.array(TrainingLabels)            
        print("Below is the training Data set details dimensions")
        print(np.shape(TrainingData))
        print(np.shape(TrainingLabels))
        
        ValidationData = []
        ValidationLabels = []
        for SingleClass in ClassName:
            for Root, Folder, File in os.walk(os.path.join(ValidationDir, SingleClass)):
                for Data in File:
                    if Data.split(".")[0] == "dog":
                        Img = Image.open(os.path.join(ValidationDir, SingleClass, Data))
                        Img = Img.resize(Dim, Image.ANTIALIAS)
                        ValidationData.append(np.array(Img))
                        ValidationLabels.append(0)
                    elif Data.split(".")[0] == "cat":
                        Img = Image.open(os.path.join(ValidationDir, SingleClass, Data))
                        Img = Img.resize(Dim, Image.ANTIALIAS)
                        ValidationData.append(np.array(Img))
                        ValidationLabels.append(1)
    
        # Convert the FeatureIndex to numpy array
        print("Below is the validation Data set details dimensions")
        ValidationData = np.array(ValidationData)
        ValidationLabels = np.array(ValidationLabels)            
        
        print(np.shape(ValidationData))
        print(np.shape(ValidationLabels))    
        
        # save the data points in the training and validation folder location
        # for quick access
        SaveTrainData = os.path.join(trainDir,"trainData.npy")
        SaveTrainLBl = os.path.join(trainDir,"trainlabel.npy")
        np.save(SaveTrainData,TrainingData)
        np.save(SaveTrainLBl, TrainingLabels)
        
        # save the validation data here
        SaveTestData =os.path.join(ValidationDir,"testData.npy")
        SaveTestLBl = os.path.join(ValidationDir,"testlabel.npy")
        np.save(SaveTestData,ValidationData)
        np.save(SaveTestLBl, ValidationLabels)
        
        return TrainingData, TrainingLabels, ValidationData, ValidationLabels
        
    elif PreData == True:
        print("Loading the already preprocessing numpy arrays.")
        print("Due to LoadPreprocessData set to True")
        TrainingData = np.load(os.path.join(trainDir, "trainData.npy"))
        TrainingLabels = np.load(os.path.join(trainDir, "trainlabel.npy"))
        
        ValidationData = np.load(os.path.join(ValidationDir, "testData.npy"))
        ValidationLabels = np.load(os.path.join(ValidationDir, "testlabel.npy"))
        
        print("Below is the training data set details dimensions")
        print(np.shape(TrainingData))
        print(np.shape(TrainingLabels))
        print("Below is the validation data set details dimensions")
        print(np.shape(ValidationData))
        print(np.shape(ValidationLabels))
        
        return  TrainingData, TrainingLabels, ValidationData, ValidationLabels
        
    else:
        print("Requried predata to either true or flase")
        

# Create the Model Here
def CNNModel(ClassName):
    model = models.Sequential()
    # 1st layer
    model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(64,64,3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    # 2nd layer
    model.add(Conv2D(64,(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    #3rd layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(ClassName),activation='softmax'))
    
    model.summary()
    
    return model
    

def PlotModelGraph(history):
    fig, axis = plt.subplots(2)
    axis[0].plot(history.history['accuracy'], label="train accuracy")
    axis[0].plot(history.history['val_accuracy'], label="validation accuracy")
    axis[0].set_ylabel("Accuracy")
    axis[0].legend(loc="lower right")
    axis[0].set_title("Model accuracy")
    
    
    axis[1].plot(history.history['loss'], label="train loss")
    axis[1].plot(history.history['val_loss'], label="validation loss")
    axis[1].set_ylabel("loss")
    axis[1].set_xlabel("epoch")
    axis[1].legend(loc="upper right")
    axis[1].set_title("Model accuracy")
    
    
    plt.show()

# Create the main and start with preprocessing
if __name__ == "__main__":
    Dim = (64,64)
    ClassName = ["dogs","cats"]
    LoadPreprocessData = True
    # Get the data and do preprocessing
    TrainData, TrainIndex, TestData, TestIndex = Preprocessing(Dim, ClassName, PreData=LoadPreprocessData) 
    
    # Load the model now
    model = CNNModel(ClassName)
    
    # Adding the optimizer for learning rate
    optimizer = optimizers.Adam(learning_rate=0.0001)
    # Now Compile the model.
    model.compile(optimizer=optimizer,
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    mc = callbacks.ModelCheckpoint('weights{epoch:08d}.h5', 
                                     save_weights_only=False, period=1)
    
    history = model.fit(TrainData, TrainIndex, 
                        validation_data=(TestData, TestIndex),epochs=20, 
                        batch_size=2,callbacks=[mc])
    # Plot the graph
    PlotModelGraph(history)

        
    
    
    