#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Swaminathan t
"""

from tensorflow.keras import models
from PIL import Image
import numpy as np


def Predict(FileName):
    dim=(64,64)
    className=["dog","cat"]
    model = models.load_model("weights00000013.h5")
    model.summary()
    # Read the file
    Img = Image.open(FileName)
    Img = Img.resize(dim, Image.ANTIALIAS)
    RawData = np.array(Img).reshape(-3,64,64,3)
    GetResult = model.predict(RawData)
    Result = np.argmax(GetResult)
    print("Image predicted as {}".format(className[Result]))



if __name__ == '__main__':
    FileName = "Cat_for_predict.jpeg"
    Predict(FileName)