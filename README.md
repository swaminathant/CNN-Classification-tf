# CNN-Classification-tf
[Dog, Cat] Classification using tensorflow. 

========================================================================

OS : Ubuntu 20.04.1 LTS

GPU : Nvidia 920M (2GB Memory)
training minutes : 10 minutes

CPU : Intel Core i3-4005U
Ram : 2GB utilized  / 12 GB total memory
training minutes : 25 minutes

IDE editor : spyder

========================================================================

I have collected the data of 2000 images for training and 1000 images 
 - training images
	- 1000 Dog images
	- 1000 cat images
 - validation images
	- 500 Dog images
	- 500 cat images

 - Labels:
	- [0] Dog
	- [1] Cat 

=========================================================================

Required modules.
	os - to read the directory structure
	numpy - convert the image to multi dimentional array
	PIL - Used for reading the images and store as numpy array
	matplotlib - Used for visualize the train and validatiion accuracy on each epochs
	tensorflow - To build our model, train and prediction

=========================================================================

Here I have uploaded already preprocessed images to numpy array which ready for training and testing.
	- All images are converted into numpy array as 64x64x3 dimension
		- 64x64 width and height
		- 3 is the (RGB) channels
	- numpy array located under the subfolder of cats_and_dogs/
		-train
			- cats_and_dogs/train/trainData.npy
			- cats_and_dogs/train/trainlabel.npy
		-validation
			- cats_and_dogs/validation/testData.npy
			- cats_and_dogs/validation/testlabel.npy

==========================================================================

Python Files:
	- CNNDogCatClassifier.py 
		- required for preprocessing, training and train/validation accuracy visualization.
		- Flags
			Line no : 156  LoadPreprocessData = True (for using the existing preprocess numpy data)
			Changing to False will read the images under the directories.
		- Each weights will be stored in same dir and we can choose the best weights for prediction.py by 
		  viewing the visualization.
	- Prediction.py
		- required for predicting the cat / dog images
		- Place the cat or dog images on same directory where the Prediction.py located.
		- Prediction.py use the weight weights00000013.h5 which is already training by the CNNDogCatClassifier.py

==========================================================================
