# Exudation-of-Eye
Using UNet for detection of Exudation. Medical Imaging


## Prerequisites
* Python
* Keras (with tensorflow-gpu preffered)
* Opencv,Numpy,Pandas

## Test and Train dataset
Fundus images of Retina.
Train - 191 Test - 25
Preprocessing includes finding contours/radius of the eyeballs and normalizing the dataset to that radius.  
Augmentation on the dataset includes rotation, gamma, colour shift and contrast.

## CNN Model and Performance
Used UNet for detecting exudations in fundus images.
Performance of dice coefficient close to 0.45-0.6 respectively for training and testing dataset and the accuracy on the test data is close to 94% on the testing data[binary classification accuracy of existence of exudation in the retina images].

