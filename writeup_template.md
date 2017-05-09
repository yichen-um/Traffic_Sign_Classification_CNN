#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
###Writeup / README

This file provides a description to a computer vision project to classify traffic sign. The entire code can be found via this link [project code](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is square consist of 32 by 32 pixels
* The number of unique classes (labels) in the data set is 43, including training, validation and testing data set

####2. Exploratory visualization of the dataset.
Below is a depict of examples randomly selected from each class in the training data. Notice that lighting condition, the orientation of the traffic sign and the background has varies. 

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/DataSetExamples.png)

To visualize the data set, a bar chart showing how the number of examples are distributed among different classes is given as below

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/DataSetVisual.png)

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Several computer vision techniques are used to pre-process the image, including applying gamma correction to increase the brightness of the pictures, histogram equalization to normalize the contrast and brightness, and convert the image to grayscale and normalize the pixel values between 0 to 1. The grayscaling of the images reduces the parameters need to be trained without sacrifysing accuracy [LeCun, 2012]

Here are two examples of traffic sign images before and after pre-processing.

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/preProcessDemo8150.png)

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/preProcessDemo8109.png)

In order to train a classifier that is robust in classify all types of traffic signs, the discrepancy in the distributions of examples among the training and validation set are compensated by generating augmented data. Specifically, the following processing steps are taken

a. The class under represented are selected, and a number of compensation is calculate
b. Examples are randomly selected from under represented class
c. Additional data are produced by rotating selected examples by -15 to 15 degree

Here is an example of an original image and an augmented image:

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/AugDataDemo.png)

The distribution of the original data set and the augmented data set are shown below:

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/AugTrainDist.png)

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/AugValidDist.png)

####2. Description of final model architecture

The first architecture was chosen as the LeNet, which was provided in the Udacity material. The initial validation accuracy was decent but not enough for 93 % threshold due to its insufficent parameters. During iteratively tuning process, a more sophisticated network given above was adopted with more layers and weighting parameters. After the augmented data added into the data set, the training process became longer and the validation accuracy is improved. 

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 3x3     	| 5x5 stride, valid padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | 5x5 stride, valid padding, outputs 32x32x64   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Fully connected		| etc.        									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|

####3. Description of meodel training

The model was trained via  an amazon EC2 GPU instance. The optimizer is chose to be Adam Optimizer, dropout technique is used to prevent overfitting. Below is the parameters used in training:

* leanring rate = 0.0005
* batch size = 128
* drop rate = 50 %
* ipochs = 40

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

The accuracy and loss of training and validation during the traing process are shown below. We can see that the accuracy on training and validation data set are off by margine of X %, and reaches to steady after X epoch. 
![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/TrainingHistory.png)

The final model results are:
* training set accuracy = 98.4 %
* validation set accuracy =  92.7 %
* test set accuracy = 91.9 %

###Test a Model on New Images

####1. To further test the model, 9 German traffic signs are found on the web and tested, the results are presented and discussed below. 

Here are 9 German traffic signs that were selected from web, with correct class labeled:

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/NewTestData.png)

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      		| Priority road   									| 
| Speed limit (30km/h)     			| Speed limit (30km/h) 										|
| Road work					| Road work											|
| General caution	      		| General caution					 				|
| Right-of-way at the next intersection			| Right-of-way at the next intersection      							|
| Turn right ahead	      		| Turn right ahead					 				|
| Ahead only	      		| Aheadd only					 				|
| Keep right	      		| Keep right					 				|
| Vehicles over 3.5 metric tons prohibited	      		| Vehicles over 3.5 metric tons prohibited					 				|

The model was able to correctly guess 9 of the 9 traffic signs, which gives an accuracy of 100 %. This compares favorably to the accuracy on the test set of 91.9 %.

####3. To describe how certain the model is when predicting on each of the new images, the top 5 softmax probabilities for each prediction are listed under each new testing figure. 

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/TopKDemo.png)


