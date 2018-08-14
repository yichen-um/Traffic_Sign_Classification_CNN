# Project: Traffic Sign Classification
### Udacity Self-Driving Car Engineer Nanodegree

---
## 1. Summary of Project
This project aims to build a classifier that is able to classify traffic sign with advanced computer vision technique and deep neural net work model. The classifier is trained, validated and tested with over hundreds of thousands of examples among 43 classes of traffic signs from the data base of German Traffic Signs Recognition Benchmark [(GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The code for the project can be found [here](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/Traffic_Sign_Classifier.ipynb). More technical details can be found at in the reference listed as:
[Sermanet and LeCun, 2011](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
[Ciresan et al., 2012](http://xa.yimg.com/kq/groups/14962965/499730234/name/NN_PAPER.pdf)

This project includes the following procedures:
* Explore, summarize and visualize the data set
* Augment data to balance distribution of training and validation data set
* Design, train and test a deep neural net work model
* Use the model to make predictions on new images and analyze the softmax probabilities
---
## 2. Data Set Summary & Exploration
### 2.1. Summary of the data set. 
I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is square consist of 32 by 32 pixels
* The number of unique classes (labels) in the data set is 43, including training, validation and testing data set

### 2.2. Exploratory visualization of the dataset.
Below is a depict of examples randomly selected from each class in the training data. Notice that lighting condition, the orientation of the traffic sign and the background has varies.

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/DataSetVisual.png)

To visualize the data set, a bar chart showing how the number of examples are distributed among different classes is given as below

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/DataSetDistribution.png)

## 3. Design and Test a Model Architecture
### 3.1. Description of pre-processing 
Several computer vision techniques are used to pre-process the image, including applying gamma correction to increase the brightness of the pictures, histogram equalization to normalize the contrast and brightness, and convert the image to grayscale and normalize the pixel values between 0 to 1. The grayscaling of the images reduces the parameters need to be trained without sacrifysing accuracy [LeCun, 2012]
Here are two examples of traffic sign images before and after pre-processing.

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/preProcessDemo8150.png)
![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/preProcessDemo8109.png)

### 3.2. Description of data augmentation
In order to train a classifier that is robust in classify all types of traffic signs, the discrepancy in the distributions of examples among the training and validation set are compensated by generating augmented data. Specifically, the following processing steps are taken
* Combine the original training and validation data set
* Select the classes underrepresented in terms of number of examples
* Add additional data for the underrepresented by rotating randomly selected examples by -5 to 5 degree
Here is an example of an original image and an augmented image:

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/AugDemo7991.png)

The distribution of the augmented data set are shown below:

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/AugDataDist.png)

* The size of augmented training set is 46510
* The size of augmented validation set is 11628
* The test set is not augmented and the size remains 12630
### 3.3. Description of final model architecture
The first architecture was chosen as the LeNet, which was provided in the Udacity material. The initial validation accuracy was decent but not enough for 93 % threshold due to its insufficent parameters. During iteratively tuning process, a more sophisticated network given above was adopted with more layers and weighting parameters. After the augmented data added into the data set, the training process became longer and the validation accuracy is improved. 

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x70 	|
| RELU					|												|
| Max pooling	2x2      	| 2x2 stride,  outputs 14x14x70 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x140   |
| RELU					|												|
| Max pooling	2x2      	| 2x2 stride,  outputs 5x5x140 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 3x3x210   |
| RELU					|												|
| Fully connected		| outputs 1890x1        									|
| Fully connected		| outputs 300x1        									|
| Softmax				| outputs 43x1        									|

### 3.4. Description of model training and tuning of hyper-parameter
The model was trained via  an amazon EC2 GPU instance. The optimizer is chose to be Adam Optimizer, dropout technique is used to prevent overfitting. Below is the parameters used in training:
* leanring rate = 0.00001
* batch size = 128
* drop rate = 20 %
* ipochs = 50
The accuracy and loss of training and validation during the traing process are shown below. We can see that the accuracy on training and validation data set are off by margine of X %, and reaches to steady after X epoch.

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/TrainingHistory.png)

The final model results are:
* training set accuracy = 96.4 %
* validation set accuracy =  95.8 %
* test set accuracy = 92.4 %

## 4. Test a Model on New Images
### 4.1. New testing images
To further test the model, 9 German traffic signs are found on the web and tested, the results are presented and discussed below. 
Here are 9 German traffic signs that were selected from web, with correct class labeled:

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/NewTestData.png)

The first image might be difficult to classify because ...

### 4.2. Discussion on prediction of new testing images
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

For demonstrative purpose, the model is able to correctly guess 9 of the 9 traffic signs, which gives an accuracy of 100 %. This compares favorably to the accuracy on the test set of 91.9 %.

### 4.3. Top 5 predictions
To describe how certain the model is when predicting on each of the new images, the top 5 softmax probabilities for each prediction are listed under each new testing figure.

![alt text](https://github.com/davidsky900/SelfDrivingCar-TrafficSign/blob/master/examples/TopKDemo.png)


