# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[training_set_examples]: ./results/set_examples.png "training_set_examples"
[training_examples_distribution]: ./results/histogram.png "training_examples_distribution"
[raw_image]: ./results/raw_image.png "raw_image"
[normalized_image]: ./results/normalized_image.png "normalized_image"
[gray_image]: ./results/gray.png "gray_image"
[images_from_the_internet]: ./results/images_from_the_internet.png "images_from_the_internet"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
## Writeup

### Link to [project code](https://github.com/andriikushch/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

#### 2. Exploratory visualization of the dataset.

Here are the examples of the images from the training set:

![alt text][training_set_examples]

Here is an example of classes distribution from the training set:

![alt text][training_examples_distribution]

### Design and Test a Model Architecture

#### 1. Data preprocessing

As a first step, I decided to convert the images to grayscale. It reduces the dimesnion of an input and makes normalization of the image easier.

At next step I am normalizing the data in order to decrease complexity and potentialy increase network acuracy and learning speed.


| Raw  | Gray  | Normalized  | 
|:-:|:-:|:-:|
| ![alt text][raw_image]  | ![alt text][gray_image]  | ![alt text][normalized_image]  |



#### 2. Model structure

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| 1. Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| 2. Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16    									|
| 3. Fully connected		| input 5x5x16, output 400   									|
| RELU				|         									|
| 4. Fully connected	|	imput 120, output 84											|
|	RELU					|				|
| 5.  Fully Connected  |	input = 84. Output = 43			|
 


#### 3. How model is trained


To train the model, I used an `AdamOptimizer`, with `rate = 0.001`, `EPOCHS = 20`, `BATCH_SIZE = 128`.

AdamOptimizer is known as computationally efficient, low memory consuming and easy to tune optimizer. Rate 

`rate` and `BATCH_SIZE` were selected via trial and error method. Smaller `rate` makes learning too slow, high one doesn't converge well. 

`EPOCHS` was choosen to achive necessary precision.

#### 4. The approach.

My final model results were:

* training set accuracy of `0.999`
* validation set accuracy of `0.945`
* test set accuracy of `0.933`

Some thoughts about the architecture:

* Current `LaNet` architecture was one I knew from previous cource lessons and it did it's job.
* For current architecture was quite tricky to find proper hypeparameters and normalize input.
* I tuned the `EPOCHS` parameter, increased it to 20 to reach learning threshold. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report.

Here are five German traffic signs that I found on the web:

![alt text][images_from_the_internet]


#### 2. Result of prediction.


| Image			        |     Prediction (top 5 softmax probabilities)	        					| 
|:---------------------:|:---------------------------------------------:| 
|Speed limit (30km/h)   |                  [9.9999928e-01 4.6627065e-07 1.8073200e-07 1.1348734e-08 4.4676702e-12] |
|General caution        |                  [1.0000000e+00 1.7135681e-08 4.0559920e-09 3.0214271e-09 1.0853046e-09] |
|Priority road          |                  [1.0000000e+00 7.0696413e-24 1.4433812e-24 1.0934178e-25 5.1488289e-26] |
|No entry               |                  [1.0000000e+00 5.7582330e-20 2.2768636e-24 8.4445566e-30 7.0737169e-31] |
|Road work              |                  [1.0000000e+00 3.8328948e-15 6.1613616e-16 3.8493330e-16 7.0987985e-17] |
|End of all speed and passing limits |     [9.9997044e-01 2.9554330e-05 2.5264612e-08 2.8319382e-11 8.3975883e-14] |
|Stop                   |                  [1.0000000e+00 9.8190428e-17 3.0260198e-19 1.7365903e-19 1.4812312e-19] |
|Keep right             |                  [1.0000000e+00 8.9166329e-25 7.2660056e-32 4.0964350e-32 1.1183832e-32] |

The model was able to correctly guess all the signs, which gives an accuracy of 100%. This is expected result, because newly found images are clear, without noise and have a good lighting conditions. Taking into account that model shown `> 93%` accuracy on the test set, there should be no problem to classify these images. 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


