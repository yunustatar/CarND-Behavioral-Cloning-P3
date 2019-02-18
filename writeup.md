#**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network, using data collected by driving on both tracks
* writeup_report.md (this file) summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. 

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used a similar model I used in Traffic Sign Classification project with slight changes and improvements. 
The main difference is the preprocessing layers, the output layer, and number of dropout layers used.

My final model is presented in the section "Final Model Architecture"
 
####2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py line 131). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py lines 101, 144).

####4. Appropriate training data

I collected data from both tracks to generalize the model. I've run 3 laps on the first track, 2 more laps on the first track reverse direction, and 2 laps on the second track. I've also collected 1 lap of "recovery" lap data from the first track.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with simple fully connected layers and add more complexity to overfit for the 20% of the data gathered first. This was the same approach I had used in the traffic sign classiffier project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
I added a dropout layer after the flatten layer to overcome the overfitting.

At the end of the process, I tested my model using the test track. I've confirmed that he vehicle is able to drive autonomously around the first track without leaving the road. I've also tested my model on the second track, even though my model was able to drive the car without leaving the road most of the track, close to the end the car actually left the road. I've included the videos for both first track driving as well as the second track. 

####2. Final Model Architecture

The final model architecture (model.py lines 104-144) consisted of a convolution neural network presented below.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						|
| Cropping Layer        | Output = 65x320x3 RGB image                   |
| Normalize&Centralize  | Output : pixel values normalized to [-0.5,0.5]|
| Resize layer          | Output = 32x32x3 image with normalized pixels |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32	|
| RELU		            |              									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32   				|
| Flatten				| output = 800									|
| Dropout               | used 0.5 keep_prob while training             |
| Fully Connected       | input = 800 output = 400                      |
| RELU		            |              									|
| Fully Connected       | input = 400 output = 200                      |
| RELU		            |              									|
| Fully Connected       | input = 200 output = 1 (steering ang)         |

####3. Creation of the Training Set & Training Process

Ive collected 3 laps of data from the first track, 2 more laps on the first track by going backwards, and 2 laps from second track by drivng in the center as much as possible.

After the collection process, I had about 25k number of data points. I then preprocessed this data by using only 10% of data with no steering, as we don't use the steering in most of the track.

To augment the data sat, I also flipped images and angles which created extra data for model to learn.

After randomly shuffling and using 20% of the data as the validation set, I trained the model, using the processes I described in previous sections. (adam optimizer, EPOCHS = 3) 

Initially, I did not collect any data for "recovery" and the model did not do very good without this data. Then I collected some recovery data by getting close to/or out of the lines, and then start recording data and get back into track. I've done this on both sides of the track so that my model could learn how to recover from both sides. This made a huge difference, and the model was working significantly better after that.
