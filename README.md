## A Traffic Sign Recognition Program

This is my third project of [Self-Driving Car Engineer nanodegree program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) in udacity.

You can see the first project at [this link](https://github.com/Akitsuyoshi/CarND-LaneLines-P1), second one is [here](https://github.com/Akitsuyoshi/CarND-Advanced-Lane-Lines).

## Table of Contents

- [A Traffic Sign Recognition Program](#a-traffic-sign-recognition-program)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
  - [0 Setting](#0-setting)
  - [1 Load and Explore the data set](#1-load-and-explore-the-data-set)
  - [2 Preprocess the data set](#2-preprocess-the-data-set)
  - [3 Design, train and test a model](#3-design-train-and-test-a-model)
  - [4 Use the model to make predictions on new images](#4-use-the-model-to-make-predictions-on-new-images)
  - [5 Analyze the softmax probabilities of the new images](#5-analyze-the-softmax-probabilities-of-the-new-images)
  - [6 Summary](#6-summary)
- [Discussion](#discussion)
  - [Problem during my implementation](#problem-during-my-implementation)
  - [Improvements to pipeline](#improvements-to-pipeline)
  - [Future Feature](#future-feature)
- [References](#references)
- [Issues](#issues)

---

## Overview

**The final exported HTML file can be found at [report.html](./report.html). Project code is [Traffic_Sign_Classifier.ipynb](./Traffic_Sign_Classifier.ipynb)**

The goals / steps of this project are the following:

- Load the data set, using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results

[//]: # (Image References)

[image1]: ./examples/hist.png "Histgram"
[image2]: ./examples/plot.png "Plot"
[image3]: ./examples/normalized_plot.png "Normalized"
[image4]: ./examples/augumented_hist.png "Augumented"
[image5]: ./examples/new_data.png "New Data"
[image6]: ./examples/normalized_new_data.png "Normalized New Data"
[image7]: ./examples/softmax.png "Softmax"

---

### 0 Setting

To set it up to run this script at first, I followd this [starter kit](https://github.com/udacity/CarND-Term1-Starter-Kit) with docker. If you use mac and docker was installed successfuly, you can run jupyter notebook on your local machine by the command below.

```sh
docker run -it --rm --entrypoint "/run.sh" -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit
```

This repo doesn't contain datasets, I got them from Udacity lessons.

### 1 Load and Explore the data set

I put image datasets at `./traffic-signs-data/`. Datasets serves as three purposes, for training, validating, and testing. Each data was named accordingly.

Here is datasets basic descriptions.

```sh
Number of training examples = 34799
Number of validating examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

And then I plot hitogram of the datasets so that I can see how many images each label(traffic signs) exits in that datas.

![alt text][image1]

Each images look like this:

![alt text][image2]

### 2 Preprocess the data set

At first, I decided to covert image data from RGB to Gray scale because it reduces datasize. The RGB channels values are in the `[0, 255]` range. This range is not ideal for a neaural network, in general small input values are better. I rescale it to `[0, 1]` instead for that reason. Above that steps, I first trid Gaussian Blur in preproce step, but I realized that blur doesn't work well for later traing step so I commet it out for now.

I got normalized imagedataset, and then I made some additional fake data especially for the one that aren't good enough for training model. Some are 2000 pics but some are even less than 200, like 180 pics. To deal with that issue, I do image augumentation. If specific image does't exits over 1000 pics, I add three fake data in three ways, like random cropping, vertical flipping, and cutting out. I use [this repo](https://github.com/xkumiyu/numpy-data-augmentation) as reference for implementing each augumentation. A detail cutting out process is found at [this paper](https://arxiv.org/abs/1708.04552).


To make sure that I can make additional data for less images, I plot histogram again. It shows that all each images exits over 700 at least.

![alt text][image4]

Here is images after normalization and augumentation.
![alt text][image3]

### 3 Design, train and test a model

My final LeNet model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Dropout   	      	| 0.85 remains for trainig, 1.0 for validating and testing   			   |
| Flatten Convolution   | input 5 * 5 * 16, outputs 400                 |
| Fully connected		| outputs 200  									|
| RELU				    |           									|
| Fully connected		| outputs 120									|
| RELU				    |            									|
| Fully connected		| outputs 86   									|
| RELU				    |           									|
| Fully connected	    | outputs 43(number of classes)					|
| Softmax				|												|

To train the defined model, I used an `Adam optimizer` with `learning rate = 0.005`, `batch size = 256`, and `15 Epocs`.
A bit tricky part is that I should pass different dropout rate for training from that for validating and testing. And label(class) must be made as one hot coded before I pass it to model.

The final accuracy from model are following:

```sh
Training Accuracy = 0.985
Validation Accuracy = 0.936
Test Accuracy = 0.936
```

Through training, I save previous validation accuracy at each epoch. If current epoch model make better accuracy than previous, that model was saved. I make assure that only best accuracy model was saved.

### 4 Use the model to make predictions on new images

I got 6 images below on the web. Image label id is described at [signnames.csv](./signnames.csv)

![alt text][image5]

New image labels are following:

- 12,Priority road + (37,Go straight or left)
- 36,Go straight or right
- 14,Stop
- 26,Traffic signals
- 25,Road work
- 31,Wild animals crossing

I then applied same normalizing pipeline to new images. In addition to normalization, I make all of them be the same shape size, `(32, 32, 3)` as that of training image.

Here is the output:

![alt text][image6]

Above image shows actual label id and prediction below the picture.

It shows that the model predicted 1 out of 6 signs correctly, it's 16.7% accurate on these new images.

### 5 Analyze the softmax probabilities of the new images

To get softmax probabilities for each new images, I use `tf.nn.top_k`.

Here is output for each images.

![alt text][image7]

### 6 Summary

**The final exported HTML file can be found at [report.html](./report.html).**

The accuracy on new test images is bad. There are some reasons. One possible reason is that I tested images that are less number in traing datasets. The model doesn't trained thosed images well because of its little number.

## Discussion

### Problem during my implementation

After I decided model architecture at first, I got 85% validation accuracy. I changed hypter parameters like learning rate to increase accuracy but it didn't well. At that time, it shows lows accuracy both for traing for validating so I need additional data in training set. I decided to do image augumentation. And then I got a bit better result from 85% to 90%. I might be wrong, but I kinda noticed that formatting datasets, including normalization and image augumentations, affects much more that changing model architecture.

I resove underfitting, and then overfitting came next. A traing accuracy at that time increased, whereas validation accuracy got stuck at 80% or something. I adjust model to add dropout layer to deal with that overfitting. With added dropout, model works well but not next to expected 93%. I decided to implement data augumentatoin. I first made mistake that I make fake data to all images but it didn't change result. The problem here is that there are small number of traing datasets for some images, but not for all images. I made 93.6 validation accuracy after implementing data augumentation and drop layer.

I should have defined model and trained it first, and then I did some pre process step for datasets. I plaied aroud image augumentation process before training step, but the thing is how well my model classified datasets by seeing differences between training and valitating accuracy.

### Improvements to pipeline

A model accuracy for new images is low for sure. I might have better image proprocess way that current one.

Current normalized images are bit too vague for human eyes.
![alt text][image6]

Another possible suggestion is to make all of each image in training datasets exist the same number, like 2000 for example.

### Future Feature

I will try to test another model architecture apart from LeNet.
I train the model on Udacity workspace, but in actual situation, it should be on some clud vendor or own gpu server. I try this progect out on [AWS](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/AWS+Instructions.pdf). Last challenge still remains, so I do `Step 4 (Optional): Visualize the Neural Network's State with Test Images` on project code.

---

## References

- [Set up for training model on AWS GPU instances](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/AWS+Instructions.pdf)
- [Image augumentation using Numpy](https://github.com/xkumiyu/numpy-data-augmentation)
- [Image Classification using TensorFlow](https://www.tensorflow.org/tutorials/images/classification)

---

## Issues

Feel free to submit issues and enhancement requests.
If you see some bugs in here, you can contact me at my [twitter account](https://twitter.com/).

