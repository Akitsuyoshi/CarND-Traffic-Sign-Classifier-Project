## A Traffic Sign Recognition Program

This is my third project of [Self-Driving Car Engineer nanodegree program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) in udacity.

You can see the first project at [this link](https://github.com/Akitsuyoshi/CarND-LaneLines-P1), the second one is [here](https://github.com/Akitsuyoshi/CarND-Advanced-Lane-Lines).

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
- [Author](#author)

---

## Overview

**The final exported HTML file can be found at [report.html](./report.html). Project code is [Traffic_Sign_Classifier.ipynb](./Traffic_Sign_Classifier.ipynb)**

The goals/steps of this project are the following:

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

To set it up to run this script at first, I followed this [starter kit](https://github.com/udacity/CarND-Term1-Starter-Kit) with docker. If you use mac and docker was installed successfully, you can run jupyter notebook on your local machine by the command below.

```sh
docker run -it --rm --entrypoint "/run.sh" -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit
```

Note: This repo doesn't contain datasets.

### 1 Load and Explore the data set

I put image datasets at `./traffic-signs-data/`. Datasets are split into three sets, training, validating, and testing.

Here are the basic dataset descriptions.

```sh
Number of training examples = 34799
Number of validating examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

Plotting the histogram of the datasets so that I can see how many images each label(traffic signs) exists.

![alt text][image1]

Each images look like this:

![alt text][image2]

### 2 Preprocess the data set

At first, I decided to convert image data from RGB to Grayscale because it reduces data size. The RGB channel values are in the `[0, 255]` range. This range is not ideal for a neural network, in general, small input values are better. I rescale it to `[0, 1]` instead for that reason. Above that steps, I first tried Gaussian Blur in the preprocess step, but I realized that blur doesn't work well for later training steps so I comment it out for now.

I got a normalized image dataset, and then I made some additional fake data especially for the ones that aren't good enough for training the model. Some are 2000 pics but some are even less than 200, like 180 pics. To deal with that issue, I do image augmentation. If a specific image doesn't exit over 1000 pics, I add three fake data in three ways, like random cropping, vertical flipping, and cutting out. I use [this repo](https://github.com/xkumiyu/numpy-data-augmentation) as a reference for implementing each augmentation. A detailed cutting out process is found in [this paper](https://arxiv.org/abs/1708.04552).


To make sure that I can make additional data for fewer images, I plot the histogram again. It shows that each image exit over 700 at least.

![alt text][image4]

Here are images after normalization and augmentation.
![alt text][image3]

### 3 Design, train and test a model

My final LeNet model consisted of the following layers:

| Layer             |     Description                   |
|:---------------------:|:---------------------------------------------:|
| Input             | 32x32x3 RGB image                 |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x6  |
| RELU          |                       |
| Max pooling         | 2x2 stride,  outputs 14x14x6          |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU          |                       |
| Max pooling         | 2x2 stride,  outputs 5x5x16           |
| Dropout             | 0.85 remains for trainig, 1.0 for validating and testing           |
| Flatten Convolution   | input 5 * 5 * 16, outputs 400                 |
| Fully connected   | outputs 200                   |
| RELU            |                             |
| Fully connected   | outputs 120                 |
| RELU            |                             |
| Fully connected   | outputs 86                    |
| RELU            |                             |
| Fully connected     | outputs 43(number of classes)         |
| Softmax       |                       |

To train the defined model, I used an `Adam optimizer` with `learning rate = 0.005`, `batch size = 256`, and `15 Epocs`.
A bit tricky part is that I should pass a different dropout rate for training from that for validating and testing. And label(class) must be made as one hot coded before I pass it to the model.

The final accuracy from model are following:

```sh
Training Accuracy = 0.985
Validation Accuracy = 0.936
Test Accuracy = 0.936
```

Through training, I save previous validation accuracy at each epoch. If a current epoch model makes better accuracy than the previous, that model was saved. I make assure only the best accuracy model was saved at every epoch.

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

I then applied the same normalizing pipeline to new images. In addition to normalization, I make all of them be the same shape size, `(32, 32, 3)` as that of the training image.

Here is the output:

![alt text][image6]

The above image shows the actual label id and prediction below the picture.

It shows that the model predicted 1 out of 6 signs correctly, it's 16.7% accurate on these new images Too bad prediction on new datasets, 😢.

### 5 Analyze the softmax probabilities of the new images

To get softmax probabilities(likelyhood prediction) for each new image, I use `tf.nn.top_k`.

Here is the output for that probabilities.

![alt text][image7]

### 6 Summary

**The final exported HTML file can be found at [report.html](./report.html).**

The accuracy of the new test images is bad. There are some reasons. One possible reason is that I tested images that are fewer amount of training datasets. I found the current model doesn't work well on new datasets.

## Discussion

### Problem during my implementation

After I decided on model architecture at first, I got 85% validation accuracy. I changed hyperparameters like learning rate to increase accuracy but it didn't work well. At that time, it shows low accuracy both for training and validating sets so I need additional data in the training set. I decided to do image augmentation. And then I got a bit better result from 85% to 90%. I might be wrong, but I kinda noticed that formatting datasets, including normalization and image augmentations, affects much more than changing a whole model architecture, in my case at least.

After resolved underfitting problem, and then overfitting came to me. A training accuracy at that time increased, whereas validation accuracy got stuck or decrease sometime at 80% accuracy. I adjust the model to add one dropout layer to deal with that overfitting. With added dropout, the model works well but it didn't go over 93%, project criteria. I decided to implement data augmentation. I first made the mistake that I made fake data to all images but it didn't change the result. The problem here is that there is a small number of training datasets for some images, but not for all images. I just need to do image augmentation only for a small amount of image on training sett. I finally made 93.6 validation accuracy after the above processing.

I should have defined the model and trained it first, and then I did some preprocess steps for datasets. I played around with the image augmentation process before the training step, but the thing is how well my model classified datasets by seeing differences between training and validating accuracy. The problem here was underfitting or overfitting needs to be deal with after it appears actually. I didn't need to tackle those beforehand.

### Improvements to pipeline

A model accuracy for new images is low for sure. I might have a better image preprocess way than the current one.

Current normalized images are a bit too vague for human eyes.
![alt text][image6]

Another possible suggestion is to make all of each image in training datasets exist the same number, like 2000 for example.

### Future Feature

I will try to test another model architecture apart from LeNet.
I train the model on Udacity workspace, but in the actual situation, it should be on some cloud vendor or own GPU server. I try this project notebook out on [AWS](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/AWS+Instructions.pdf). Last challenge remains, so I do `Step 4 (Optional): Visualize the Neural Network's State with Test Images` on project code.

---

## Author

- [Tsuyoshi Akiyama](https://github.com/Akitsuyoshi)
