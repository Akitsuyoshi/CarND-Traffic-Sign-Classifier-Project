## A Traffic Sign Recognition Program

This is my third project of [Self-Driving Car Engineer nanodegree program](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) in udacity.

You can see the first project at [this link](https://github.com/Akitsuyoshi/CarND-LaneLines-P1), second one is [here](https://github.com/Akitsuyoshi/CarND-Advanced-Lane-Lines).

## Table of Contents

- [A Traffic Sign Recognition Program](#a-traffic-sign-recognition-program)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
  - [0 Setting](#0-setting)
  - [1 Load and Explore the data set](#1-load-and-explore-the-data-set)
  - [2 Design, train and test a model](#2-design-train-and-test-a-model)
  - [3 Use the model to make predictions on new images](#3-use-the-model-to-make-predictions-on-new-images)
  - [4 Analyze the softmax probabilities of the new images](#4-analyze-the-softmax-probabilities-of-the-new-images)
  - [5 Summary](#5-summary)
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

---

### 0 Setting

To set it up to run this script at first, I followd this [starter kit](https://github.com/udacity/CarND-Term1-Starter-Kit) with docker. If you use mac and docker was installed successfuly, you can run jupyter notebook on your local machine by the command below.

```sh
docker run -it --rm --entrypoint "/run.sh" -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit
```

This repo doesn't contain datasets, I got them from Udacity lessons.

### 1 Load and Explore the data set

asdf

### 2 Design, train and test a model

### 3 Use the model to make predictions on new images

### 4 Analyze the softmax probabilities of the new images

### 5 Summary

## Discussion

### Problem during my implementation

When I first saved output image from each function, like undistortion, I resized it's shape. However, I realized that input image, processing image, and output image need to be the same shape. I got some error from my resizing. In reality, I also need to check image shape before any process maybe.

Its a small pisky bug that I faced, but cv2.imread is BGR image, and sometimes another functions process RGB image. I got some color problem due to that difference.

When implementing video pipeline, I noticed that some of my src and dst points definition didn't work as expected. Especially, curve or shadow time, it likely to fail. I change src accordingly, but it is still written in a hard code way. I couldn't figure it out to get those points correctly.

In the part of getting curvature, I still get confused about the way to calculate it. I mostly write my code from udacity lesson code, but not sure its formula.

Since it's kinda new for me to write code in python, I always put pring statement for debugging code. Might be anotehr better way that that.

### Improvements to pipeline

My current pipeline likely to fail tracking when show or strong light appeans in image. To deal with that, One possible solution would change color threshold.

Current src and dst points are written in a hardcode way. Especially src points, it's better to adjust to the image, not only by its shape but also somthing like lane condition.

### Future Feature

I will try to test my pipeline in two challenge video. I only test for [project_video.mp4](./project_video.mp4) for now.

---

## References

- [Set up for training model on AWS GPU instances](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/AWS+Instructions.pdf)
- [Image augumentation using Numpy](https://github.com/xkumiyu/numpy-data-augmentation)
- [Image Classification using TensorFlow](https://www.tensorflow.org/tutorials/images/classification)

---

## Issues

Feel free to submit issues and enhancement requests.
If you see some bugs in here, you can contact me at my [twitter account](https://twitter.com/).

