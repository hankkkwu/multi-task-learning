# Multi-task Learning
The goal of this project is to learn how to build a Neural Network that has:
* Input: **a monocular RGB Image**
* Output: **a Depth Map**, and **a Segmentation Map**

A single model, two different outputs. For that, the model will need to use a principle called Multi Task Learning. To do that, I define the model from the paper [Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations](https://arxiv.org/pdf/1809.04766.pdf), which takes an input RGB image, make it go through an encoder(MobileNetV2), and a lightweight refinenet as decoder, and then has 2 heads, one for each task.

Here is the result on video1:

![multi-taks1](https://github.com/hankkkwu/multi-task-learning/blob/main/outputs/Residential.gif)


Here is the result on video2:

![multi-taks2](https://github.com/hankkkwu/multi-task-learning/blob/main/outputs/campus.gif)
