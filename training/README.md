# Training the Multi-task learning
The purpose of this wrokshop is to learn how to train a Neural Network that does **real-time semantic segmentation and monocular depth prediction**.

The Model is [a Multi-Task Learning algorithm designed by Vladimir Nekrasov](https://arxiv.org/pdf/1809.04766.pdf). The entire work is based on the **DenseTorch Library**, that you can find and use [here](https://github.com/DrSleep/DenseTorch).

The **KITTI Dataset only has 200 examples of segmentation**. Therefore, the authors used a technique called Knowledge Distillation and finetuned using the Cityscape dataset.<p>

In our case, we'll use another dataset called the [NYUDv2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). **It contains 1449 annotated images for depth and segmentation**, which makes our life much simpler.
