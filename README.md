# Pytorch Vision Template

This is a template repo for computer vision experiments in Pytorch. 

A somewhat more than barebones template that implements automatic logging, multi-GPU training, some standard datasets, and some standard models out of the box. 
This repo is intented to speed up the early experimental phase of computer vision deep learning research by setting up the essential framework for in-depth experiments out the gate. 


Random seeds may be set easily. By default the seed is set to 1 for easy experimental reproduction. 

The usual training hyper-parameters such as
 - optimizer 
 - learning rate
 - learning rate scheduling
 - batch size
 - momentum (if applicable)
 - gradient clipping
 - weight decay
 - early stopping

are implemented.

Some standard models are implemented
 - Resnet 18/50/152 (pretrained on imagenet available through Pytorch)
 - LeNet 5 /300-100 
 - A basic 3 layer MLP
 - A simple CNN
 
 
 Some standard datasets are supported
  - MNIST
  - FashionMNIST
  - CIFAR10 / 100
  - Imagenet (data not provided, loader support is)
  - Tiny Imagenet (data not provided, loader support is)
