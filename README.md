# Scalar Autograd Engine

A neural network and autograd engine built completely from scratch in Python without using deep learning frameworks.

## Features

* Scalar-based autograd engine
* Backpropagation implementation
* Custom MLP implementation
* XOR training
* Binary cross entropy loss
* Cross entropy for multiclass classification
* Momentum optimizer
* Adam optimizer
* L2 regularization
* MNIST multiclass classification experiment

## Current Limitation

The engine is scalar-based, so training becomes extremely slow for larger datasets like MNIST.

Current benchmark:

* MNIST (50 samples)
* ~1.45 minutes per epoch

## Goal

Next step is building a vectorized tensor-based engine for better performance and scalability.

## Learning Purpose

This project was built to deeply understand:

* computational graphs
* backpropagation
* optimizer internals
* gradient flow
* performance bottlenecks in deep learning systems
