# From Scalar Autograd to Tensor Autograd

A deep learning project built completely from scratch in Python to understand how modern neural network frameworks work under the hood.

This repository documents the journey from a scalar-based autograd engine to a tensor-based autograd system capable of training neural networks on MNIST without using PyTorch or TensorFlow.
## for reference go through learning.ipynb
## Phase 1: Scalar Autograd Engine

Implemented a micrograd-style automatic differentiation engine from scratch.

### Features

* Scalar computational graph
* Automatic differentiation
* Backpropagation
* Custom MLP implementation
* XOR training
* Binary Cross Entropy Loss
* Cross Entropy Loss
* Momentum Optimizer
* Adam Optimizer
* L2 Regularization

### Key Learning

While the scalar engine successfully trained small neural networks, it exposed an important bottleneck when applied to larger datasets such as MNIST.

### Limitation

* Every number becomes a separate graph node
* Thousands of Value objects created during training
* Poor scalability for matrix operations
* MNIST training became extremely slow

Benchmark:

* MNIST (50 samples)
* ~1.45 minutes per epoch

---

## Phase 2: Vectorized Neural Networks

To overcome the scalar bottleneck, I reimplemented neural networks using NumPy vectorization.

### Implemented

* Matrix-based forward propagation
* ReLU activation
* Softmax
* Cross Entropy Loss
* Fully vectorized backpropagation
* MNIST classification

### Key Learning

This phase helped build intuition for:

* Matrix calculus
* Tensor shapes
* Gradient flow through matrices
* Efficient neural network computation

---

## Phase 3: Tensor Autograd Engine

Built a tensor-based automatic differentiation engine from scratch.

### Supported Operations

* Addition (+)
* Multiplication (*)
* Matrix Multiplication (@)
* Division (/)
* Power (**)
* ReLU
* Sum
* Mean
* Exponential
* Logarithm

### Engine Components

* Computational graph construction
* Topological sorting
* Automatic backpropagation
* Gradient accumulation
* Tensor-based gradient computation

---

## Phase 4: Tensor MLP

Built a neural network framework on top of the tensor engine.

### Components

* Linear Layer
* MLP Architecture
* Parameter Management
* Automatic Gradient Updates

### Result

Successfully trained an MLP on MNIST using:

```python
loss.backward()
```

without manually deriving gradients.

---

## Key Concepts Learned

* Computational Graphs
* Chain Rule
* Automatic Differentiation
* Matrix Backpropagation
* Broadcasting
* Tensor Operations
* Neural Network Optimization
* Performance Bottlenecks in Deep Learning Systems

---

## Future Work

* CNNs from scratch
* Convolution backpropagation
* PyTorch implementation comparisons
* GPU acceleration concepts
* Transformer architectures
* Large Language Models

## Motivation

The goal of this project is not to replace existing frameworks, but to deeply understand the engineering and mathematics behind modern deep learning systems.

Building both scalar and tensor autograd engines provided a much deeper understanding of what actually happens when we call:

```python
loss.backward()
```
