## Broadcasting and Backpropagation

Currently, the engine has a limitation regarding **broadcasting**.

Broadcasting is a fundamental mechanism used in deep learning frameworks such as PyTorch, TensorFlow, and JAX. It allows arithmetic operations between tensors of different shapes without explicitly replicating data.

For example:

```python
X.shape = (64, 784)
b.shape = (1, 784)

Y = X + b
```

During the forward pass, the bias tensor is automatically broadcast across the batch dimension:

```text
(1, 784)
    ↓
(64, 784)
```

This allows the addition operation to be performed element-wise.

### The Backward Pass Problem

Suppose the gradient flowing from the next layer is:

```python
dY.shape = (64, 784)
```

The output gradient has the same shape as `Y`.

However, the original bias parameter has shape:

```python
b.shape = (1, 784)
```

A gradient with shape `(64, 784)` cannot be directly accumulated into a tensor with shape `(1, 784)`.

To obtain the correct gradient for `b`, we must reverse the broadcasting operation performed during the forward pass.

Conceptually:

```text
(64, 784)
    ↓
sum over axis=0
    ↓
(1, 784)
```

Each row of the broadcasted bias contributed to the output, so the gradients from all broadcasted copies must be summed together.

Mathematically:

```python
db = dY.sum(axis=0, keepdims=True)
```

Result:

```python
db.shape = (1, 784)
```

which matches the original parameter shape.

### General Rule

Whenever a tensor was expanded through broadcasting during the forward pass, its gradient must be reduced along the broadcasted dimensions during the backward pass.

Examples:

```python
(1, 784)  ->  (64, 784)
```

Backward:

```python
sum(axis=0)
```

---

```python
(64, 1)  ->  (64, 784)
```

Backward:

```python
sum(axis=1, keepdims=True)
```

---

```python
(1, 1)  ->  (64, 784)
```

Backward:

```python
sum(axis=(0,1), keepdims=True)
```

### Future Improvement

A proper broadcasting implementation should store enough shape information during the forward pass to identify which dimensions were broadcasted. During backpropagation, gradients can then be automatically reduced to match the original tensor shape before accumulation.

Implementing this feature would make the engine significantly closer to the behavior of modern deep learning frameworks.
