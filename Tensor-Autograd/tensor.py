import numpy as np    
class Tensor:
    def __init__(self,data,_children=(),_op=''):
        self.data = np.array(data,dtype=float)
        self._prev = set(_children)
        self._op = _op
        self.grad = np.zeros_like(self.data)
        self._backward = lambda:None
    # Operations
    def __add__(self, other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self,other), '+')
        def _backward():
            self.grad += out.grad 
            other.grad += out.grad 
        out._backward = _backward
        return out
    def __neg__(self):
        out = Tensor(-self.data,(self,),'neg')
        def _backward():
            self.grad += -out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (other,self), '*')
        def _backward():

            grad_self = other.data * out.grad
            grad_other = self.data * out.grad

            if self.grad.shape != grad_self.shape:
                grad_self = np.sum(grad_self)

            if other.grad.shape != grad_other.shape:
                grad_other = np.sum(grad_other)
            self.grad += grad_self
            other.grad += grad_other
        out._backward = _backward
        return out
    def __matmul__(self, other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(np.dot(self.data,other.data),(self,other),'@')
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
       
    def sum(self):
        out = Tensor(self.data.sum(),(self,),'sum')
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out
    def exp(self):
        out = Tensor(np.exp(self.data),(self,),'exp')
        def _backward():
            self.grad += (out.data) * out.grad
        out._backward = _backward
        return out
    def log(self):
        out = Tensor(np.log(self.data),(self,),'log')
        def _backward():
            self.grad += (1/self.data) * out.grad
        out._backward = _backward
        return out
    def mean(self):
        out = Tensor(np.mean(self.data),(self,),'mean')
        def _backward():
            self.grad += (np.ones_like(self.data) / self.data.size) * out.grad
        out._backward = _backward
        return out
    def __pow__(self,power):

        out = Tensor(self.data ** power,(self,),f'pow({power})')

        def _backward():

            self.grad += (power * (self.data ** (power-1)) * out.grad)

        out._backward = _backward

        return out
    def __truediv__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)

        return self * (other ** -1)
    # Activation functions
    def relu(self):
        out = Tensor(np.maximum(0,self.data),(self,),'relu')
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out
     
    # Backpropagation
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
    # utility functions
    def __repr__(self):
        return f"Tensor(Shape={self.data.shape},Data={self.data})"
        