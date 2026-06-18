import math
class Value:
    def __init__(self, data, _children=(),_op=''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._backward = lambda:None
        self._op = _op
    # Operations
    def __add__(self,other,_op='+'):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,(self,other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    def __sub__(self, other,_op='-'):
        return (self + (other * -1))
    def __mul__(self,other,_op='*'):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data,(self,other))
        def _backward():
            self.grad += other.data * out.grad 
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1
    def __pow__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data**other.data,(self,other))
        def _backward():
            self.grad += other.data *(self.data**(other.data-1)) * out.grad     # f = x^y
            other.grad += (self.data**other.data)*math.log(self.data) * out.grad
        out._backward = _backward
        return out
    def log(self):
        
        out = Value(math.log(self.data),(self,)) ## self.data > 0
        def _backward():
            self.grad += (1/self.data) * out.grad
        out._backward = _backward
        return out
    def exp(self):
        out = Value(math.exp(self.data),(self,))
        def _backward():
            self.grad += (math.exp(self.data)) * out.grad
        out._backward = _backward
        return out
    # Autograd
    def backward(self):
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()
    
    # Utilities
    def __repr__(self):
        return f'Value(data = {self.data}\ngrad = {self.grad})'
    def __radd__(self, other):
        return self+other
    def __rmul__(self, other):
        return self*other
    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other - self
    # Activation functions
    def Tanh(self):
        x = self.data
        t = math.tanh(x)

        out = Value(t, (self,))

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out
    def ReLU(self):
        x = self.data
        R = self.data if self.data > 0 else 0
        out = Value(R,(self,))
        def _backward():
            self.grad += out.grad if(x>0) else 0
        out._backward = _backward
        return out
    def LreLU(self):
        x = self.data
        R = self.data if self.data > 0 else 0.01*x
        out = Value(R,(self,))
        def _backward():
            self.grad += self.grad + 1 * out.grad if(x>0) else 0.01 * out.grad
        out._backward = _backward
        return out
    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))

        out = Value(s, (self,))

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out
    def softmax(self):
        pass
    