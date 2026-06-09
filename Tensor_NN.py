from tensor import Tensor
import numpy as np
class Linear:
    def __init__(self,in_features,out_features):
        self.w = Tensor(np.random.randn(in_features,out_features)*0.01)
        self.b = Tensor(np.zeros((1,out_features)))
    def __call__(self,x):
        return x @ self.w + self.b
    def parameters(self):
        return [self.w,self.b]
class MLP:
    def __init__(self):
        self.l1 = Linear(784,128)
        self.l2 = Linear(128,10)
    def __call__(self,x):
        x = self.l1(x).relu()
        x = self.l2(x)
        return x
    def parameters(self):
        return (
            self.l1.parameters()
            +
            self.l2.parameters()
            
        )
    