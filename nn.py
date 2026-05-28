from engine import Value
import random
class Neuron:
    def __init__(self,nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
    def __call__(self, x):
        ai = sum((w*xi for w,xi in zip(self.w,x)),self.b)
        return ai
    def parameters(self):
        return self.w+[self.b]
    
class Layer:
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out
    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
class MLP:
    def __init__(self,nin,nouts):
        self.layers = []
        sizes = [nin] + nouts
        for i in range(len(nouts)):
            self.layers.append(Layer(sizes[i],sizes[i+1]))
    def __call__(self, x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
            if (i != len(self.layers)-1):
                x = [xi.ReLU() for xi in x]
            else:
                x = [xi.sigmoid() for xi in x]
        return x
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params