import random
from autograd import Value

"""This is the simple implementation of a neural network using the autograd module I have implemented."""
class Module:
    """Base class for all modules in the neural network. It provides a common interface for all modules and allows for easy parameter management."""
    def zero_grad(self): 
        """Zero out the gradients of all nodes in the neural net"""
        # parameters is the weight and bias of each node/neuron
        for p in self.parameters(): 
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module): # Inherits from Module
    """A single neuron in the neural network. It has weights, a bias, and an activation function (ReLU by default)."""
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] # given a random weight for each input which is then changed during training
        self.b = Value(0) # bias is initialized to 0
        self.nonlin = nonlin # if nonlin is True, then the activation function is ReLU, otherwise it is linear (identity function)

    # calculates the activation of the neuron given an input x
    def __call__(self, x):
        """Compute the activation of the neuron given an input x."""
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act # checks if nonlin true -> ReLU else False -> linear

    def parameters(self):
        return self.w + [self.b] # Value objects so a list is returned here 

    def __repr__(self):
        """Representation function"""
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    """Rows of neurons"""
    # Kwargs handles the nonlin parameter for each neuron
    def __init__(self, nin, nout, **kwargs):        
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons] # calculates the output of each neuron in the layer given an input x and stores in list 
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()] # returns a list of all parameters (weights and biases) of all neurons in the layer

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module): # Multi-Layer Perceptron

    def __init__(self, nin, nouts): 
        sz = [nin] + nouts 
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))] # For the last layer nonlin is False (linear) and for all other layers nonlin is True (ReLU) so that we can use the output of last layer for regression or classfication tasks 

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) # provides input x to the first layer and then the output of each layer is passed as input to the next layer
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()] # returns parameters for all layers in the MLP

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
