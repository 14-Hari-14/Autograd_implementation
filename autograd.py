class Value:
    """ stores a single scalar value and its gradient its just a replica of what pytorch does at a smaller scale it uses tensors to keep calculations simple i use scalars"""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        
        # Function to compute gradient during backpropagation by default its none
        self._backward = lambda: None
        # _children is a tuple of Value objects that are the parents of this node but since we are going in backwards direction I have named this datastructure as children
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    """Following are the operators I use to implement the autograd functionality each operator returns a new Value object that represents the result of the operation and defines a backward function to compute gradients"""
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        # We need to accumulate the gradients since if two different nodes have the same parents (_children) then we need to add the gradients from both nodes
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    # This is the implementation of an activation function. I have used ReLU (Rectified Linear Unit) as an example. since the operator is easy to implement
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1
    
    """Following are the reverse operators to allow for operations like 1 + x, x - 1, etc. these are used to make the code more intuitive and allow for operations with scalars"""
    def __radd__(self, other): 
        return self + other

    def __sub__(self, other): 
        return self + (-other)

    def __rsub__(self, other): 
        return other + (-self)

    def __rmul__(self, other): 
        return self * other

    def __truediv__(self, other): 
        return self * other**-1

    def __rtruediv__(self, other): 
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
