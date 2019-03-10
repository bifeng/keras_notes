TensorFlow provides primitives for defining functions on tensors and automatically computing their derivatives.

### tensor

Formally, tensors are multilinear maps from vector spaces to the real numbers

a tensor can be represented as a multidimensional array of numbers



### auto-differentiation

TensorFlow nodes in computation graph have attached gradient operations.
Use backpropagation (using node-specific gradient ops) to compute required gradients for all variables in graph.



### computation graph

Big idea: express a numeric computation as a graph.
● Graph nodes are operations which have any number of inputs and outputs
● Graph edges are tensors which flow between nodes

