refer: CS224n, CS20si



## tensorflow

TensorFlow provides primitives for defining functions on tensors and automatically computing their derivatives.



### Why tensorflow ?

1. tensor/flow

   Formally, tensors are multilinear maps from vector spaces to the real numbers

   a tensor can be represented as a multidimensional array of numbers

   

   TensorFlow = tensor + flow = data + flow 

2. separates definition of computations from their execution

   Phase 1: assemble a graph

   ​	Graph

   ​	Nodes: operators, variables, and constants

   ​	Edges: tensors

   Phase 2: use a session to execute operations in the graph.

   ​	A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated.

   ​	Session will also allocate memory to store the current values of variables.

### Why graph ?

1. Save computation. Only run subgraphs that lead to the values you want to fetch.
2. Break computation into small, differential pieces to facilitate auto-differentiation
3. Facilitate distributed computation, spread the work across multiple CPUs, GPUs, TPUs, or other devices
4. Many common machine learning models are taught and visualized as directed graphs



### auto-differentiation

TensorFlow nodes in computation graph have attached gradient operations.
Use backpropagation (using node-specific gradient ops) to compute required gradients for all variables in graph.



### computation graph

Big idea: express a numeric computation as a graph.
● Graph nodes are operations which have any number of inputs and outputs
● Graph edges are tensors which flow between nodes



















