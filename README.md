# Neural-Networks-Multi-Layer-Perceptron-MLP

<p>
The scikit-learn library is well-known for providing robust and efficient tools for Machine Learning and Statistical Learning such as regression, classification, and clustering. It also contains an interface that allows us to work with neural networks, which is the Multi-layer Perceptron (MLP) class.

A Multilayer Perceptron (MLP) is a fully connected class of feedforward artificial neural network (ANN). It consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Except for the input layer, each layer contains nodes (neurons) that use nonlinear activation functions such as ReLu to learn complex and abstract features in the input.

Class MLPClassifier utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish a MLP model from a linear model, as it can distinguish data that is not linearly separable.

Note that Multilayer perceptrons are sometimes referred to as "vanilla" neural networks, especially when they have a single hidden layer.
</p>

<p>
With scikit-learn's **MLPClassifier**, we can utilize the GridSearch cross validation method to optimize the following parameters:

1. hidden_layer_sizes:

Type: Tuple, 
length = n_layers - 2 (excluding input and output layers)
Default: (100,)
Description: Specifies the number of neurons in each hidden layer. The ith element in the tuple represents the number of neurons in the ith hidden layer. 
For example, if hidden_layer_sizes=(50, 30), it means there are two hidden layers with 50 and 30 neurons, respectively.

2. alpha:

Type: Float
Default: 0.0001
Description: Represents the strength of the L2 regularization term. 
L2 regularization is a regularization technique that adds a penalty term to the loss function to prevent overfitting. 
The value of alpha determines the strength of this penalty, and it is divided by the sample size when added to the loss.


3. max_iter:

Type: Integer
Default: 200
Description: Specifies the maximum number of iterations for training. 
The solver iterates until convergence (determined by the tolerance parameter 'tol') or until it reaches this maximum number of iterations. 
For stochastic solvers like 'sgd' (stochastic gradient descent) or 'adam', this parameter determines the number of epochs, i.e., how many times each data point will be used.

4. learning_rate_init:

Type: Float
Default: 0.001
Description: Sets the initial learning rate for weight updates. 
The learning rate controls the step size in updating the weights during training. 
This parameter is only used when the solver is set to 'sgd' (stochastic gradient descent) or 'adam'. 
The learning rate can affect the convergence and stability of the training process.
</p>

![image](https://github.com/varshahindupur09/Neural-Networks-Multi-Layer-Perceptron-MLP/assets/114629181/0894d416-550a-448f-9f8a-2d008dc2a1de)


<p>
  And this is a summary for usage of Different models in Different areas:

![image](https://github.com/varshahindupur09/Neural-Networks-MLP-IBM-Coursera/assets/114629181/8b1b3770-04fe-418d-8a7e-354683bdb1a8)
</p>

<p>
  Gradient Descent:

  I have used different use cases to demonstrate my knowledge on Gradient Descent:

  ![image](https://github.com/varshahindupur09/Neural-Networks-MLP-IBM-Coursera/assets/114629181/e8c78b29-05fb-46a0-b0db-0db147d5ad69)

</p>
