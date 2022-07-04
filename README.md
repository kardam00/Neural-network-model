## Neural Network Model

Basic neural network model from scratch in Python using machine learning formulas and derivatives.

This model is designed for 3 layers (input layer, 1 hidden layer and output layer). Nodes in every layers can vary according to users need.

If user wishes to increase hidden layers, he/she have to change a little bit of source code.

Model uses given input and actual output to initialize weights and biases and computes hidden layer and output layer using those weights and biases.
Computes loss and uses it to back propagate computing derivatives to find new weights and biases which decreases the loss.
It iterates doing forward and backward propagation and finds the best weights and biases which are then used to predict output for the next given set of inputs.


Library needed is **numpy** for matrices calculations.

We have to call **Initialise()** function to intialise weights randomly between range -0.15 to +0.15.

for prediction, we have to call **forward_prop()** function, which takes inputs, X, size of input, hidden and output layers concatenated weights and biases.

It return hidden layer and output layer

for calculating the loss, we have to call **loss()** function which take two inputs arrays, predicted output and actual output, and return mean squared error.

for minimizing the loss, we have to call **minimise()** function which takes inputs, (X, actual output, size of input, hidden and output layers, weights, biases, learning rate and numver of iterations to be performed) and returns minimum loss and new weights and new biases.

Forward and back propagation are done inside the **minimise()** function. For computing change in weights/biases we uses derivatives.
