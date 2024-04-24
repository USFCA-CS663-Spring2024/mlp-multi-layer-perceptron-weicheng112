## Reference: https://towardsdatascience.com/step-by-step-guide-to-building-your-own-neural-network-from-scratch-df64b1c5ab6e

import numpy as np

class MyMLP:

    #     hidden_layer_tuple defines the number of inputs starting at the input layer and
    # ending at the output layer — eg. with the value (11, 5, 1) there are 11 units in the
    # input layer, 5 units in the hidden layer and — as a regression MLP — 1 unit in the
    # output layer

    def __init__(self, hidden_layer_tuple, activation_tuple):
        self.layers = hidden_layer_tuple
        self.activations = activation_tuple
        self.weights = []
        self.biases = []
        self.activation_funcs = []
        self.derivatives = []

        # ○ Learn weights — initially random.
        # ○ Learn biases — initially 0.0 … or random. The bias can be initialized to 0.
        for i in range(len(self.layers) - 1):
            # self.layers[i] == size of the layer
            curSize = self.layers[i]
            nexSize = self.layers[i+1]
            self.weights.append(np.random.randn(curSize, nexSize))
            self.biases.append(np.zeros((1, nexSize)))

        # Activation functions

        #     activation_layer_tuple defines the activation function used on each layer — eg.
        # with the value ('relu', 'sigmoid'), the activation function used in the first layer is
        # “ReLU” and the activation function on the second layer is sigmoid. Only values
        # 'relu', 'sigmoid' and 'none' are allowed here.


        ## Reference: 
        ## https://www.digitalocean.com/community/tutorials/sigmoid-activation-function-python
        def relu(x):
            # ReLU(x)=max(0,x)
            # return max(0.0, x)
        	return np.maximum(0, x)

        # gradient
        def relu_gradient(x):
            # either 1 or 0
            return np.where(x > 0, 1, 0)

        ## Reference: 
        ## https://www.digitalocean.com/community/tutorials/sigmoid-activation-function-python
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_gradient(x):
            return x * (1 - x)

        for function in self.activations:
            if function =='relu':
                self.activation_funcs.append(relu)
                self.derivatives.append(relu_gradient)
            elif function =='sigmoid':
                self.activation_funcs.append(sigmoid)
                self.derivatives.append(sigmoid_gradient)
            else:
                # Linear activation
                self.activation_funcs.append(lambda x: x)  
                # Derivative of linear activation is 1
                self.derivatives.append(lambda x: 1)  

    #   forward will take a matrix (X) and forward it through the MLP, producing a
    #   vector of hypotheses.
    def forward(self, X):
        # Store inputs as the initial activation
        self.activations_outputs = [X]
        activation = X

        # go through each layer
        for i in range(len(self.weights)):
            # Compute the input to the next layer
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            # Use activation function
            activation = self.activation_funcs[i](z)
            # Store the output for use in backpropagation
            self.activations_outputs.append(activation)
    
        # Return the final layer's activation
        return activation

    
    #   backprop will take a vector (y) and adjust the weights and biases in the MLP
    # according to the MSE loss.
    def backprop(self, y):
        # Calculate the initial error 
        error = self.activations_outputs[-1] - y
        
        weight_gradients = []
        bias_gradients = []
    
        # Loop through layers backwards
        for i in reversed(range(len(self.derivatives))):
            # Calculate delta at current layer
            delta = error * self.derivatives[i](self.activations_outputs[i + 1])

            # dw = (1/m) * np.dot(X, (A-Y).T)
            # db = (1/m) * np.sum((A-Y))
            # Calculate gradient for weights and biases
            weight_gradient = np.dot(self.activations_outputs[i].T, delta)
            bias_gradient = np.sum(delta, axis=0, keepdims=True)
    
            # store gradients in front of the array
            weight_gradients.insert(0, weight_gradient)
            bias_gradients.insert(0, bias_gradient)
    
            # Prepare error for next layer (if not the input layer)
            if i != 0:
                error = np.dot(delta, self.weights[i].T)
    
        return weight_gradients, bias_gradients
    def train(self, X, y, epochs, learning_rate):

        loss_history = [] 
        for epoch in range(epochs):
            # Forward pass to get output
            output = self.forward(X)
            
            # perform backpropagation to get gradients
            weight_gradients, bias_gradients = self.backprop(y)
            
            # Update weights and biases
            for i in range(len(self.weights)):
                self.weights[i] -= learning_rate * weight_gradients[i]
                self.biases[i] -= learning_rate * bias_gradients[i]
    
            ## MSE
            loss = np.mean((output - y) ** 2)  
            loss_history.append(loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        return loss_history


    # def train_with_val(self, X_train, y_train, X_val, y_val, epochs, learning_rate):
    #     train_loss_history = []  # to store training loss values
    #     val_loss_history = []  # to store validation loss values
    
    #     for epoch in range(epochs):
    #         # Training part
    #         train_output = self.forward(X_train)
    #         train_loss = np.mean((train_output - y_train) ** 2)
    #         train_loss_history.append(train_loss)  # append training loss to the history
    
    #         # Update weights and biases based on training data
    #         weight_gradients, bias_gradients = self.backprop(y_train)
    #         for i in range(len(self.weights)):
    #             self.weights[i] -= learning_rate * weight_gradients[i]
    #             self.biases[i] -= learning_rate * bias_gradients[i]
    
    #         # Validation part
    #         val_output = self.forward(X_val)
    #         val_loss = np.mean((val_output - y_val) ** 2)
    #         val_loss_history.append(val_loss)  # append validation loss to the history
    
    #         print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
    #     return train_loss_history, val_loss_history

# if __name__ == "__main__":
#     
#     mlp = MyMLP((11, 5, 1), ('relu', 'none'))  # Note: Last layer usually has no activation (linear)
#     X = np.random.rand(10, 11)  # 10 samples, 11 features each
#     y = np.random.rand(10, 1)  # 10 target values
#     mlp.train(X, y, epochs=100, learning_rate=0.01)

