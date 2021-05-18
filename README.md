# ai-ml-ds-lab
Files to setup AI/Machinle Learning/DataScience lab

Setup
--

1. git clone
2. pip install -r requirements.txt
3. jupyter lab
4. go to: http://localhost:8888/lab
5. ???
6. PROFIT!!

nn.py
--
nn.py is simple implementation of generic(kind-of) neural network.
It was based mostly on this art: https://victorzhou.com/blog/intro-to-neural-networks/ 
and it's actually generalized version of provided network.

Implemented network has the following traits:
1. You can specify number of input neurons
   ```python
    from nn import *
    network = NeuralNetwork(inputs=2)
   ```

2. You can add layers
    ```python
   network.add_layer(neurons=2)
   network.add_layer(neurons=2)
   network.add_layer(neurons=1)
   ```
3. last layer is always output:
    ```python
    network.output == network.layers[-1]
    ```
4. You can train network by providing samples(ndarray), labels(ndarray) and optionaly learning rate and epochs.
    ```python
    network.train(data, labels)
    ```
5. You can make predictions by feeding data to network:
   ```python
    network.feedforward(sample)
    ```
6. Network uses SGD after each sample
7. Network uses backpropagation to tweek weights and biases
8. Network uses sigmoid as activation function
9. There are 2 utility methods: 
    * `draw_loss_plot` - which uses matplotlib to draw plot of how loss changed in time
    * `print_network` - which prints all neurons' weights layer by layer
    
10. When in doubt check bottom of `nn.py` file for some examples.
