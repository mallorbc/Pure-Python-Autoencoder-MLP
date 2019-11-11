from layer import *


class Multi_Layer_Perceptron:
    def __init__(self, input_size, output_size, hidden_layers):
        print("MLP")
        print("input size of ", input_size)
        print("output size of ", output_size)
        print(hidden_layers, " hidden_layers")
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        neurons_in_each_layer = []

        # stores a list of all the neurons at each stage
        neurons_in_each_layer.append(input_size)
        # adds all the hidden layers to the list
        for i in range(hidden_layers):
            layer_number = i + 1
            number_of_neurons = input(
                f"How many neurons for hidden layer #{layer_number}: ")
            number_of_neurons = int(number_of_neurons)
            neurons_in_each_layer.append(number_of_neurons)
        # appends the output layer to the list
        neurons_in_each_layer.append(self.output_size)
        self.neurons_in_each_layer = neurons_in_each_layer

        self.network_layers = []
        # makes the layers
        self.make_layers()

    def make_layers(self):
        for new_layer in self.neurons_in_each_layer:
            print(new_layer)
            self.network_layers.append(layer(new_layer))
        # print(self.network_layers[0].__dict__)
