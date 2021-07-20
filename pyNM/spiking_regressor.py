__author__ = 'Ajay Arunachalam'
__version__ = '0.0.1'
__date__ = '17.07.2021'

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class SpikingNeuronLayer(nn.Module):
    def __init__(self, device, n_inputs, n_hidden=100, decay_multiplier=0.9, threshold=2.0, penalty_threshold=2.5):
        super(SpikingNeuronLayer, self).__init__()
        self.device = device
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.decay_multiplier = decay_multiplier
        self.threshold = threshold
        self.penalty_threshold = penalty_threshold

        self.fc = nn.Linear(n_inputs, n_hidden)

        self.init_parameters()
        self.reset_state()
        self.to(self.device)
        
    def init_parameters(self):
        for param in self.parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
        
    def reset_state(self):
        self.prev_inner = torch.zeros([self.n_hidden]).to(self.device)
        self.prev_outer = torch.zeros([self.n_hidden]).to(self.device)

    def forward(self, x):
        """
        Call the neuron at every time step.

        x: activated_neurons_below

        return: a tuple of (state, output) for each time step. Each item in the tuple
        are then themselves of shape (batch_size, n_hidden) and are PyTorch objects, such 
        that the whole returned would be of shape (2, batch_size, n_hidden) if casted.
        """
        if self.prev_inner.dim() == 1:
            # Adding batch_size dimension directly after doing a `self.reset_state()`:
            batch_size = x.shape[0]
            self.prev_inner = torch.stack(batch_size * [self.prev_inner])
            self.prev_outer = torch.stack(batch_size * [self.prev_outer])

        # 1. Weight matrix multiplies the input x
        #input_excitation = self.fc1(x)
        input_excitation = self.fc(x)
        #input_excitation = self.fc3(x)

        # 2. We add the result to a decayed version of the information we already had.
        inner_excitation = input_excitation + self.prev_inner * self.decay_multiplier

        # 3. We compute the activation of the neuron to find its output value, 
        #    but before the activation, there is also a negative bias that refrain thing from firing too much.
        outer_excitation = F.relu(inner_excitation - self.threshold)

        # 4. If the neuron fires, the activation of the neuron is subtracted to its inner state 
        #    (and with an extra penalty for increase refractory time), 
        #    because it discharges naturally so it shouldn't fire twice. 
        do_penalize_gate = (outer_excitation > 0).float()
        # TODO: remove following /2?
        inner_excitation = inner_excitation - (self.penalty_threshold/self.threshold * inner_excitation) * do_penalize_gate

        # 5. The outer excitation has a negative part after the positive part. 
        outer_excitation = outer_excitation #+ torch.abs(self.prev_outer) * self.decay_multiplier / 2.0

        # 6. Setting internal values before returning. 
        #    And the returning value is the one of the previous time step to delay 
        #    activation of 1 time step of "processing" time. For logits, we don't take activation.
        delayed_return_state = self.prev_inner
        delayed_return_output = self.prev_outer
        self.prev_inner = inner_excitation
        self.prev_outer = outer_excitation
        return delayed_return_state, delayed_return_output


class InputDataToSpikingPerceptronLayer(nn.Module):
    def __init__(self, device):
        super(InputDataToSpikingPerceptronLayer, self).__init__()
        self.device = device
        self.reset_state()
        self.to(self.device)

    def reset_state(self):
        #     self.prev_state = torch.zeros([self.n_hidden]).to(self.device)
        pass

    def forward(self, x, is_2D=True):
        x = x.view(x.size(0), -1)  # Flatten 2D image to 1D for FC
        random_activation_perceptron = torch.rand(x.shape).to(self.device)
        return random_activation_perceptron * x


class OutputDataToSpikingPerceptronLayer(nn.Module):
    def __init__(self, device, input_dim, hidden_dim_l1, hidden_dim_l2, hidden_dim_l3, output_dim=1):

        """
        average_output: might be needed if this is used within a regular neural net as a layer.
        Otherwise, sum may be numerically more stable for gradients with setting average_output=False.
        """
        super(OutputDataToSpikingPerceptronLayer, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reset_state()
        self.to(self.device)
        

        self.layer1 = nn.Linear(input_dim, hidden_dim_l1)
        self.layer2 = nn.Linear(hidden_dim_l1, hidden_dim_l2)
        self.layer3 = nn.Linear(hidden_dim_l2, hidden_dim_l3)
        self.layerout = nn.Linear(hidden_dim_l3, output_dim)
        self.relu = nn.ReLU()

    def reset_state(self):
        #     self.prev_state = torch.zeros([self.n_hidden]).to(self.device)
        pass

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layerout(x)
        return (x)

    def predict(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layerout(x)
        return (x)



    '''
        if average_output:
            self.reducer = lambda x, dim: x.sum(dim=dim)
        else:
            self.reducer = lambda x, dim: x.mean(dim=dim)
    

    def forward(self, x):
        if type(x) == list:
            x = torch.stack(x)
        return self.reducer(x, 0)
    
    
    def forward(self, x):
        x = (self.layer1(x))
        x = (self.layer2(x))
        #x = self.relu(self.layer3(x))
        x = self.output_conversion(x)
        return (x)
    '''


class SpikingNeuralNetwork(nn.Module):
    def __init__(self, device, input_dim, hidden_dim_l1, hidden_dim_l2, hidden_dim_l3,  n_time_steps, begin_eval, output_dim=1):
        super(SpikingNeuralNetwork, self).__init__()
        assert (0 <= begin_eval and begin_eval < n_time_steps)
        self.device = device
        self.input_dim = input_dim
        #self.n_hidden = n_hidden
        self.n_time_steps = n_time_steps
        self.begin_eval = begin_eval
        self.hidden_dim_l1 = hidden_dim_l1
        self.hidden_dim_l2 = hidden_dim_l2
        self.hidden_dim_l3 = hidden_dim_l3

        self.input_conversion = InputDataToSpikingPerceptronLayer(device)

        self.layer1 = SpikingNeuronLayer(
            device, n_inputs=input_dim, n_hidden=100,
            decay_multiplier=0.9, threshold=1.0, penalty_threshold=1.5
        )

        self.layer2 = SpikingNeuronLayer(
            device, n_inputs=100, n_hidden=output_dim,
            decay_multiplier=0.9, threshold=1.0, penalty_threshold=1.5
        )

        #self.relu = nn.ReLU()

        #self.layerout = nn.Linear(50, 1)

        #self.output_conversion = OutputDataToSpikingPerceptronLayer(average_output=False)  # Sum on outputs.

        #self.output_conversion = OutputDataToSpikingPerceptronLayer(device, 1, output_dim)  # Sum on outputs.

        #self.output_conversion = nn.Linear(1,1)  # Sum on outputs.

        self.output_conversion = OutputDataToSpikingPerceptronLayer(device, input_dim, hidden_dim_l1, hidden_dim_l2, hidden_dim_l3, output_dim)  # Sum on outputs.

        #self.relu = nn.ReLU()

        self.to(self.device)

    def forward_through_time(self, x):
        """
        This acts as a layer. Its input is non-time-related, and its output too.
        So the time iterations happens inside, and the returned layer is thus
        passed through global average pooling on the time axis before the return 
        such as to be able to mix this pipeline with regular backprop layers such
        as the input data and the output data.
        """
        self.input_conversion.reset_state()
        self.layer1.reset_state()
        self.layer2.reset_state()

        out = []

        all_layer1_states = []
        all_layer1_outputs = []
        all_layer2_states = []
        all_layer2_outputs = []
        for _ in range(self.n_time_steps):
            xi = self.input_conversion(x)

            # For layer 1, we take the regular output.
            layer1_state, layer1_output = self.layer1(xi)

            # We take inner state of layer 2 because it's pre-activation and thus acts as out logits.
            layer2_state, layer2_output = self.layer2(layer1_output)

            all_layer1_states.append(layer1_state)
            all_layer1_outputs.append(layer1_output)
            all_layer2_states.append(layer2_state)
            all_layer2_outputs.append(layer2_output)
            out.append(layer2_state)

        #out = self.output_conversion(out[self.begin_eval:])
        #out = self.output_conversion(x)
        out = self.output_conversion(x[self.begin_eval:])
        #out = self.output_conversion(out)
        #out = F.relu(out)
        #out = self.layerout(out[self.begin_eval:])
        #out = self.layerout(layer2_output)
       
        return out, [[all_layer1_states, all_layer1_outputs], [all_layer2_states, all_layer2_outputs]]

    def forward(self, x):
        out, _ = self.forward_through_time(x)
        #out, _ = self.output_conversion(x)
        return out
        
        #return F.log_softmax(out, dim=-1)
    '''

    def forward(self, inputs):
        x = self.relu(self.layer1(inputs))
        #x = self.relu(self.layer2(x))
        x = self.layerout(x)
        return (x)
    '''

    def visualize_all_neurons(self, x):
        assert x.shape[0] == 1 and len(x.shape) == 4, (
            "Pass only 1 example to SpikingNet.visualize(x) with outer dimension shape of 1.")
        _, layers_state = self.forward_through_time(x)

        for i, (all_layer_states, all_layer_outputs) in enumerate(layers_state):
            layer_state  =  torch.stack(all_layer_states).data.cpu().numpy().squeeze().transpose()
            layer_output = torch.stack(all_layer_outputs).data.cpu().numpy().squeeze().transpose()

            self.plot_layer(layer_state, title="Inner state values of neurons for layer {}".format(i))
            self.plot_layer(layer_output, title="Output spikes (activation) values of neurons for layer {}".format(i))

    def visualize_neuron(self, x, layer_idx, neuron_idx):
        assert x.shape[0] == 1 and len(x.shape) == 4, (
            "Pass only 1 example to SpikingNet.visualize(x) with outer dimension shape of 1.")
        _, layers_state = self.forward_through_time(x)

        all_layer_states, all_layer_outputs = layers_state[layer_idx]
        layer_state  =  torch.stack(all_layer_states).data.cpu().numpy().squeeze().transpose()
        layer_output = torch.stack(all_layer_outputs).data.cpu().numpy().squeeze().transpose()

        self.plot_neuron(layer_state[neuron_idx], title="Inner state values neuron {} of layer {}".format(neuron_idx, layer_idx))
        self.plot_neuron(layer_output[neuron_idx], title="Output spikes (activation) values of neuron {} of layer {}".format(neuron_idx, layer_idx))

    def plot_layer(self, layer_values, title):
        """
        plot the layer. 
        """
        width = max(16, layer_values.shape[0] / 8)
        height = max(4, layer_values.shape[1] / 8)
        plt.figure(figsize=(width, height))
        plt.imshow(
            layer_values,
            interpolation="nearest",
            cmap=plt.cm.rainbow
        )
        plt.title(title)
        plt.colorbar()
        plt.xlabel("Time")
        plt.ylabel("Neurons of layer")
        plt.show()

    def plot_neuron(self, neuron_through_time, title):
        width = max(16, len(neuron_through_time) / 8)
        height = 4
        plt.figure(figsize=(width, height))
        plt.title(title)
        plt.plot(neuron_through_time)
        plt.xlabel("Time")
        plt.ylabel("Neuron's activation")
        plt.show()