# https://github.com/theArijitDas/Haphazard-Inputs-Representation-Learning/blob/main/Code/Models/model_factory.py

# Libraries requied
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim

class AuxDrop_ODL_AuxLayer1stlayer(nn.Module):
    def __init__(self,
                 n_classes, n_aux_feat, device, 
                 max_num_hidden_layers = 6, 
                 qtd_neuron_per_hidden_layer = 50, 
                 n_neuron_aux_layer = None, 
                 batch_size=1, b=0.99, s=0.2, lr=0.01, 
                 dropout_p=0.5):
        super(AuxDrop_ODL_AuxLayer1stlayer, self).__init__()

        self.device = device

        self.max_num_hidden_layers = max_num_hidden_layers
        self.qtd_neuron_per_hidden_layer = qtd_neuron_per_hidden_layer
        self.n_classes = n_classes
        self.n_neuron_aux_layer = n_neuron_aux_layer if n_neuron_aux_layer is not None else 5*n_aux_feat
        self.batch_size = batch_size

        self.b = Parameter(torch.tensor(
            b), requires_grad=False).to(self.device)
        self.n = Parameter(torch.tensor(
            lr), requires_grad=False).to(self.device)
        self.s = Parameter(torch.tensor(
            s), requires_grad=False).to(self.device)
        self.p = Parameter(torch.tensor(
            dropout_p), requires_grad=False).to(self.device)
        self.n_aux_feat = n_aux_feat

        # Stores hidden and output layers
        self.hidden_layers = []
        self.output_layers = []

        # The number of hidden layers would be "max_num_hidden_layers". First layer is the aux layer
        self.hidden_layers.append(
            nn.Linear(n_aux_feat, n_neuron_aux_layer))
        self.hidden_layers.append(
                    nn.Linear(n_neuron_aux_layer, qtd_neuron_per_hidden_layer))
        for i in range(max_num_hidden_layers - 2):
            self.hidden_layers.append(
                nn.Linear(qtd_neuron_per_hidden_layer, qtd_neuron_per_hidden_layer))
            
        # The number of output layer would be total number of layer ("max_num_hidden_layers") - 1 (no output from the aux layer/first layer)
        for i in range(max_num_hidden_layers - 1):
                self.output_layers.append(
                    nn.Linear(qtd_neuron_per_hidden_layer, n_classes))

        self.hidden_layers = nn.ModuleList(self.hidden_layers).to(self.device)
        self.output_layers = nn.ModuleList(self.output_layers).to(self.device)
        
        # Alphs is related to the output layers. So, the nummber of alphas would be equal to the number of hidden layers (max_num_hidden_layers) - 1.
        # The alpha value sums to 1 and is equal at the beginning of the training. 
        self.alpha = Parameter(torch.Tensor(self.max_num_hidden_layers - 1).fill_(1 / (self.max_num_hidden_layers - 1)),
                               requires_grad=False).to(self.device)


        self.loss_array = []
        self.alpha_array = []
        self.layerwise_loss_array = []
        self.prediction = []

    # Initialize the gradients of all the parameters with 0.
    def zero_grad(self):
        for i in range(self.max_num_hidden_layers - 1):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.max_num_hidden_layers):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    # This function predicts the output from each layer,calculates the loss and do gradient descent and alpha update.
    def update_weights(self, aux_feat, aux_mask, Y, show_loss):
        Y = torch.from_numpy(Y).to(self.device)

        # Predictions from each layer using the implemented forward function.
        predictions_per_layer = self.forward(aux_feat, aux_mask)

        real_output = torch.sum(torch.mul(
                self.alpha.view(self.max_num_hidden_layers - 1, 1).repeat(1, self.batch_size).view(
                    self.max_num_hidden_layers - 1, self.batch_size, 1), predictions_per_layer), 0)
        self.prediction.append(real_output.squeeze().tolist())

        criterion = nn.CrossEntropyLoss().to(self.device)
        loss = criterion(real_output.view(self.batch_size, self.n_classes), Y.view(self.batch_size).long())
        self.loss_array.append(loss.detach().cpu().numpy())
        if show_loss:
            if (len(self.loss_array) % 1000) == 0:
                print("WARNING: Set 'show_loss' to 'False' when not debugging. "
                      "It will deteriorate the fitting performance.")
                loss = np.mean(self.loss_array[-1000:])
                print("Alpha:" + str(self.alpha.data.cpu().numpy()))
                print("Training Loss: " + str(loss))
        
        losses_per_layer = []

        for out in predictions_per_layer:
            criterion = nn.CrossEntropyLoss().to(self.device)
            loss = criterion(out.view(self.batch_size, self.n_classes), Y.view(
                self.batch_size).long())
            losses_per_layer.append(loss)
        
        w = [None] * (len(losses_per_layer) + 1) 
        b = [None] * (len(losses_per_layer) + 1)

        with torch.no_grad():

            for i in range(len(losses_per_layer)):
                losses_per_layer[i].backward(retain_graph = True)                
                
                self.output_layers[i].weight.data -= self.n* \
                                                     self.alpha[i]*self.output_layers[i].weight.grad.data
                self.output_layers[i].bias.data -= self.n * \
                                                   self.alpha[i] * self.output_layers[i].bias.grad.data

                for j in range(i+2):
                    if w[j] is None:
                        w[j] = self.alpha[i] * self.hidden_layers[j].weight.grad.data
                        b[j] = self.alpha[i] * self.hidden_layers[j].bias.grad.data
                    else:
                        w[j] += self.alpha[i] * self.hidden_layers[j].weight.grad.data
                        b[j] += self.alpha[i] * self.hidden_layers[j].bias.grad.data

                self.zero_grad()

            for i in range(self.max_num_hidden_layers):
              self.hidden_layers[i].weight.data -= self.n * w[i]
              self.hidden_layers[i].bias.data -= self.n * b[i]

            for i in range(len(losses_per_layer)):
                self.alpha[i] *= torch.pow(self.b, losses_per_layer[i])
                self.alpha[i] = torch.max(
                    self.alpha[i], self.s / (len(losses_per_layer)))
            
        self.alpha.data = self.alpha.data / torch.sum(self.alpha.data)
        
        # To save the loss
        detached_loss = []
        for i in range(len(losses_per_layer)):
            detached_loss.append(losses_per_layer[i].detach().cpu().numpy())
        self.layerwise_loss_array.append(np.asarray(detached_loss))
        self.alpha_array.append(self.alpha.detach().cpu().numpy())

    # Forward pass. Get the output from each layer.     
    def forward(self, aux_feat, aux_mask):
        hidden_connections = []
        linear_x = []
        relu_x = []

        aux_feat = torch.from_numpy(aux_feat).float().to(self.device)
        aux_mask = torch.from_numpy(aux_mask).float().to(self.device)

        # Forward pass to the Aux layer.
        # Input to the aux layer will be the output from its previous layer and the incoming auxiliary inputs.
        linear_x.append(self.hidden_layers[0](aux_feat))
        relu_x.append(F.relu(linear_x[0]))
        # Based on the incoming aux data, the aux inputs which do not come gets included in the dropout probability.
        # Based on that we calculate the probability to drop the left over neurons in auxiliary layer.
        aux_p = (self.p * self.n_neuron_aux_layer - 
                 (aux_mask.size()[1] - torch.sum(aux_mask)))/(self.n_neuron_aux_layer 
                    - aux_mask.size()[1])
        binomial = torch.distributions.binomial.Binomial(probs=1-aux_p)
        non_aux_mask = binomial.sample([1, self.n_neuron_aux_layer - aux_mask.size()[1]])
        mask = torch.cat((aux_mask, non_aux_mask), dim = 1)
        hidden_connections.append(relu_x[0]*mask*(1.0/(1-self.p)))


        for i in range(1, self.max_num_hidden_layers):
            linear_x.append(self.hidden_layers[i](hidden_connections[i - 1]))
            relu_x.append(F.relu(linear_x[i]))
            hidden_connections.append(relu_x[i])
        
        output_class = []

        for i in range(self.max_num_hidden_layers-1):
            output_class.append(
                F.softmax(self.output_layers[i](hidden_connections[i+1]), dim=1))

        pred_per_layer = torch.stack(output_class)

        return pred_per_layer

    def validate_input_X(self, data):
        if len(data.shape) != 2:
            raise Exception(
                "Wrong dimension for this X data. It should have only two dimensions.")
    
    def validate_input_Y(self, data):
        if len(data.shape) != 1:
            raise Exception(
                "Wrong dimension for this Y data. It should have only one dimensions.")
    
    def partial_fit(self, aux_data, aux_mask, Y_data, show_loss=False):
        self.validate_input_X(aux_data)
        self.validate_input_Y(Y_data)
        self.update_weights(aux_data, aux_mask, Y_data, show_loss)