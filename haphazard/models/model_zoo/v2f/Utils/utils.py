from typing import cast

import torch
import torch.nn as nn

class Embedding(nn.Module):
    '''Create a list of learnable embeddings'''
    def __init__(self, embedding_size: int, init_features: int):
        super(Embedding, self).__init__()

        # Initialize (static) embeddings (no gradient updation)
        self.register_buffer('embedding_list', torch.empty(init_features, embedding_size))
        self.embedding_list = cast(torch.Tensor, self.embedding_list)
        nn.init.kaiming_normal_(self.embedding_list, mode='fan_in', nonlinearity='relu')

        # Initialize learnable embeddings (with gradient updation)
        self.embedding_list_grad = nn.Parameter(self.embedding_list.clone())

        # Indices to be updated in the next step
        self.last_indices = None

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        indices = torch.where(x_mask)[0]
        
        if self.last_indices is not None:
            with torch.no_grad():
                # Copy the updated values of features present in the last step from learnable to static embeddings
                self.embedding_list[self.last_indices] = self.embedding_list_grad[self.last_indices]

                # Copy the updated static embedding values to learnable embeddings
                self.embedding_list_grad.copy_(self.embedding_list)
        
        # Store indices to be updated in the next step
        self.last_indices = indices

        return torch.einsum('be, b -> be', self.embedding_list_grad[indices], x[indices])
        # tensor                        : shape
        #-----------------------------------------------------------------------
        # self.embedding_list[indices]  : (no. available features x embed size)
        # x[indices]                    : (no. available features)
        # output                        : (no. available features x embed size)


class Aggregation(nn.Module):
    '''Aggregate the feature-embeddings of the available features into instance-embedding'''
    def __init__(self, mode: str):
        super(Aggregation, self).__init__()

        if mode == 'max':
            # Select the max value for each embedding index from available feature embeddings
            self.aggregate = lambda x: torch.max(x, dim=0).values.view(1, -1)

        elif mode == 'min':
            # Select the min value for each embedding index from available feature embeddings
            self.aggregate = lambda x: torch.min(x, dim=0).values.view(1, -1)
        
        elif mode == 'max_magnitude':
            # Select the embedding with largest magnitude
            self.aggregate = lambda x: x[torch.argmax((x**2).sum(dim=1))].view(1, -1)

        elif mode == 'min_magnitude':
            # Select the embedding with smallet magnitude
            self.aggregate = lambda x: x[torch.argmin((x**2).sum(dim=1))].view(1, -1)

        elif mode == 'mean':
            # Take arithmatic mean of all the embeddings
            self.aggregate = lambda x: torch.mean(x, dim=0).view(1, -1)

        elif mode == 'sum':
            # Take arithmatic mean of all the embeddings
            self.aggregate = lambda x: torch.sum(x, dim=0).view(1, -1)

        else:
            raise ValueError(f'`{mode}` is an invalid mode for Aggregation class')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aggregate(x)


class LSTMModel(nn.Module):
    '''LSTM classification/regression head implementation'''
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)  # Get last hidden state
        x = self.fc(h_n[-1])  # Use last LSTM layer's hidden state
        return x  # Raw logits, activation should be applied externally
    

class TransformerModel(nn.Module):
    '''Transformer classification/regression head implementation'''
    def __init__(self, input_size: int, output_size: int, num_layers: int, num_heads: int, dim_feedforward: int):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward,
            batch_first=True  
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)  # Add batch dimension
        x = self.transformer(x)
        x = self.output_layer(x[:, 0])  # Take the first token's output
        return x  # Raw logits, activation should be applied externally


class MLP(nn.Module):
    '''Initialize a MLP by passing hidden-layer-size list'''
    def __init__(self, input_size: int, output_size: int, hidden_layers: list[int], non_linearity: str):
        super(MLP, self).__init__()

        hidden_layers = hidden_layers + [output_size]

        # Choose non-linearity
        non_linear_map = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        self.non_linearity = non_linear_map.get(non_linearity.lower(), nn.ReLU)
        
        prev = input_size
        layers = []
        for layer_size in hidden_layers:
            layers.append(nn.Linear(prev, layer_size))
            layers.append(self.non_linearity())
            prev = layer_size
        layers.pop()

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ODL(nn.Module):
    '''Flexible ODL classifier/regressor supporting classification (binary, multi-class), and regression tasks.''' 
    def __init__(self, input_size: int, output_size: int, hidden_layers: list[int],
                 non_linearity: str, b: float, s: float):
        super(ODL, self).__init__()
        
        self.register_buffer('b', torch.tensor(b, dtype=torch.float))
        self.register_buffer('s', torch.tensor(s, dtype=torch.float))
        
        # Choose non-linearity
        non_linear_map = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        self.non_linearity = non_linear_map.get(non_linearity.lower(), nn.ReLU)()
        
        # Define hidden layers
        layer_sizes = [input_size] + hidden_layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)
        ])
        
        # Define output layers corresponding to each hidden layer
        self.output_layers = nn.ModuleList([
            nn.Linear(layer_size, output_size) for layer_size in hidden_layers
        ])
        
        # Initialize alpha weights
        num_layers = max(len(hidden_layers), 1)
        self.register_buffer('alpha', torch.full((num_layers,), 1 / num_layers, dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_outputs = []
        
        for hidden_layer, output_layer in zip(self.hidden_layers, self.output_layers):
            x = self.non_linearity(hidden_layer(x))
            hidden_outputs.append(output_layer(x).view(-1))
        
        hidden_outputs = torch.stack(hidden_outputs)  # Shape: (num_layers, output_size)
        final_pred = torch.sum(hidden_outputs * self.alpha.view(-1, 1), dim=0)  # Weighted sum
        
        return final_pred, hidden_outputs # Raw logits, activation should be applied externally
    
    def update_alpha(self, losses_per_layer: torch.Tensor):
        with torch.no_grad():
            self.alpha.mul_(torch.pow(self.b, losses_per_layer.squeeze()))
            self.alpha.copy_(torch.maximum(self.alpha, self.s / len(self.hidden_layers)))
            self.alpha.div_(self.alpha.sum())


class TaskHead(nn.Module):
    '''Getter class for various task heads available for V2F'''
    def __init__(self, head_name: str, input_size: int, output_size: int, **kwargs):
        super(TaskHead, self).__init__()
        self.name = head_name
        if head_name == 'lstm':
            self.model = LSTMModel(
                input_size=input_size, 
                output_size=output_size, 
                hidden_size=kwargs.get("hidden_size", 32), 
                num_layers=kwargs.get("num_layers", 1)
            )
        elif head_name == 'transformer':
            self.model = TransformerModel(
                input_size=input_size,
                output_size=output_size,
                num_layers=kwargs.get('num_layers', 1),
                num_heads=kwargs.get('num_heads', 4),
                dim_feedforward=kwargs.get('dim_feedforward', 32)
            )
        elif head_name == 'mlp':
            self.model = MLP(
                input_size=input_size,
                output_size=output_size,
                hidden_layers=kwargs.get('hidden_layers', [64, 64]),
                non_linearity=kwargs.get('non_linearity', 'ReLU')
            )
        elif head_name == 'odl':
            self.model = ODL(
                input_size=input_size,
                output_size=output_size,
                hidden_layers=kwargs.get('hidden_layers', [64, 64]),
                non_linearity=kwargs.get('non_linearity', 'ReLU'),
                b=kwargs.get('b', 0.99),
                s=kwargs.get('s', 0.2)
            )
        else:
            raise ValueError(f'Invalid value of `head_name` encountered: {head_name} is not supported.')
    
    def forward(self, x):
        return self.model(x)
