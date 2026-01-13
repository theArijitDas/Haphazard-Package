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