from typing import Literal

import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import AdamW

from .Utils.utils import Embedding, Aggregation, TaskHead


AggregationType = Literal["max", "min", "max_magnitude", "min_magnitude", "mean", "sum"]
HeadType = Literal["lstm", "transformer", "mlp", "odl"]

class V2F:
    def __init__(self,
                 n_features: int,
                 num_classes: int,
                 embedding_size: int,
                 aggregation_mode: AggregationType="max",
                 lr: float=0.01,
                 head: HeadType="lstm",
                 **kwargs
                 ):
        
        self.num_classes = num_classes
        
        self.embed = Embedding(
            embedding_size = embedding_size, 
            init_features = n_features
            )

        self.aggregate = Aggregation(mode = aggregation_mode)

        self.head = TaskHead(head_name = head, 
                             input_size = embedding_size, 
                             output_size = num_classes if num_classes > 2 else 1, 
                             **kwargs)

        self.criterion = BCEWithLogitsLoss() if num_classes <= 2 else CrossEntropyLoss()
        self.criterion_per_layer = BCEWithLogitsLoss(reduction='none') if num_classes <= 2 else CrossEntropyLoss(reduction='none')

        parameters = list(self.embed.parameters()) + list(self.aggregate.parameters()) + list(self.head.parameters())
        self.optimizer = AdamW(parameters, lr=lr)
    
    def __call__(self, x, mask, y):

        x = torch.tensor(x, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.bool)
        y = torch.tensor(y)
        y = y.long() if self.num_classes > 2 else y.float()
        
        feature_embeddings = self.embed(x, mask)
        instance_embedding = self.aggregate(feature_embeddings)

        if self.head.name == 'odl':
            logit, hidden_outputs = self.head(instance_embedding)
            logit = logit.detach()
            losses_per_layer = self.criterion_per_layer(hidden_outputs, y.expand_as(hidden_outputs)).squeeze()  # (num_layers,)
            loss = torch.sum(losses_per_layer * self.head.model.alpha)
        
        else:
            logit = self.head(instance_embedding)
            logit = torch.squeeze(logit)

            loss = self.criterion(logit, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.head.name == 'odl':
            self.head.model.update_alpha(losses_per_layer.detach())

        with torch.no_grad():
            if self.num_classes == 2:
                logit = logit.detach().cpu().item()
                pred = int(logit >= 0.0)
            else:
                logit = logit.detach().cpu()
                pred = torch.argmax(logit)

                logit = logit.tolist()
                pred = int(pred)
        
        return pred, logit
