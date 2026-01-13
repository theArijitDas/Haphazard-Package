from typing import Literal

import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import AdamW

from .Utils.utils import Embedding, Aggregation, LSTMModel


AggregationType = Literal["max", "min", "max_magnitude", "min_magnitude", "mean", "sum"]
HeadType = Literal["lstm"]

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
        
        self.embed = Embedding(embedding_size = embedding_size,
                 init_features = n_features)

        self.aggregate = Aggregation(mode = aggregation_mode)

        if head == "lstm":
            self.head = LSTMModel(
                input_size = embedding_size, 
                output_size = num_classes if num_classes > 2 else 1, 
                hidden_size = kwargs.get("hidden_size", 32), 
                num_layers = kwargs.get("num_layers", 1)
                )

        self.criterion = CrossEntropyLoss() if num_classes > 2 else BCEWithLogitsLoss()

        parameters = list(self.embed.parameters()) + list(self.aggregate.parameters()) + list(self.head.parameters())
        self.optimizer = AdamW(parameters, lr=lr)
    
    def __call__(self, x, mask, y):

        x = torch.tensor(x, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.bool)
        y = torch.tensor(y)
        
        feature_embeddings = self.embed(x, mask)
        instance_embedding = self.aggregate(feature_embeddings)

        logit = self.head(instance_embedding)

        y = y.long() if self.num_classes > 2 else y.float()
        loss = self.criterion(logit, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
