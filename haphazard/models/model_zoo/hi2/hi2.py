# https://github.com/Rohit102497/HaphazardInputsAsImages/blob/main/Code/main.py

from typing import Literal, cast

import numpy as np 
import torch 
import torch.optim as optim 
import torch.nn as nn 


PlotType = Literal["pie", "bar", "bar_nan"]
VisionModelType = Literal["res34", "vit_small"]

'''

For ease of coding, we determine the number of features from the data. However, note that this information is not needed.
This code can be very easily adapted to work with any number of features. The paper HI2 does not require the number of features to be known.

'''
class HI2:
    '''Modular implementation of HI2'''
    def __init__(
            self,
            n_class: int,
            n_features: int,
            plot_type: PlotType="bar",
            model_name: VisionModelType="res34",
            lr: float=2e-5,
            vert: bool=True,
            spacing: float=0.3,
            device: str|None = None,
            norm: str|None = None
        ):

        # Optional dependencies for vision features
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import torchvision             # type: ignore
            import timm                    # type: ignore
            global utils
            from .Utils import utils
        except ImportError as e:
            raise ImportError(
                "The following packages are required for vision features but one or more "
                "are not installed: Pillow, matplotlib, torchvision, timm\n\n"
                "To install them, run:\n\n"
                "    pip install haphazard[hi2]\n\n"
                "    or\n\n"
                "    pip install Pillow matplotlib torchvision timm\n\n"
                "Note: 'torchvision' and 'timm' require PyTorch. Visit\n"
                "https://pytorch.org/get-started/locally/ for the correct install command\n"
                "for your OS, Python version, and CUDA version."
            ) from e
        
        out_size = 1 if n_class == 2 else n_class

        self.n_class = n_class
        self.plot_type = plot_type
        self.vert = vert
        self.spacing = spacing
        self.norm = norm
        
        if model_name == 'res34':
            model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
            model.fc = nn.Linear(model.fc.in_features, out_size)

        elif model_name == 'vit_small':
            model = timm.create_model('vit_small_patch16_224', pretrained=True)

            head = cast(nn.Linear, model.head)
            model.head = nn.Linear(head.in_features, out_size)

        self.criterion = nn.BCEWithLogitsLoss() if n_class == 2 else nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr = lr)

        if device is None:
            self.device: str='cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.colors = utils.gen_colors(n_features, seed=42)

        model.to(device)
        self.model = model

        # create lists for storing various data for evaluation and analysis
        self.preds=[]
        self.pred_logits=[]

        self.feat=np.arange(n_features)
        min_arr=[np.nan]*n_features
        self.min_arr=np.array(min_arr)
        self.max_arr=np.copy(min_arr)
    
        self.model.train()
        self.run_sum=np.zeros(n_features)
        self.sum_sq=np.zeros(n_features)
        self.count=np.zeros(n_features)

    def __call__(self, x, y, mask):

            row = np.where(mask, x, np.nan)

            rev_mask =(1-mask) # reverse mask for plotting nan values in the form of crosses, if needed.
            mat_rev_mask=np.where(rev_mask, 0.5, np.nan) # reverse mask for plotting nan values in the form of crosses, if needed.
            rev = mat_rev_mask                         # fetch data to be plotted
            
            label = y
            label = torch.tensor(label)


            # ========== Convert haphazard input to image ==========
            if self.plot_type=='bar_nan':
                img=utils.bar_nan_plot(row, rev, self.colors, self.feat, self.vert, dpi=56)        #obtain bar plot tensor 
            elif self.plot_type == 'bar' and self.norm == "minmax":
                img=utils.bar_min_max_plot(row, self.colors, self.spacing)
            elif self.plot_type== 'bar':
                img=utils.bar_plot(row, self.colors)
            elif self.plot_type== 'pie':
                row_min = np.nanmin(row)
                row = row - row_min if row_min < 0 else row  # For pie plot, values must be positive
                img=utils.pie_plot(row, self.colors)
            else:
                raise AssertionError(f"Unhandled plot_type: {self.plot_type}") 

            # ========== Predict using image model ==========
            # ========== (Inference) ==========
            img, label = img.to(self.device), label.to(self.device)      #transfer to GPU
            img = torch.reshape(img,(-1,3,224,224))      # add extra dimension corresponding to batch, as required by model

            self.optimizer.zero_grad()
            outputs = self.model(img)                
            outputs = torch.squeeze(outputs)

            label = label.float() if self.n_class == 2 else label.long()
            loss = self.criterion(outputs, label)         # compute loss
            loss.backward()                               
                
            self.optimizer.step()
            
            with torch.no_grad():
                if self.n_class == 2:
                    logit = outputs.detach().cpu().item()
                    pred = int(logit >= 0.0)
                else:
                    logit = outputs.detach().cpu()
                    pred = torch.argmax(logit)

                    logit = logit.tolist()
                    pred = int(pred)
            
            return pred, logit