import torch
import torch.nn as nn

class PlaceholderModel(nn.Module):
    def __init__(self):
        super(PlaceholderModel, self).__init__()
    def forward(self, x):
        return torch.rand(x.shape)
    def predict(self,x):
        return(self.forward(x)) 
    
    
if __name__ == "__main__":
    model = PlaceholderModel()
    batch = torch.rand(1,3,512,512),torch.rand(1,1,512,512)
    model.predict(batch).shape