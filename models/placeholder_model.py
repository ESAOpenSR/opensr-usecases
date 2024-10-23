import torch
import torch.nn as nn

class PlaceholderModel(nn.Module):
    def __init__(self):
        super(PlaceholderModel, self).__init__()
    def forward(self, x):
        b,c,h,w, = x.shape
        return torch.rand((b,1,h,w,))
    def predict(self,x):
        return(self.forward(x)) 
    
    
if __name__ == "__main__":
    model = PlaceholderModel()
    batch = torch.rand(41,4,512,512)
    model.predict(batch).shape