from torchvision import transforms as T
from torchvision import models
import torch

# JIT-able model
class InferenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_resnet = models.resnet18().eval()
        self.model_resnet.fc = torch.nn.Linear(self.model_resnet.fc.in_features, 2).eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model_resnet(x)
        return outputs
            
if __name__ == '__main__':
    m = InferenceModel()
    m.model_resnet.load_state_dict(torch.load('state_dict.pt'))
    torch.jit.save(torch.jit.script(m), 'resnet18.pt')

