import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

class FeatureExtractor(nn.Module):
  def __init__(self):
    super(FeatureExtractor, self).__init__()
    self.model = models.resnet50(pretrained=True)
    self.model.fc = nn.Identity()

    self.transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

  def forward(self, x):
    return self.model(self.transform(x).unsqueeze(0)).detach().numpy()[0]
