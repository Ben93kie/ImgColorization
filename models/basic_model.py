import torch.nn as nn
import torchvision.models as models


class Net(nn.Module):
  def __init__(self, input_size=256, color_space='RGB',download=False):
    super(Net, self).__init__()
    if download:
      raise ValueError("Download not supported for this model.")
    self.color_space = color_space
    if self.color_space == 'RGB':
      final_channel_dim = 3
    elif self.color_space == 'LAB':
      final_channel_dim = 2
    # ResNet - First layer accepts grayscale images, 
    # and we take only the first few layers of ResNet for this task
    resnet = models.resnet18(num_classes=100)
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])
    RESNET_FEATURE_SIZE = 128
    ## Upsampling Network
    self.upsample = nn.Sequential(     
      nn.Conv2d(RESNET_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, final_channel_dim, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):
    midlevel_features = self.midlevel_resnet(input)
    output = self.upsample(midlevel_features)
    return output
