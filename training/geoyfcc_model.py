"""ResNet

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

[2] https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ResNet50(nn.Module):
   def __init__(self, n_out = 29, reset_last = False, pretrained=False):
      super().__init__()
      self.resnet = torchvision.models.resnet50()
      self.resnet.fc = nn.Sequential (
         nn.Linear(2048, n_out),
      )
      self.require_all_grads()

   def require_all_grads(self):
      for param in self.parameters():
         param.requires_grad = True

   def forward(self, x):
      outputs = self.resnet(x)
      return outputs

                         