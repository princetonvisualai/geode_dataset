"""ResNet

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

[2] https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50(nn.Module):
   def __init__(self, n_out = 38, reset_last = False, weights=ResNet50_Weights.DEFAULT):
      super().__init__()
      self.resnet = resnet50(weights=weights)
      self.resnet.fc = nn.Sequential (
         nn.Linear(2048, n_out),
      )
      self.require_all_grads()

   def require_all_grads(self):
      for param in self.parameters():
         param.requires_grad = True

   def forward(self, x):
      outputs = self.resnet(x)
      '''
      # uncomment when testing model
      func = nn.Softmax(dim=1)
      outputs = func(outputs)
      '''
      return outputs

                         
