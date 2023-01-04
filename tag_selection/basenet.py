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


class ResNet50(nn.Module):    
    def __init__(self, n_out = 1000, reset_last = False, pretrained=True):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(2048, n_out)
        #self.fc = nn.Linear(hidden_size, n_classes)
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(dropout)        

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        #outputs = self.fc(self.dropout(self.relu(features)))

        return features
    
    def get_activations(self, x):
        feat_conv = torch.nn.Sequential( *list(self.resnet.children())[:-1])
        return feat_conv(x)

class ResNet18(nn.Module):    
    def __init__(self, n_out = 1000, pretrained=True):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(512, n_out)
        #if load:
        #    A = torch.load(path)
        #    self.resnet.load_state_dict(A['model'])
        
        #self.features_conv = torch.nn.Sequential( *list(self.resnet.children())[:8])

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        outputs = self.resnet(x)
        return outputs 

    def get_activations(self, x):
        feat_conv = torch.nn.Sequential( *list(self.resnet.children())[:8])
        return feat_conv(x)
    
class ResNet50_base(nn.Module):   
    """ResNet50 but without the final fc layer"""
    
    def __init__(self, pretrained, hidden_size=2048, dropout=0.5):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(2048, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)        

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        features = self.dropout(self.relu(features))

        return features


