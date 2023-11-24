import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
import torchvision.transforms as transforms
import pickle
import math
import torchvision.models as models
from torch.autograd.function import Function
from lib.normalize import Normalize
    
class Res18Featured(nn.Module):
    def __init__(self, pretrained = True, num_classes = 6, drop_rate = 0):
        super(Res18Featured, self).__init__()
        self.drop_rate = drop_rate
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)       
        self.features = nn.Sequential(*list(resnet.children())[:-2]) # after avgpool 512x1
        self.feavg = nn.Sequential(*list(resnet.children())[-2:-1])
        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512
   
        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7
        self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1),nn.Sigmoid())
        self.l2norm = Normalize(2)
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3 ,1, 1),   # [, 256, 7, 7]
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 3, 2, 1, 1),   # [, 128, 14, 14]
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),    # [, 64, 28, 28]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),      # [, 32, 56, 56]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),      # [, 32, 112, 112]
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),  # [, 16, 224, 224]
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 3, 3, 1, 1),         # [, 3, 224, 224]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x1 = self.feavg(x)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.l2norm(x1)
        out = self.fc(x1)
        rec = self.Decoder(x)
        #print(rec.size())
        return rec,x1,out