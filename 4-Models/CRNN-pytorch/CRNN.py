import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CRNN(nn.Module):
    def __init__(self, hidR = 256, layerR = 1, hidC = 256):
        super(CRNN, self).__init__()
        # len of input (time domain)
        self.P = 2500
        # width of input
        self.m = 12
        # hidden size of RNN
        self.hidR = hidR
        self.layerR = layerR
        # hidden size of CNN
        self.hidC = hidC
        # kernel size of CNN
        self.Ck = 5;

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m)),
            nn.BatchNorm2d(self.hidC),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
    
        self.GRU1 = nn.GRU(self.hidC, self.hidR, num_layers=self.layerR, bidirectional=False);
        
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hidR * self.layerR * 1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        batch_size = x.size(0);
        
        #CNN
        c = x.view(-1, 1, self.P, self.m);
        c = self.conv1(c);
        c = torch.squeeze(c, 3);
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous();
        _, r = self.GRU1(r);
        
        r = r.view(batch_size, -1)

        res = self.fc1(r);
        return res.view(-1)