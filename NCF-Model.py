import torch
import torch.nn as nn

class NeuralContextFusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(512,256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256,128),

            nn.ReLU(),

            nn.Linear(128,64),

            nn.ReLU(),

            nn.Linear(64,12)

        )

    def forward(self,x):

        return self.network(x)

model = NeuralContextFusion()