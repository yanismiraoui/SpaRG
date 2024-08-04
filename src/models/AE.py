import math

import torch
from torch import nn

class BaselineMask(nn.Module):
    def __init__(self, input_dim):
        super(BaselineMask, self).__init__()
        # Learnable Mask
        self.mask = nn.Parameter(torch.ones((input_dim, input_dim)), requires_grad=True)
        self.input_dim = input_dim
        print(self.mask.shape)

    def forward(self, x):
        batch_size = x.size(0) // self.input_dim
        x_reshaped = x.view(batch_size, self.input_dim, self.input_dim)
        masked = x_reshaped * self.mask.unsqueeze(0)

        return masked.view(batch_size*self.input_dim, -1)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MaskedAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(True)
        )
        # Learnable Mask
        self.mask = nn.Parameter(torch.ones((input_dim, input_dim)), requires_grad=True)
        self.input_dim = input_dim
        print(self.mask.shape)

    def forward(self, x):
        batch_size = x.size(0) // self.input_dim
        x_reshaped = x.view(batch_size, self.input_dim, self.input_dim)
        masked = x_reshaped * self.mask.unsqueeze(0)
        encoded = self.encoder(masked.view(batch_size*self.input_dim, -1))
        decoded = self.decoder(encoded).view(batch_size*self.input_dim, self.input_dim)
        masked_flattened = masked.view(batch_size*self.input_dim, self.input_dim)

        return decoded, masked_flattened, masked.view(batch_size*self.input_dim, -1)