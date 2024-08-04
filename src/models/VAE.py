import math

import torch
from torch import nn

import torch
from torch import nn
from torch.nn import functional as F

class MaskedVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MaskedVariationalAutoencoder, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        # Learnable Mask
        self.mask = nn.Parameter(torch.ones((input_dim, input_dim)), requires_grad=True)
        self.mu = None
        self.log_var = None
        self.input_dim = input_dim
        print(self.mask.shape)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_log_var(h1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        batch_size = x.size(0) // self.input_dim
        x_reshaped = x.view(batch_size, self.input_dim, self.input_dim)
        masked = x_reshaped * self.mask.unsqueeze(0)
        masked_flattened = masked.view(batch_size*self.input_dim, self.input_dim)
        self.mu, self.log_var = self.encode(masked_flattened)
        z = self.reparameterize(self.mu, self.log_var)
        return self.decode(z), self.mu, self.log_var, masked_flattened
