import torch
from torch.nn import functional as F
from torch import nn


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation, n_classes=0):
        super(MLP, self).__init__()
        self.net = []
        self.net.append(torch.nn.Linear(input_dim, hidden_dim))
        self.net.append(activation())
        for _ in range(num_layers - 1):
            self.net.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.net.append(activation())
        self.net = torch.nn.Sequential(*self.net)
        self.shortcut = torch.nn.Linear(input_dim, hidden_dim)

        if n_classes != 0:
            self.classifier = torch.nn.Linear(hidden_dim, n_classes)

    def forward(self, data):
        if isinstance(data, torch.Tensor):
            x = data
        else:
            x = data.x
        out = self.net(x) + self.shortcut(x)
        if hasattr(self, "classifier"):
            return out, self.classifier(out)
        return
    
class FC(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation, n_classes=0, sparse_method=None):
        super(FC, self).__init__()
        if sparse_method is not None:
            self.mask = nn.Parameter(torch.ones((input_dim, input_dim)), requires_grad=True)
        self.net = []
        self.net.append(torch.nn.Linear(input_dim*input_dim, hidden_dim))
        self.net.append(activation())
        for _ in range(num_layers - 1):
            self.net.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.net.append(activation())
        self.net = torch.nn.Sequential(*self.net)
        self.shortcut = torch.nn.Linear(input_dim*input_dim, hidden_dim)

        if n_classes != 0:
            self.classifier = torch.nn.Linear(hidden_dim, n_classes)

    def forward(self, data):
        if isinstance(data, torch.Tensor):
            x = data
        else:
            x = data.x
        if hasattr(self, "mask"):
            x = x * self.mask
        x = x.view(-1, 1).squeeze()
        out = self.net(x) + self.shortcut(x)
        if hasattr(self, "classifier") and hasattr(self, "mask"):
            return F.log_softmax(self.classifier(out), dim=-1).unsqueeze(0), self.mask
        if hasattr(self, "classifier"):
            return F.log_softmax(self.classifier(out), dim=-1).unsqueeze(0)
