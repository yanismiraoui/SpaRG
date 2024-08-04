import torch
from torch.nn import functional as F


class BrainGNN(torch.nn.Module):
    """Adapted from https://github.com/HennyJie/BrainGB"""

    def __init__(self, gnn, mlp, args, sparse_model=None, discriminator=lambda x, y: x @ y.t()):
        super(BrainGNN, self).__init__()
        self.gnn = gnn
        self.mlp = mlp
        self.sparse_model = sparse_model
        self.pooling = args.pooling
        self.discriminator = discriminator
        self.sparse_method = args.sparse_method

    def forward(self, data, edge_index=None, edge_attr=None, batch=None):
        if edge_index is None:
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            x = data.x
        if self.sparse_method == "mae":
            decoded, masked_flattened, masked = self.sparse_model(x)
            g = self.gnn(decoded, edge_index, edge_attr, batch)
            if self.pooling == "concat":
                _, g = self.mlp(g)
            log_logits = F.log_softmax(g, dim=-1)
            return log_logits, decoded
        elif self.sparse_method == "vae":
            decoded, mu, log_var, masked = self.sparse_model(x)
            g = self.gnn(masked, edge_index, edge_attr, batch)
            if self.pooling == "concat":
                _, g = self.mlp(g)
            log_logits = F.log_softmax(g, dim=-1)
            return log_logits, decoded
        elif self.sparse_method == "baseline_mask":
            masked = self.sparse_model(x)
            g = self.gnn(masked, edge_index, edge_attr, batch)
            if self.pooling == "concat":
                _, g = self.mlp(g)
            log_logits = F.log_softmax(g, dim=-1)
            return log_logits, masked
        else:
            g  = self.gnn(x, edge_index, edge_attr, batch)
            if self.pooling == "concat":
                _, g = self.mlp(g)
            log_logits = F.log_softmax(g, dim=-1)
            return log_logits

        return log_logits
