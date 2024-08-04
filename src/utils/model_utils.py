import torch

from src.models.BrainGNN import BrainGNN
from src.models.SpaRG import SpaRG
from src.models.GATv2 import GATv2
from src.models.GCN import GCN
from src.models.MLP import MLP, FC
from src.models.AE import MaskedAutoencoder, Autoencoder, BaselineMask
from src.models.VAE import MaskedVariationalAutoencoder

def build_model(args, num_features, ood_dataset=None):
    """Build a classification model, e.g. GATv2, GCN, MLP

    Args:
        args (_type_): _description_
        num_features (_type_): _description_

    Raises:
        ValueError: if model not found

    Returns:
        nn.module: pyGmodel
    """
    print(f"Building model: {args.model_name}")
    if args.sparse_method in ["mae", "baseline_mask", "vae"] and args.model_name == "gcn":
        if args.model_name == "gcn" and args.sparse_method == "mae":
            model = BrainGNN(
                GCN(num_features, args),
                MLP(
                    args.num_classes,
                    args.hidden_dim,
                    args.n_MLP_layers,
                    torch.nn.ReLU,
                    n_classes=args.num_classes,
                ),
                args,
                sparse_model=MaskedAutoencoder(num_features, args.hidden_dim_sparse),
            )
        elif args.model_name == "gcn" and args.sparse_method == "baseline_mask":
            model = BrainGNN(
                GCN(num_features, args),
                MLP(
                    args.num_classes,
                    args.hidden_dim,
                    args.n_MLP_layers,
                    torch.nn.ReLU,
                    n_classes=args.num_classes,
                ),
                args,
                sparse_model=BaselineMask(num_features),
            )
        elif args.model_name == "gcn" and args.sparse_method == "vae":
            print("Using VAE")
            model = BrainGNN(
                GCN(num_features, args),
                MLP(
                    args.num_classes,
                    args.hidden_dim,
                    args.n_MLP_layers,
                    torch.nn.ReLU,
                    n_classes=args.num_classes,
                ),
                args,
                sparse_model=MaskedVariationalAutoencoder(num_features, args.hidden_dim_sparse, args.latent_dim_sparse),
            )
    elif args.model_name == "gatv2":
        model = BrainGNN(
            GATv2(num_features, args),
            MLP(
                args.num_classes,
                args.hidden_dim,
                args.n_MLP_layers,
                torch.nn.ReLU,
                n_classes=args.num_classes,
            ),
            args,
        )
    elif args.model_name == "gcn":
        model = BrainGNN(
            GCN(num_features, args),
            MLP(
                args.num_classes,
                args.hidden_dim,
                args.n_MLP_layers,
                torch.nn.ReLU,
                n_classes=args.num_classes,
            ),
            args,
        )
    elif args.model_name == "fc":
        print("Using FC model")
        model = FC(
            num_features,
            args.hidden_dim,
            args.n_MLP_layers,
            torch.nn.ReLU,
            n_classes=args.num_classes,
            sparse_method=args.sparse_method,
        )
    elif args.model_name == "sparg" and args.sparse_method == "vae":
        model = SpaRG(
            GCN(num_features, args),
            MLP(
                args.num_classes,
                args.hidden_dim,
                args.n_MLP_layers,
                torch.nn.ReLU,
                n_classes=args.num_classes,
            ),
            args,
            sparse_model=MaskedVariationalAutoencoder(num_features, args.hidden_dim_sparse, args.latent_dim_sparse),
        )
    elif args.model_name == "sparg" and args.sparse_method == "mae":
        model = SpaRG(
            GCN(num_features, args),
            MLP(
                args.num_classes,
                args.hidden_dim,
                args.n_MLP_layers,
                torch.nn.ReLU,
                n_classes=args.num_classes,
            ),
            args,
            sparse_model=MaskedAutoencoder(num_features, args.hidden_dim_sparse),
        )
    else:
        raise ValueError(f'ERROR: Model name "{args.model_name}" not found!')
    return model
