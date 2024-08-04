import argparse
import logging
import os
import os.path as osp
import pickle
import sys
from datetime import datetime
from typing import List

import nni
import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb

from src.dataset import BrainDataset
from src.utils.data_utils import create_features, get_x, get_y
from src.utils.explain import explain
from src.utils.get_transform import get_transform
from src.utils.model_utils import build_model
from src.utils.modified_args import ModifiedArgs
from src.utils.sample_selection import select_samples_per_class
from src.utils.save_model import save_model
from src.utils.train_and_evaluate import test, train_eval, get_masks_best_binarization, compute_similarity_masks

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class SpaRG_main:
    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="PRIVATE")
        parser.add_argument(
            "--model_name", type=str, default="sparg", choices=["fc", "gcn", "gatv2", "sparg"]
        )
        parser.add_argument(
            "--sparse_method", type=str, default=None, choices=["baseline_mask", "mae", "vae"]
        )
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument(
            "--node_features",
            type=str,
            default="adj",
            choices=[
                "identity",
                "degree",
                "degree_bin",
                "LDP",
                "node2vec",
                "adj",
                "diff_matrix",
                "eigenvector",
                "eigen_norm",
            ],
        )
        parser.add_argument(
            "--centrality_measure",
            type=str,
            default="node",
            choices=[
                "abs",
                "geo",
                "tan",
                "node",
                "eigen",
                "close",
                "concat_orig",
                "concat_scale",
            ],
            help="Chooses the topological measure to be used",
        )
        parser.add_argument("--epochs", type=int, default=15)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--weight_decay", type=float, default=2e-2)
        parser.add_argument(
            "--gcn_mp_type",
            type=str,
            default="node_concat",
            choices=[
                "weighted_sum",
                "bin_concat",
                "edge_weight_concat",
                "edge_node_concat",
                "node_concat",
            ],
        )
        parser.add_argument(
            "--gat_mp_type",
            type=str,
            default="attention_weighted",
            choices=[
                "attention_weighted",
                "attention_edge_weighted",
                "sum_attention_edge",
                "edge_node_concat",
                "node_concat",
            ],
        )
        parser.add_argument(
            "--pooling", type=str, choices=["sum", "concat", "mean"], default="concat"
        )
        parser.add_argument("--n_GNN_layers", type=int, default=4)
        parser.add_argument("--n_MLP_layers", type=int, default=4)
        parser.add_argument("--num_heads", type=int, default=4)
        parser.add_argument("--hidden_dim", type=int, default=16)
        parser.add_argument("--hidden_dim_sparse", type=int, default=16)
        parser.add_argument("--latent_dim_sparse", type=int, default=8)
        parser.add_argument("--loss_lambda", type=int, default=1)
        parser.add_argument("--weights_lambda", type=int, default=0.1)
        parser.add_argument("--weights_elastic", type=float, default=0.2)
        parser.add_argument("--edge_emb_dim", type=int, default=1)
        parser.add_argument("--bucket_sz", type=float, default=0.05)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--repeat", type=int, default=1)
        parser.add_argument("--k_fold_splits", type=int, default=4)
        parser.add_argument("--k_list", type=list, default=[4])
        parser.add_argument("--n_select_splits", type=int, default=4)
        parser.add_argument("--test_interval", type=int, default=1)
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--diff", type=float, default=0.2)
        parser.add_argument("--mixup", type=int, default=1, choices=[0, 1])
        parser.add_argument("--sample_selection", action="store_true")
        parser.add_argument("--enable_nni", action="store_true")
        parser.add_argument("--explain", action="store_true")
        parser.add_argument("--wandb", action="store_true", help="Track experiment")
        parser.add_argument("--log_result", action="store_true")
        parser.add_argument("--data_folder", type=str, default="datasets/")
        args = parser.parse_args()

        self_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir = osp.join(self_dir, args.data_folder)
        print(root_dir)
        dataset = BrainDataset(
            root=root_dir,
            name=args.dataset,
            pre_transform=get_transform(args.node_features),
            num_classes=args.num_classes,
        )
        
        args.num_nodes = dataset.num_nodes
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        init_wandb(
            name=f"{args.model_name}-{args.dataset}",
            heads=args.num_heads,
            epochs=args.epochs,
            hidden_channels=args.hidden_dim,
            node_features=args.node_features,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_classes=args.num_classes,
            device=args.device,
        )

        if args.enable_nni:
            args = ModifiedArgs(args, nni.get_next_parameter())

        # init model
        model_name = str(args.model_name).lower()
        args.model_name = model_name
        sparse_method = str(args.sparse_method).lower()
        args.sparse_method = sparse_method

        y = get_y(dataset)
        connectomes = get_x(dataset).T

        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(y), y=y
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(args.device)

        train_accs, train_aucs, train_losses, val_accs, val_aucs, val_losses, preds_all, labels_all, test_accs, test_aucs, search = (
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
        )

        if args.sample_selection:
            # Check if node centrality features and subject labels exist
            if os.path.exists(
                f"{args.data_folder}data_dict_{args.node_features}_{args.num_classes}.pkl"
            ):
                with open(
                    f"{args.data_folder}data_dict_{args.node_features}_{args.num_classes}.pkl",
                    "rb",
                ) as d_d:
                    data_dict = pickle.load(d_d)
                with open(
                    f"{args.data_folder}score_dict_{args.node_features}_{args.num_classes}.pkl",
                    "rb",
                ) as s_d:
                    score_dict = pickle.load(s_d)
            else:  # Create node centrality features and subject labels
                data_dict, score_dict = create_features(
                    connectomes.numpy(), y, args, args.centrality_measure
                )
                with open(
                    f"{args.data_folder}data_dict_{args.node_features}_{args.num_classes}.pkl",
                    "wb",
                ) as d_d:
                    pickle.dump(data_dict, d_d)
                with open(
                    f"{args.data_folder}score_dict_{args.node_features}_{args.num_classes}.pkl",
                    "wb",
                ) as s_d:
                    pickle.dump(score_dict, s_d)

        
        with open(f"{args.data_folder}idx_scanner.pkl", "rb") as f:
            scanner_indices = pickle.load(f)

        test_scanner = dataset[[idx for idx in range(len(dataset)) if idx in scanner_indices]]
        test_scanner_loader = DataLoader(
                    test_scanner, batch_size=args.test_batch_size, shuffle=False, drop_last=True
                )
        print("Length of test_scanner: ", len(scanner_indices))
        
        shuffled_indices = torch.randperm(len(dataset))
        test_size = int(0.2 * len(dataset))
        test_indices = shuffled_indices[:test_size]
        train_indices = shuffled_indices[test_size:]
        print("Length of val_indices: ", len(test_indices))
        print("Length of train_indices: ", len(train_indices))
        test_dataset = dataset[test_indices]
        test_loader = DataLoader(
                    test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=True
                )
        train_val_dataset = dataset[train_indices]
        y = get_y(train_val_dataset)
        print(len(y))

        # Hyperparameter tuning search space
        MLP_layers_values = [4]
        GNN_layers_values = [1]
        num_heads_values = [4]
        hidden_dim_values = [64]
        dropout_values = [0.9]
        loss_lambda_values = [100]
        weights_lambda_values = [0.01]
        weights_elastic_values = [0.001]

        save_result_tuning = {}

        for MLP_layers in MLP_layers_values:
            args.n_MLP_layers = MLP_layers
            for GNN_layers in GNN_layers_values:
                args.n_GNN_layers = GNN_layers
                for num_heads in num_heads_values:
                    args.num_heads = num_heads
                    for hidden_dim in hidden_dim_values:
                        args.hidden_dim = hidden_dim
                        for dropout in dropout_values:
                            args.dropout = dropout
                        for loss_lambda in loss_lambda_values:
                            args.loss_lambda = loss_lambda
                            for weights_lambda in weights_lambda_values:
                                args.weights_lambda = weights_lambda
                                for weights_elastic in weights_elastic_values:
                                    args.weights_elastic = weights_elastic
                                    print(f"MLP_layers: {MLP_layers}, GNN_layers: {GNN_layers}, num_heads: {num_heads}, hidden_dim: {hidden_dim}, dropout: {dropout}, loss_lambda: {loss_lambda}, weights_lambda: {weights_lambda}, weights_elastic: {weights_elastic}")
                                    save_result_tuning[(MLP_layers, GNN_layers, num_heads, hidden_dim, dropout, loss_lambda, weights_lambda, weights_elastic)] = []
                                    fold = -1
                                    masks_dict = {}
                                    for train_idx, val_idx in KFold(
                                        args.k_fold_splits,
                                        shuffle=True,
                                        random_state=args.seed,
                                    ).split(train_val_dataset):
                                        fold += 1
                                        print(f"Cross Validation Fold {fold+1}/{args.k_fold_splits}")

                                        if args.sample_selection:
                                            # Select top-k subjects with highest predictive power for labels
                                            sample_atlas = select_samples_per_class(
                                                train_idx,
                                                args.n_select_splits,
                                                args.k_list,
                                                data_dict,
                                                score_dict,
                                                y,
                                                shuffle=True,
                                                rs=args.seed,
                                            )

                                        for k in args.k_list:
                                            if args.sample_selection:
                                                selected_train_idxs = np.array(
                                                    [
                                                        sample_idx
                                                        for class_samples in sample_atlas.values()
                                                        for sample_indices in class_samples.values()
                                                        for sample_idx in sample_indices
                                                    ]
                                                )
                                            else:
                                                selected_train_idxs = np.array(train_idx)
                                            print(f"Length selected_train_idxs: {len(selected_train_idxs)}")
                                            # Apply RandomOverSampler to balance classes
                                            train_res_idxs, _ = RandomOverSampler().fit_resample(
                                                selected_train_idxs.reshape(-1, 1),
                                                [y[i] for i in selected_train_idxs],
                                            )
                                            print(f"Length train_res_idxs: {len(train_res_idxs)}")

                                            train_set = [train_val_dataset[i] for i in train_res_idxs.ravel()]
                                            print("Length train_set: ", len(train_set))
                                            val_set = [train_val_dataset[i] for i in val_idx]
                                            print("Length val_set: ", len(val_set))
                                            train_loader = DataLoader(
                                                train_set, batch_size=args.train_batch_size, shuffle=True,
                                            )
                                            val_loader = DataLoader(
                                                val_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True
                                            )
                                            model = build_model(args, train_val_dataset.num_features, None)
                                            model = model.to(args.device)
                                            optimizer = torch.optim.AdamW(
                                                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
                                            )
                                            scheduler = ReduceLROnPlateau(
                                                optimizer, mode="min", factor=0.5, patience=5, verbose=True
                                            )

                                            train_acc, train_auc, train_loss, val_acc, val_auc, val_loss, train_model = train_eval(
                                                model,
                                                optimizer,
                                                scheduler,
                                                class_weights,
                                                args,
                                                train_loader,
                                                val_loader,
                                                sparse_method=args.sparse_method,
                                            )

                                            save_model(
                                                args.epochs, train_model, optimizer, args
                                            )  # save trained model

                                            # test the best epoch saved model
                                            best_model_cp = torch.load(
                                                f"model_checkpoints/best_model_{args.model_name}_{args.num_classes}.pth"
                                            )
                                            model.load_state_dict(best_model_cp["model_state_dict"])

                                            train_accs[fold] = train_acc
                                            train_aucs[fold] = train_auc
                                            train_losses[fold] = train_loss
                                            val_accs[fold] = val_acc
                                            val_aucs[fold] = val_auc
                                            val_losses[fold] = val_loss

                                            if args.explain:
                                                explain(model, val_loader, args)

                                    # get the max accs and aucs for each fold
                                    max_val_accs = {fold: {perc: max(iteration[perc] for iteration in iterations) for perc in iterations[0]} for fold, iterations in val_accs.items()}
                                    max_val_aucs = {fold: {perc: max(iteration[perc] for iteration in iterations) for perc in iterations[0]} for fold, iterations in val_aucs.items()}
                                    print(max_val_accs)
                                    print(max_val_aucs)
                                    result_str = "(K Fold Final Result)| "

                                    # Get all unique percentages
                                    percentages = set(perc for fold_results in max_val_accs.values() for perc in fold_results)

                                    # Iterate over each percentage
                                    for perc in percentages:
                                        accs_at_perc = [fold_results[perc] for fold_results in max_val_accs.values()]
                                        aucs_at_perc = [fold_results[perc] for fold_results in max_val_aucs.values()]

                                        avg_acc = np.mean(accs_at_perc) * 100
                                        std_acc = np.std(accs_at_perc) * 100
                                        avg_auc = np.mean(aucs_at_perc) * 100
                                        std_auc = np.std(aucs_at_perc) * 100

                                        result_str += f"perc {perc}: avg_acc={avg_acc:.2f} +- {std_acc:.2f}, avg_auc={avg_auc:.2f} +- {std_auc:.2f}, "

                                    # Add maximum average acc and maximum auc across all percentages
                                    max_avg_acc = np.max([np.mean([fold_results[perc] for fold_results in max_val_accs.values()]) for perc in percentages]) * 100
                                    max_avg_auc = np.max([np.mean([fold_results[perc] for fold_results in max_val_aucs.values()]) for perc in percentages]) * 100
                                    result_str += f"max_avg_acc={max_avg_acc:.2f}, max_avg_auc={max_avg_auc:.2f}\n"

                                    for perc in percentages:
                                        print("====================================")
                                        print("Testing for Scanner group perc: ", perc)
                                        test_acc, test_auc, _, _, _ = test(model, test_scanner_loader, args, nulling_out=perc)
                                        print(f"Test acc: {test_acc}, Test auc: {test_auc}")
                                        save_result_tuning[(MLP_layers, GNN_layers, num_heads, hidden_dim, dropout, loss_lambda, weights_lambda, weights_elastic)].append(f"For Scanner group perc {perc}: (Test acc: {test_acc}, Test auc: {test_auc})")
                                        search[(MLP_layers, GNN_layers, num_heads, hidden_dim, dropout, loss_lambda, weights_lambda, weights_elastic)] = (max_avg_acc, max_avg_auc, train_accs, train_aucs, train_losses, val_accs, val_aucs, val_losses, test_acc, test_auc)

                                    search[(MLP_layers, GNN_layers, num_heads, hidden_dim, dropout, loss_lambda, weights_lambda, weights_elastic)] = (max_avg_acc, max_avg_auc, train_accs, train_aucs, train_losses, val_accs, val_aucs, val_losses, test_acc, test_auc)

                                    # save search 
                                    with open(
                                        f"./save_results.pkl",
                                        "wb",
                                    ) as f:
                                        pickle.dump(search, f)


                                    logging.info(result_str)
                                    save_result_tuning[(MLP_layers, GNN_layers, num_heads, hidden_dim, dropout, loss_lambda, weights_lambda, weights_elastic)].append(result_str)

                                    # save the results
                                    with open(
                                        f"./save_results_details.pkl",
                                        "wb",
                                    ) as f:
                                        pickle.dump(save_result_tuning, f)


if __name__ == "__main__":
    SpaRG_main().main()
