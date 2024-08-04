import logging

import nni
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
import pickle

from src.utils.metrics import multiclass_roc_auc_score
from src.utils.save_model import SaveBestModel

# Create logger
logger = logging.getLogger("__name__")
level = logging.INFO
logger.setLevel(level)
ch = logging.StreamHandler()
ch.setLevel(level)
logger.addHandler(ch)

def train_eval(
    model, optimizer, scheduler, class_weights, args, train_loader, test_loader=None, sparse_method=None, ood_dataset=None
):
    """
    Train model
    """
    model.train()
    save_best_model = SaveBestModel()  # initialize SaveBestModel class
    criterion = nn.NLLLoss(weight=class_weights)

    train_preds, train_labels, train_aucs, train_accs, train_loss, test_accs, test_aucs, test_losses = [], [], [], [], [], [], [], []
    total_correct = 0
    total_samples = 0
    print("Starting training...")
    for i in range(args.epochs):
        running_loss = 0  # running loss for logging
        avg_train_losses = []  # average training loss per epoch

        if args.model_name == "sparg" and ood_dataset:
            running_loss_ood = 0
            print("Training with OOD dataset for the sparse model...")
            for param in model.gnn.parameters():
                param.requires_grad = False
            for data in ood_dataset:
                data = data.to(args.device)
                optimizer.zero_grad()
                decoded = model(data, ood_dataset=True)
                reconstruction_loss = 0
                kl_divergence = 0
                if args.sparse_method == "mae":
                    mse_loss = nn.MSELoss()
                    reconstruction_loss = mse_loss(decoded, data.x)  # Autoencoder loss
                elif args.sparse_method == "vae":
                    mse_loss = nn.MSELoss()
                    reconstruction_loss = mse_loss(decoded, data.x)
                    kl_divergence = -0.5 * torch.sum(1 + model.sparse_model.log_var - model.sparse_model.mu.pow(2) - model.sparse_model.log_var.exp())
                if args.model_name == "fc":
                    mask = model.mask
                else:
                    mask = model.sparse_model.mask
                #total_loss = args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.linalg.norm(mask) # Frobenius norm
                total_loss =  reconstruction_loss + 0.01 * torch.sum(torch.abs(mask))  # L1 regularization 
                #total_loss =  reconstruction_loss + 0.01 * torch.sum(mask ** 2)  # L2 regularization (squared)
                #total_loss =  args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * (torch.sum(torch.abs(mask)) + args.weights_elastic * torch.sum(mask ** 2)) # Elastic Net
                #total_loss =  reconstruction_loss + 0.01 * k_support_norm(mask, 5) # K-support norm
                #total_loss =  reconstruction_loss + 0.01 * torch.sum(x**2 * (1 - x**2)) # Forcing to 1 or 0 values
                total_loss.backward()
                optimizer.step()
                running_loss_ood += float(total_loss.item())
            avg_train_loss_ood = running_loss_ood / len(ood_dataset.dataset)
            for param in model.gnn.parameters():
                param.requires_grad = True
            print("EPOCH OOD: ", i, "LOSS: ", avg_train_loss_ood)
            print("Training with the ID dataset...")

        for data in train_loader:
            if sparse_method == "mae":
                data = data.to(args.device)
                optimizer.zero_grad()
                gcn_output, decoded = model(data)
                pred = gcn_output.max(dim=1)[1]  # Get predicted labels
                train_preds.append(pred.detach().cpu().tolist())
                train_labels.append(data.y.detach().cpu().tolist())
                total_correct += int((pred == data.y).sum())
                total_samples += data.y.size(0)  # Increment the total number of samples
                mse_loss = nn.MSELoss()
                reconstruction_loss = mse_loss(decoded, data.x)  # Autoencoder loss
                classification_loss = criterion(gcn_output, data.y)
                if args.model_name == "fc":
                    mask = model.mask
                else:
                    mask = model.sparse_model.mask
                #total_loss = classification_loss + args.loss_lambda*reconstruction_loss + args.weights_lambda * torch.linalg.norm(mask) # Frobenius norm
                total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(torch.abs(mask))  # L1 regularization 
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(mask ** 2)  # L2 regularization (squared)
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.linalg.norm(mask, 1) + 0.01 * torch.linalg.norm(mask, 2) # Elastic Net
                #total_loss = classification_loss + reconstruction_loss + 0.01 * k_support_norm(mask, 5) # K-support norm
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(x**2 * (1 - x**2)) # Forcing to 1 or 0 values
                total_loss.backward()
                optimizer.step()
                running_loss += float(total_loss.item())

            elif sparse_method == "vae":
                data = data.to(args.device)
                optimizer.zero_grad()
                gcn_output, decoded = model(data)
                pred = gcn_output.max(dim=1)[1]  # Get predicted labels
                train_preds.append(pred.detach().cpu().tolist())
                train_labels.append(data.y.detach().cpu().tolist())
                total_correct += int((pred == data.y).sum())
                total_samples += data.y.size(0)  # Increment the total number of samples
                mse_loss = nn.MSELoss()
                reconstruction_loss = mse_loss(decoded, data.x)  # Autoencoder loss
                kl_divergence = -0.5 * torch.sum(1 + model.sparse_model.log_var - model.sparse_model.mu.pow(2) - model.sparse_model.log_var.exp())
                classification_loss = criterion(gcn_output, data.y)
                if args.model_name == "fc":
                    mask = model.mask
                else:
                    mask = model.sparse_model.mask
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.linalg.norm(mask) # Frobenius norm
                total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(torch.abs(mask))   # L1 regularization 
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(mask ** 2)  # L2 regularization (squared)
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * (torch.sum(torch.abs(mask)) + args.weights_elastic * torch.sum(mask ** 2)) # Elastic Net
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * k_support_norm(mask, 5) # K-support norm
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(mask**2 * (1 - mask**2)) # Forcing to 1 or 0 values
                total_loss.backward()
                optimizer.step()
                running_loss += float(total_loss.item())

            elif args.sparse_method == "baseline_mask":
                data = data.to(args.device)
                optimizer.zero_grad()
                gcn_output, masked = model(data)
                pred = gcn_output.max(dim=1)[1]  # Get predicted labels
                train_preds.append(pred.detach().cpu().tolist())
                train_labels.append(data.y.detach().cpu().tolist())
                total_correct += int((pred == data.y).sum())
                total_samples += data.y.size(0)  # Increment the total number of samples
                classification_loss = criterion(gcn_output, data.y)
                if args.model_name == "fc":
                    mask = model.mask
                else:
                    mask = model.sparse_model.mask
                #total_loss = classification_loss + args.weights_lambda* torch.linalg.norm(mask) # Frobenius norm
                total_loss = classification_loss + args.weights_lambda * torch.sum(torch.abs(mask))  # L1 regularization
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(mask ** 2)  # L2 regularization (squared)
                #total_loss =  classification_loss + args.weights_lambda * (torch.sum(torch.abs(mask)) + args.weights_elastic * torch.sum(mask ** 2)) # Elastic Net
                #total_loss = classification_loss + reconstruction_loss + 0.01 * k_support_norm(mask, 5) # K-support norm
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(x**2 * (1 - x**2)) # Forcing to 1 or 0 values
                #with torch.autograd.detect_anomaly():
                    #total_loss.backward(retain_graph=True)
                total_loss.backward()
                optimizer.step()
                running_loss += float(total_loss.item())         
            else:
                data = data.to(args.device)
                optimizer.zero_grad()
                out = model(data)
                #out = out[0] # fully connected
                pred = out.max(dim=0)[1]  # Get predicted labels
                train_preds.append(pred.detach().cpu().tolist())
                train_labels.append(data.y.detach().cpu().tolist())
                total_correct += int((pred == data.y).sum())
                total_samples += data.y.size(0)  # Increment the total number of samples
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
        

        avg_train_loss = running_loss / len( 
            train_loader.dataset
        )  # Correctly calculate loss per epoch
        avg_train_losses.append(avg_train_loss)

        train_acc, train_auc, _, _, _ = test(model, train_loader, args)

        logging.info(
            f"(Train) | Epoch={i+1:03d}/{args.epochs}, loss={avg_train_loss:.4f}, "
            + f"train_acc={(train_acc * 100):.2f}, "
            + f"train_auc={(train_auc * 100):.2f}"
        )

        if (i + 1) % args.test_interval == 0:
            tests_perc_accs = {}
            tests_perc_aucs = {}
            tests_perc_losses = {}
            best_perc = 0
            for perc in [0, 0.7, 0.8, 0.9, 0.95, 0.99]:
                test_acc, test_auc, test_loss, _, _ = test(model, test_loader, args, nulling_out=perc)
                text = (
                    f"(Test) {perc} | Epoch {i+1}), test_acc={(test_acc * 100):.2f}, "
                    f"test_auc={(test_auc * 100):.2f}\n"
                )
                logging.info(text)
                if test_acc > best_perc:
                    best_perc = perc
                tests_perc_accs[perc] = test_acc
                tests_perc_aucs[perc] = test_auc
                tests_perc_losses[perc] = test_loss    
            
        if args.enable_nni:
            nni.report_intermediate_result(train_acc)

        if scheduler:
            scheduler.step(avg_train_loss)

        train_accs.append(train_acc)
        train_aucs.append(train_auc)
        train_loss.append(avg_train_loss)

        test_accs.append(tests_perc_accs)
        test_aucs.append(tests_perc_aucs)
        test_losses.append(tests_perc_losses)

        save_best_model(avg_train_loss, i, model, optimizer, criterion, args)

    train_accs, train_aucs = np.array(train_accs), np.array(train_aucs)
    return train_accs, train_aucs, train_loss, test_accs, test_aucs, test_losses, model


@torch.no_grad()
def test(model, loader, args, test_loader=None, nulling_out=None):
    """
    Test model
    """
    model.eval()

    preds = []
    # preds_prob = []
    labels = []
    test_aucs = []
    running_loss = 0
    mse_loss = nn.MSELoss()
    if args.sparse_method == "mae":
        print("Testing model using sparse method")
    l = 0
    for data in loader:
        data = data.to(args.device)

        if args.sparse_method == "mae":
            if args.model_name == "fc":
                    mask = model.mask
            else:
                mask = model.sparse_model.mask
            if nulling_out:
                data.x = data.x * mask
                threshold = torch.quantile(np.abs(data.x.view(-1)), nulling_out)
                data.x = data.x * (np.abs(data.x) >= threshold)
                gcn_output, decoded = model(data)
                pred = gcn_output.max(dim=1)[1]
                reconstruction_loss = mse_loss(decoded, data.x)  # Autoencoder loss
                classification_loss = nn.NLLLoss()(gcn_output, data.y)
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.linalg.norm(mask) # Frobenius norm
                total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(torch.abs(mask))  # L1 regularization
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(mask ** 2)  # L2 regularization (squared)
                #total_loss = classification_loss + args.loss_lambda*reconstruction_loss + args.weights_lambda * (torch.sum(torch.abs(mask)) + args.weights_elastic * torch.sum(mask ** 2)) # Elastic Net
                #total_loss = classification_loss + reconstruction_loss + 0.01 * k_support_norm(mask, 5) # K-support norm
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(x**2 * (1 - x**2)) # Forcing to 1 or 0 values
            else:
                gcn_output, decoded = model(data)
                pred = gcn_output.max(dim=1)[1]
                reconstruction_loss = mse_loss(decoded, data.x)  # Autoencoder loss
                classification_loss = nn.NLLLoss()(gcn_output, data.y)
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.linalg.norm(mask) # Frobenius norm
                total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(torch.abs(mask))  # L1 regularization
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(mask ** 2)  # L2 regularization (squared)
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.linalg.norm(mask, 1) + 0.01 * torch.linalg.norm(mask, 2) # Elastic Net
                #total_loss = classification_loss + reconstruction_loss + 0.01 * k_support_norm(mask, 5) # K-support norm
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(x**2 * (1 - x**2)) # Forcing to 1 or 0 values
        elif args.sparse_method == "baseline_mask":
            if args.model_name == "fc":
                    mask = model.mask
            else:
                mask = model.sparse_model.mask
            if nulling_out:
                data.x = data.x * mask
                threshold = torch.quantile(np.abs(data.x.view(-1)), nulling_out)
                data.x = data.x * (np.abs(data.x) >= threshold)
                gcn_output, masked = model(data)
                pred = gcn_output.max(dim=1)[1]
                classification_loss = nn.NLLLoss()(gcn_output, data.y)
                #total_loss = classification_loss + args.weights_lambda * torch.sum(torch.abs(mask))# Frobenius norm
                total_loss = classification_loss + args.weights_lambda * torch.sum(torch.abs(mask))  # L1 regularization
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(mask ** 2)  # L2 regularization (squared)
                #total_loss =  classification_loss + args.weights_lambda * (torch.sum(torch.abs(mask)) + args.weights_elastic * torch.sum(mask ** 2)) # Elastic Net
                #total_loss = classification_loss + 0.01 * k_support_norm(mask, 5) # K-support norm
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(x**2 * (1 - x**2)) # Forcing to 1 or 0 values

            else:
                gcn_output, masked = model(data)
                pred = gcn_output.max(dim=1)[1]
                classification_loss = nn.NLLLoss()(gcn_output, data.y)
                #total_loss = classification_loss + args.weights_lambda * torch.sum(torch.abs(mask))# Frobenius norm
                total_loss = classification_loss + args.weights_lambda * torch.sum(torch.abs(mask))  # L1 regularization
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(mask ** 2)  # L2 regularization (squared)
                #total_loss =  classification_loss + args.weights_lambda * (torch.sum(torch.abs(mask)) + args.weights_elastic * torch.sum(mask ** 2)) # Elastic Net
                #total_loss = classification_loss + 0.01 * k_support_norm(mask, 5) # K-support norm
                #total_loss = classification_loss + reconstruction_loss + 0.01 * torch.sum(x**2 * (1 - x**2)) # Forcing to 1 or 0 values
        elif args.sparse_method == "vae":
            if args.model_name == "fc":
                    mask = model.mask
            else:
                mask = model.sparse_model.mask
            if nulling_out:
                data.x = data.x * mask
                threshold = torch.quantile(np.abs(data.x.view(-1)), nulling_out)
                data.x = data.x * (np.abs(data.x) >= threshold)
                gcn_output, decoded = model(data)
                pred = gcn_output.max(dim=1)[1]
                reconstruction_loss = mse_loss(decoded, data.x)  # Autoencoder loss
                classification_loss = nn.NLLLoss()(gcn_output, data.y)
                kl_divergence = -0.5 * torch.sum(1 + model.sparse_model.log_var - model.sparse_model.mu.pow(2) - model.sparse_model.log_var.exp())
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.linalg.norm(mask) # Frobenius norm
                total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(torch.abs(mask))  # L1 regularization 
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(mask ** 2)  # L2 regularization (squared)
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * (torch.sum(torch.abs(mask)) + args.weights_elastic * torch.sum(mask ** 2)) # Elastic Net
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * k_support_norm(mask, 5) # K-support norm
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(mask**2 * (1 - mask**2)) # Forcing to 1 or 0 values
            else:
                gcn_output, decoded = model(data)
                pred = gcn_output.max(dim=1)[1]
                reconstruction_loss = mse_loss(decoded, data.x)  # Autoencoder loss
                classification_loss = nn.NLLLoss()(gcn_output, data.y)
                kl_divergence = -0.5 * torch.sum(1 + model.sparse_model.log_var - model.sparse_model.mu.pow(2) - model.sparse_model.log_var.exp())
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.linalg.norm(mask) # Frobenius norm
                total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(torch.abs(mask))  # L1 regularization 
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(mask ** 2)  # L2 regularization (squared)
                #total_loss = classification_loss + args.loss_lambda*( + 0.1*kl_divergence) + args.weights_lambda * (torch.sum(torch.abs(mask)) + args.weights_elastic * torch.sum(mask ** 2)) # Elastic Net
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * k_support_norm(mask, 5) # K-support norm
                #total_loss = classification_loss + args.loss_lambda*(reconstruction_loss + 0.1*kl_divergence) + args.weights_lambda * torch.sum(mask**2 * (1 - mask**2)) # Forcing to 1 or 0 values
        else:
            out = model(data)
            #out = out[0] # fully connected
            pred = out.max(dim=1)[1]
            total_loss = nn.NLLLoss()(out, data.y)

        preds.append(pred.detach().cpu().numpy().flatten())
        labels.append(data.y.detach().cpu().numpy().flatten())
        running_loss += total_loss.item()
    labels = np.array(labels).ravel()
    preds = np.array(preds).ravel()
    avg_test_loss = running_loss / len(loader.dataset)

    if args.num_classes > 2:
        try:
            # Compute the ROC AUC score.
            t_auc = multiclass_roc_auc_score(labels, preds)
        except ValueError as err:
            # Handle the exception.
            print(f"Warning: {err}")
            t_auc = 0.5
    else:
        t_auc = metrics.roc_auc_score(labels, preds, average="weighted")

    test_aucs.append(t_auc)

    if test_loader is not None:
        _, test_auc, preds, labels = test(model, test_loader, args)
        test_acc = metrics.balanced_accuracy_score(labels, preds)
        return test_auc, test_acc
    else:
        t_acc = metrics.balanced_accuracy_score(labels, preds)
        return t_acc, t_auc, avg_test_loss, preds, labels

@torch.no_grad()
def get_masks_best_binarization(test_loader, model, best_perc, args):
    """
    Compute similarity between masks
    """
    model.eval()
    if args.sparse_method:
        if args.model_name == "fc":
                mask = model.mask
        else:
            mask = model.sparse_model.mask
        threshold = torch.quantile(np.abs(mask.view(-1)), best_perc)
        mask = mask * (np.abs(mask) >= threshold)
        mask = (mask != 0).float()
        return mask
        
def compute_similarity_masks(matrices_dict, args):
    """
    Compute similarity between masks accross folds
    """
    results = {}
    for fold1 in matrices_dict.keys():
        for fold2 in matrices_dict.keys():
            if fold1 != fold2:
                similarity = torch.sum(matrices_dict[fold1] == matrices_dict[fold2]) / matrices_dict[fold1].numel()
                print(f"Similarity between {fold1} and {fold2} is {similarity}")
                results[(fold1, fold2)] = similarity

    with open(f"./save_results_masks_similarity.pkl", "wb") as f:
        pickle.dump(results, f)

    return results