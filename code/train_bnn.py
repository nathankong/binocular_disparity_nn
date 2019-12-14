import sys
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)

from collections import defaultdict

from model import BinocularNetwork
from utils import \
    acquire_data_loaders, \
    compute_num_correct, \
    print_update, \
    record_weights

def train_step(
    model,
    train_loader,
    optimizer,
    loss_func,
    device
):
    losses = list()
    accuracies = list()
    total_samples = 0.

    model.train()
    for i, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        total_samples += data.shape[0]

        # Reset gradients
        optimizer.zero_grad()

        # Forward propagation
        predictions = model(data)
        loss = loss_func(predictions, labels)

        # Backward propagation
        loss.backward()
        optimizer.step()

        accuracy = compute_num_correct(
            predictions,
            labels
        )

        losses.append(loss.item())
        accuracies.append(accuracy)

    return losses, accuracies, total_samples

def val_or_test_step(model, loader, loss_func, device):
    losses = list()
    accuracies = list()
    total_samples = 0.

    model.eval()
    with torch.no_grad():
        for i, (data, labels) in enumerate(loader):
            data, labels = data.to(device), labels.to(device)
            total_samples += data.shape[0]

            predictions = model(data)
            loss = loss_func(predictions, labels)

            accuracy = compute_num_correct(
                predictions,
                labels
            )

            losses.append(loss.item())
            accuracies.append(accuracy)

    return losses, accuracies, total_samples

def train(train_params, exp_id, device, verbose=True):
    # Data loaders
    train_loader, test_loader, val_loader = acquire_data_loaders(
        train_params["image_dir"],
        train_params["batch_size"],
        device
    )

    # Load model
    m = BinocularNetwork(
        n_filters=train_params["num_kernels"],
        k_size=train_params["kernel_size"],
        input_size=train_params["img_size"],
        init_gabors=train_params["init_gabors"]
    ).to(device)

    # Loss function
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    #loss_func = nn.BCELoss(reduction="sum")
    #loss_func = nn.NLLLoss(reduction="sum")

    # Optimizer
    optimizer = optim.SGD(m.parameters(), lr=train_params["learning_rate"])
    #optimizer = optim.Adam(m.parameters(), lr=train_params["learning_rate"])

    # Do training
    all_losses = defaultdict(list)
    all_accuracies = defaultdict(list)
    simple_unit_filters = list()
    complex_unit_weights = list()
    for epoch_idx in range(train_params["num_epochs"]):
        record_weights(m, simple_unit_filters, complex_unit_weights)
        # Train step
        train_losses, correct, total_samples = train_step(
            m,
            train_loader,
            optimizer,
            loss_func,
            device
        )
        avg_loss = np.sum(train_losses) / total_samples
        acc = np.sum(correct) / total_samples
        all_losses["train"].append(avg_loss)
        all_accuracies["train"].append(acc)

        if verbose:
            print_update("Train", avg_loss, acc, epoch_idx, train_params["num_epochs"])

        # Validation step every epoch
        val_losses, correct, total_samples = val_or_test_step(
            m,
            val_loader,
            loss_func,
            device
        )
        avg_loss = np.sum(val_losses) / total_samples
        acc = np.sum(correct) / total_samples
        all_losses["validation"].append(avg_loss)
        all_accuracies["validation"].append(acc)

        if verbose:
            print_update("Validation", avg_loss, acc, epoch_idx, train_params["num_epochs"])

        # Test step every 10 epochs
        if (epoch_idx+1) % 10 == 0:
            test_losses, correct, total_samples = val_or_test_step(
                m,
                test_loader,
                loss_func,
                device
            )
            avg_loss = np.sum(test_losses) / total_samples
            acc = np.sum(correct) / total_samples
            all_losses["test"].append(avg_loss)
            all_accuracies["test"].append(acc)
    
            if verbose:
                print_update("Test", avg_loss, acc, epoch_idx, train_params["num_epochs"])

    results = dict()
    results["losses"] = all_losses
    results["accuracies"] = all_accuracies
    results["simple_unit_weights"] = simple_unit_filters
    results["complex_unit_weights"] = complex_unit_weights
    results["train_params"] = train_params
    results_fname = "results/results_{}.pkl".format(exp_id)
    pickle.dump(results, open(results_fname, "wb"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    default_dataset_dir = "/mnt/fs5/nclkong/datasets/bnn_dataset/"
    parser.add_argument('--image-dir', type=str, default=default_dataset_dir)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--num-kernels', type=int, default=28)
    parser.add_argument('--kernel-size', type=int, default=19)
    parser.add_argument('--image-size', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--num-epochs', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--init-gabors', type=bool, default=False)
    parser.add_argument('--exp-id', type=str, default="test")
    args = parser.parse_args()

    # Use GPU or CPU
    if torch.cuda.is_available():
        cuda_idx = int(args.cuda)
        device = torch.device("cuda:{}".format(cuda_idx))
    else:
        device = torch.device("cpu")
    print "Device:", device

    # Experiment ID
    exp_id = args.exp_id
    print "Experiment ID:", exp_id

    # Training parameters
    train_params = dict()
    train_params["image_dir"] = args.image_dir.lower()
    train_params["num_kernels"] = int(args.num_kernels)
    train_params["kernel_size"] = int(args.kernel_size)
    train_params["img_size"] = int(args.image_size)
    train_params["batch_size"] = int(args.batch_size)
    train_params["num_epochs"] = int(args.num_epochs)
    train_params["learning_rate"] = float(args.lr)
    train_params["init_gabors"] = bool(args.init_gabors)
    print "Training parameters:", train_params

    # Do the training
    train(train_params, exp_id, device)


