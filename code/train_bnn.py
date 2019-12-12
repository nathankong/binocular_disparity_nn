import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(0)

from model import BinocularNetwork
from utils import acquire_data_loaders, compute_accuracy

def train_step(
    model,
    train_loader,
    optimizer,
    loss_func,
    device
):
    losses = list()
    accuracies = list()

    model.train()
    for i, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward propagation
        predictions = model(data)
        loss = loss_func(predictions, labels)

        # Backward propagation
        loss.backward()
        optimizer.step()

        accuracy = compute_accuracy(
            predictions,
            labels,
            data.shape[0]
        )

        losses.append(loss.item())
        accuracies.append(accuracy)

    return losses, accuracies 

def test_step(model, test_loader, loss_func, device):
    losses = list()
    accuracies = list()

    model.eval()
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            predictions = model(data)

            loss = loss_func(predictions, labels)

            accuracy = compute_accuracy(
                predictions,
                labels,
                data.shape[0]
            )

            losses.append(loss.item())
            accuracies.append(accuracy)

    return losses, accuracies

def val_step(val_loader):
    # TODO
    model.eval()
    pass

def train(train_params, device, verbose=True):
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
        input_size=train_params["img_size"]
    ).to(device)

    # Loss function
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    #loss_func = nn.BCELoss(reduction="mean")

    # Optimizer
    #optimizer = optim.SGD(m.parameters(), lr=train_params["learning_rate"])
    optimizer = optim.Adam(m.parameters(), lr=train_params["learning_rate"])

    # Do training
    for epoch_idx in range(train_params["num_epochs"]):
        # Train step
        train_losses, train_accs = train_step(
            m,
            train_loader,
            optimizer,
            loss_func,
            device
        )

        # Test step
        test_losses, test_accs = test_step(
            m,
            test_loader,
            loss_func,
            device
        )

        if verbose:
            print("[Epoch {}/{}] Train Loss/Acc.: {:.4f}/{:.4f}; Test Loss/Acc.: {:.4f}/{:.4f}"\
                .format(
                    epoch_idx+1,
                    train_params["num_epochs"],
                    np.mean(train_losses), np.mean(train_accs),
                    np.mean(test_losses), np.mean(test_accs)
                )
            )

        # Validation step (TODO)
        # ...
        # ...
        # ...
        #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    default_dataset_dir = "/mnt/fs5/nclkong/datasets/bnn_dataset/"
    parser.add_argument('--imagedir', type=str, default=default_dataset_dir)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--numkernels', type=int, default=28)
    parser.add_argument('--kernelsize', type=int, default=19)
    parser.add_argument('--imagesize', type=int, default=30)
    parser.add_argument('--batchsize', type=int, default=200)
    parser.add_argument('--numepochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    # Use GPU or CPU
    if torch.cuda.is_available():
        cuda_idx = int(args.cuda)
        device = torch.device("cuda:{}".format(cuda_idx))
    else:
        device = torch.device("cpu")
    print "Device:", device

    # Training parameters
    train_params = dict()
    train_params["image_dir"] = args.imagedir.lower()
    train_params["num_kernels"] = int(args.numkernels)
    train_params["kernel_size"] = int(args.kernelsize)
    train_params["img_size"] = int(args.imagesize)
    train_params["batch_size"] = int(args.batchsize)
    train_params["num_epochs"] = int(args.numepochs)
    train_params["learning_rate"] = int(args.lr)

    # Do the training
    train(train_params, device)


