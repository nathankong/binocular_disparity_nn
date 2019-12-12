import numpy as np

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, random_split

from functools import partial

def image_loader(device, do_transforms, image_name):
    #print(image_name)
    image = np.load(image_name)

    if do_transforms:
        image = np.copy(image).astype('uint8')
        loader = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image = loader(image)
    else:
        image = torch.from_numpy(image).float()
        image = np.transpose(image, (2,0,1))

    image = Variable(image, requires_grad=False)
    return image.to(device)

def acquire_datasets(images_dir, device, do_transforms):
    # Load dataset
    dataset = DatasetFolder(
        images_dir,
        extensions="npy",
        loader=partial(image_loader, device, do_transforms)
    )

    # Split dataset to train/test
    train_size = int(0.7 * len(dataset))
    test_size = int((len(dataset) - train_size) / 2)
    val_size = len(dataset) - train_size - test_size
    train_set, test_set, val_set = random_split(
        dataset,
        [train_size, test_size, val_size]
    )
    return train_set, test_set, val_set

def acquire_data_loaders(
    images_dir,
    batch_size,
    device,
    do_transforms=False
):
    # Acquire train/test/validation datasets
    train_set, test_set, val_set = acquire_datasets(images_dir, device, do_transforms)

    # Initialize data loaders
    train_data_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    test_data_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True
    )
    val_data_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=True
    )

    return train_data_loader, test_data_loader, val_data_loader

def compute_accuracy(predictions, labels, total_samples):
    # Predictions and labels should be on the same torch.device and
    # are of type torch.Tensor. total_samples should be a float.

    preds = predictions.argmax(dim=1, keepdim=True)
    correct = preds.eq(labels.view_as(preds)).sum().item()
    return correct / float(total_samples)

if __name__ == "__main__":
    im_dir = "/mnt/fs5/nclkong/datasets/bnn_dataset/"
    batch_size = 20
    device = torch.device("cpu")

    tr, te, va = acquire_datasets(im_dir, device, False)
    d,t = te[100]
    print(type(d), t)
    assert 0

    tr, te, va = acquire_data_loaders(im_dir, batch_size, device)

    #for i, (data, label) in enumerate(tr):
    #    print "Train batch {}/{}: {} {}".format(i+1, len(tr), data.size(), label.size())

    for i, (data, label) in enumerate(te):
        print "Test batch {}/{}: {} {}".format(i+1, len(te), data.size(), label.size())
        print label
        assert 0

    #for i, (data, label) in enumerate(va):
    #    print "Val batch {}/{}: {} {}".format(i+1, len(va), data.size(), label.size())


