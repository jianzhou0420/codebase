

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms


def get_cifar10(batch_size):
    # Define transformations to be applied to the data
    transform = transforms.Compose([
        transforms.ToTensor(),   # Convert images to tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values to range [-1, 1]
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='/home/jian/git_all/datasets/cifar10/data', train=True,
                                            download=True, transform=transform)

    # Filter the data based on the specified label
    label_to_extract = 0 
    indices = [i for i, (_, label) in enumerate(trainset) if label == label_to_extract]

    # Split the filtered indices into training and validation sets
    split_ratio = 0.8  # 80% training, 20% validation
    split = int(len(indices) * split_ratio)
    train_indices, val_indices = indices[:split], indices[split:]

    # Define samplers for training and validation sets
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Define data loaders using samplers
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,num_workers=15)
    val_loader = DataLoader(trainset, batch_size=batch_size, sampler=val_sampler,num_workers=15)
    
    return train_loader,val_loader