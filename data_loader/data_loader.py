from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def get_dataloader(
    path,
    batch_size,
):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(size=32, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    os.makedirs(path, exist_ok=True)

    train_dataset = datasets.CIFAR10(
        path, train=True, transform=train_transform, download=True
    )
    test_dataset = datasets.CIFAR10(
        path, train=False, transform=test_transform, download=True
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader
