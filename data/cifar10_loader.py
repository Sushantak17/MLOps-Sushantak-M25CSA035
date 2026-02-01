from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms


class CIFAR10CustomDataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_cifar10_dataloaders(
    data_dir="./data",
    batch_size=128,
    num_workers=2,
    val_split=0.1
):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    full_train_dataset = CIFAR10CustomDataset(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True
    )

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    val_dataset.dataset.transform = test_transform

    test_dataset = CIFAR10CustomDataset(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
