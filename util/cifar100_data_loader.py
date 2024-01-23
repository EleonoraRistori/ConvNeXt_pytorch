import torchvision.transforms as transforms
import torchvision
import torch


def basic_transform():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [0.507, 0.487, 0.441]],
                             std=[x / 255.0 for x in [0.267, 0.256, 0.276]])
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [0.507, 0.487, 0.441]],
                             std=[x / 255.0 for x in [0.267, 0.256, 0.276]])
    ])

    return transform_train, transform_test


def load_data(batch_size):

    transform_train, transform_test = basic_transform()

    dataset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=True,
                                            download=True, transform=transform_train)
    # create a split for train/validation. We can use early stop
    trainset, valset = torch.utils.data.random_split(dataset, [45000,
                                                               5000])  # si suddivide il train in train (45000 immagini) e validation (5000 immagini)

    testset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=False,
                                            download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0,
                                              drop_last=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False, num_workers=0,
                                            drop_last=False)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0,
                                             drop_last=False)
    return trainloader, valloader, testloader
