from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_data(im_size, batch_size, workers, dataset, data_path):
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if dataset == 'cifar10':
        dataset = datasets.CIFAR10(data_path, train=True, transform=transform, download=True)
    else:
        assert False, f"Unknwn dataset: {dataset}"
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=workers,
                            drop_last=True,
                            pin_memory=True)
    return dataloader
