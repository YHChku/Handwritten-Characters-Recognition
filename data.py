import os
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_train_and_test_data(data_path, data_transforms, batch_size):
    print("\nStart to load train and test data")
    images_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x]) for x in ['train_data', 'test_data']}

    train_data_loader = torch.utils.data.DataLoader(images_datasets['train_data'], batch_size=batch_size, shuffle=True, num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(images_datasets['test_data'], batch_size=batch_size, shuffle=True, num_workers=4)

    dataset_sizes = {x: len(images_datasets[x]) for x in ['train_data', 'test_data']}

    class_names = images_datasets['train_data'].classes
    num_classes = len(class_names)

    print('The total number of training and testing images: {}'.format(dataset_sizes))
    print('The total number of classes: {}'.format(num_classes))
    print('Class names: {}'.format(class_names))

    return train_data_loader, test_data_loader, dataset_sizes, class_names, num_classes

if __name__ == "__main__":
    data_transforms = {
        'train_data': transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor() ]),
        'test_data': transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
    }

    get_train_and_test_data('img', data_transforms, batch_size=1000)