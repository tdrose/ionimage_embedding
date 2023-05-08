import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np


class mzImageDataset(Dataset):
    """
    Images small enough, can be loaded into memory directly
    """
    def __init__(self, 
                 images: np.ndarray,
                 dataset_labels: np.ndarray,
                 ion_labels: np.ndarray,
                 height, width,
                 index,
                 # Rotate images
                 transform=T.RandomRotation(degrees=(0, 360)),
                ):
        
        self.images = images
        self.dataset_labels = dataset_labels
        self.ion_labels = ion_labels
        self.height = height
        self.width = width
        self.transform = transform
        self.index = index

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = torch.Tensor(self.images[idx]).reshape((1, self.height, self.width))
        dataset_label = np.array([self.dataset_labels[idx]])
        ion_label = np.array([self.ion_labels[idx]])
        sample_id = np.array(self.index[idx])
        
        if self.transform:
            image = self.transform(image)

        return image, sample_id, dataset_label, ion_label


def get_dataloader(images: np.ndarray,
                   dataset_labels: np.ndarray,
                   ion_labels: np.ndarray,
                   height,
                   width,
                   index,
                   # Rotate images
                   transform=T.RandomRotation(degrees=(0, 360)),
                   batch_size=128
                  ):
    
    dataset = mzImageDataset(images=images, 
                             dataset_labels=dataset_labels,
                             ion_labels=ion_labels,
                             height=height,
                             width=width,
                             index=index,
                             transform=T.RandomRotation(degrees=(0, 360)))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
