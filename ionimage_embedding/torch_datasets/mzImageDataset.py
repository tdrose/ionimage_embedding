import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np


class mzImageDataset(Dataset):
    """
    Torch dataset wrapper to that stores imagesm dataset labels, ion labels, and ion composition.
    """
    def __init__(self, 
                 images,
                 dataset_labels,
                 ion_labels,
                 ion_composition,
                 height, width,
                 index,
                 # Rotate images
                 transform=None,
                ):
        
        self.images = images
        self.dataset_labels = dataset_labels
        self.ion_labels = ion_labels
        self.height = height
        self.width = width
        self.transform = transform
        self.index = index
        self.ion_composition = ion_composition

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        transformed_image = torch.Tensor(self.images[idx]).reshape((1, self.height, self.width))
        dataset_label = np.array([self.dataset_labels[idx]])
        ion_label = np.array([self.ion_labels[idx]])
        ion_comp = np.array(self.ion_composition[idx])
        sample_id = np.array(self.index[idx])
        
        if self.transform is not None:
            transformed_image = self.transform(transformed_image)

        untransformed_images = torch.Tensor(self.images[idx]).reshape((1, self.height, self.width))
        
        return transformed_image, sample_id, dataset_label, ion_label, untransformed_images, ion_comp


def get_iid_dataloader(images: np.ndarray,
                       dataset_labels: np.ndarray,
                       ion_labels: np.ndarray,
                       ion_composition,
                       height,
                       width,
                       index,
                       # Rotate images
                       transform=T.RandomRotation(degrees=(0, 360)),
                       batch_size=128,
                       val=False
                      ):
    
    dataset = mzImageDataset(images=images, 
                             dataset_labels=dataset_labels,
                             ion_labels=ion_labels,
                             ion_composition=ion_composition,
                             height=height,
                             width=width,
                             index=index,
                             transform=transform)
    if val:
        return DataLoader(dataset, batch_size=len(images), shuffle=False)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
