import torch
import torch.nn.functional as functional
import torchvision.transforms as T

import open_clip
import torch
import numpy as np

from ..dataloader.crl_data import CRLdata


class BioMedCLIP:
    def __init__(self, data: CRLdata):
        self.data = data

        tmp = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_'
                                                    '256-vit_base_patch16_224')
        
        self.model, _preprocess_train, _preprocess_val = tmp
        
        # Model requires this image size
        self.resizer = T.Resize((224, 224))

    def inference_embeddings(self, new_data, device: str='cpu', batch_size: int=10):
          
        self.model.to(device)  # type: ignore
        self.model.eval()  # type: ignore

        # Process data
        shp = new_data.shape
        images = torch.tensor(new_data).reshape((shp[0], 1, shp[1], shp[2]))
        images = self.resizer(images)
        images = torch.cat([images] * 3, dim=1)

        # Run in batches
        n_batches = shp[0]//batch_size
        if shp[0]%batch_size > 0:
            n_batches += 1
        
        embed_list = []

        for i in range(n_batches):

            batch = images[(i*batch_size):((i*batch_size)+batch_size)]
            batch.to(device)

            embeddings, _, logits = self.model(batch)  # type: ignore

            # Normalize data
            embeddings = functional.normalize(embeddings, p=2, dim=-1).cpu().detach().numpy()
            embed_list.append(embeddings)

        # merge batches
        return np.concatenate(embed_list)

    def inference_embeddings_train(self, device: str='cpu', batch_size: int=10):
        return self.inference_embeddings(self.data.train_dataset.images, device=device, 
                                         batch_size=batch_size)
    
    def inference_embeddings_val(self, device: str='cpu', batch_size: int=10):
        return self.inference_embeddings(self.data.val_dataset.images, device=device, 
                                         batch_size=batch_size)
    
    def inference_embeddings_test(self, device: str='cpu', batch_size: int=10):
        return self.inference_embeddings(self.data.test_dataset.images, device=device, 
                                         batch_size=batch_size)
