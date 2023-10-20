# Deep learning approaches for the embedding of ion images

Models to embed ion images from mass spectrometry imaging experiments in a latent space to infer regulatory relationships.
Training of models can be done across different datasets/experiments. 
Dataloaders optimized to work with datasets from the [METASPACE](https://metaspace2020.eu/) knowledge base.

## Available models
| Model name  | Long name | Embedding | Clustering |
| ----------- | --------- | :-------: | :--------: |
| CRL  | Contrastive Representation learning  | ✓ | ✓ |


## Installation

Before installing the package, 
install the fitting pytorch version for your setup (cpu/gpu) from [here](https://pytorch.org/get-started/locally/).

Next, Clone the GitHub repository or pull the latest updates.

In the terminal, navigate to the system directory:

```
cd deep_MultiData_mzClustering
```

Install the package in development mode to apply the latest updates automatically 
after pulling from the GitHub repository:

```
python -m pip install -e .
```


> This code is published under the MIT license.
