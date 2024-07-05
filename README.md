# Deep learning-based embedding of spatial metabolomics data

This repositories contains implementations of models to learn latent representations of ion images.
The code provides model implementations and dataloaders optimized 
to work with datasets from the [METASPACE](https://metaspace2020.eu/) knowledge base.

This repository is work in progress. 
New models will be implemented and further model evaluation is currently being performed.
All class/function APIs might be changed without anouncements or versioning,
until a first stable version will be published.

The work in currently on hold.
In the meantime, you may cite:

> Rose et al. 
> "Generalizing Colocalization to Capture Molecular Relationships in Spatial Metabolomics Using Graph Neural Networks"
> **EMBO | EMBL Symposium AI and biology**, 2024, presented as poster


## Available models
| Model name  | Long name | Short description | Further information |
| ----------- | --------- | :-------: | :--------: |
| gnnDiscrete | Graph Neural Network with discrete Edges| GAE model for network representation of spatial metabolomics datasets, that learn a latent representation for molecules| Best performing model, from latest evaluation |
| CRL  | Contrastive Representation learning  | CNN models with different contrastive loss functions that learn a latent representation from ion images across different datasets. Various architectures are available: Vision Transformer, ResNet, Custom CNN | Adapted from [Gao et al.](https://github.com/DanGuo1223/mzClustering), to work across multiple datasets and new loss functions available. |
| CVAE | Conditional Variational Autoencoder| Model based on CNN networks learn a latent representation of ion images conditioned on the molecule identity | - |
| Baseline | Mean coloc / UMAP embedding | As baseline models, the pairwise mean colocalization is available, and the UMAP embedding of the mean colocalization network, providing a latent representation for molecules | - |


## Usage

Examples for the training, hyperparameter tuning, and evaluation of models on different training
scenarios can be found in the directory `scripts/`.

A full tutorial and documentation is not available yet, 
but will made once the project is fully published.

### For team members:

A more detailed description of the project can be found 
[here](https://docs.google.com/document/d/1ymz_QkjclIELe1EUq3NTapcnhDboOSiP1rANfnY1v3M/edit?usp=sharing) 
(for lab members only). 

## License
This code is published under the GPLv3 license. 
It includes code from the following repositories:

* [https://github.com/DanGuo1223/mzClustering](https://github.com/DanGuo1223/mzClustering), 2022 MIT license
* [https://github.com/tdrose/deep_mzClustering](https://github.com/tdrose/deep_mzClustering), 2023 MIT license

Usage of external code has been marked at the top of each file.