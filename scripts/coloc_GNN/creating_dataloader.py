# %%
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from ionimage_embedding.dataloader.constants import CACHE_FOLDER


# %%

dataset = MoleculeNet(root=CACHE_FOLDER, name="Tox21")

# %%
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# %%
tmp = next(iter(train_loader))
print(tmp)
print(f'Batch size: {len(tmp.y)}')
print(f'X shape: {tmp.x.shape}')
print(f'Edge index shape: {tmp.edge_index.shape}')

# %%

tmp.x
# %%
