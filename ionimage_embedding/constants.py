from typing import Final, Dict

# Constants for caching of data
ION_IMAGE_DATA : Final[str] = 'IonImageData'
DATASET_DATA : Final[str] = 'DatasetData'
COLOC_NET_DISCRETE_DATA : Final[str] = 'ColocNetDiscreteData'

# Data caching folder
CACHE_FOLDER : Final[str] = '/scratch/model_testing'


# Model show names:
MEAN_COLOC : Final[str] = 'Mean Coloc'
UMAP : Final[str] = 'UMAP'
GNN : Final[str] = 'GNN'
RANDOM : Final[str] = 'Random'
BMC : Final[str] = 'BioMedCLIP'
INFO_NCE : Final[str] = 'infoNCE CNN'

# Colors
MODEL_PALLETE : Final[Dict[str, str]] = {
    MEAN_COLOC: 'darkgrey',
    UMAP: 'gray',
    GNN: '#1aae54ff',
    RANDOM: 'white',
    BMC: '#229cefff',
    INFO_NCE: '#22efecff'
}
