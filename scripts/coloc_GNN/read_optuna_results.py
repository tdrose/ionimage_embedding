# %%
# Load autoreload framework when running in ipython interactive session
try:
    import IPython
    # Check if running in IPython
    if IPython.get_ipython(): # type: ignore 
        ipython = IPython.get_ipython()  # type: ignore 

        # Run IPython-specific commands
        ipython.run_line_magic('load_ext','autoreload')  # type: ignore 
        ipython.run_line_magic('autoreload','2')  # type: ignore 
    print('Running in IPython, auto-reload enabled!')
except ImportError:
    # Not in IPython, continue with normal Python code
    pass


import pandas as pd
import matplotlib.pyplot as plt
import ast
# %%


def optuna_results(filename: str) -> pd.DataFrame:

    res = {'Latent size': [],
           'Top-k': [],
           'Bottom-k': [],
           'Encoding': [],
           'Early stopping patience': [],
           'GNN layer type': [],
           'Loss type': [],
           'Number of layers': [],
           'learning rate': [],
           'score': []}
    
    with open(filename , 'r') as f:
        window = False
        counter = 0
        score = 0
        for l in f:
            if l.startswith('++++++++++++++++++++++++++++') and window == False:
                window = True
                counter = 0
            elif l.startswith('++++++++++++++++++++++++++++') and window == True:
                window = False
            elif window == True and counter == 0:
                # Read performance
                score = float(l.split('+ Current trial result ')[1])
                counter = 1
            elif window == True and counter == 1:
                counter = 2

            elif window == True and counter == 2:
                # Read parameters
                # print(l)

                d = ast.literal_eval(l.split('+  ')[1])
                res['score'].append(score)
                for k in d.keys():
                    res[k].append(d[k])
                counter = 0
    print(res)
    return pd.DataFrame(res)



# %%
p1 = optuna_results('/scratch/trose/slurm_optuna_tuning_.46913916.out').sort_values('score')
# %%

p2 = optuna_results('/scratch/trose/slurm_optuna_tuning_.46919848.out').sort_values('score')

# %%
p3 = optuna_results('/scratch/trose/slurm_optuna_tuning_.47302355.out').sort_values('score')
