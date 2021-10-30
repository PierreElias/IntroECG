# Repository Contents

constants.py: File containing all the tuneable parameters and the current values to try

paths.py: File containing all the paths to the data files on the remote server

tuningfunctions.py: File containing functions used in example.ipynb

models.py: File containing the model class definitions to be tuned, tuneable parameters should be placed in constants.py and called when using trial.suggest

example.ipynb: Jupyter Notebook detailing the necessary steps to optimize hyperparameters in Ignite+Optuna, including loading data, defining the objective, running trials, and viewing the Tensorboard logs.
