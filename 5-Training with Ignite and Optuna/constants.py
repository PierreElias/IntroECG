import optuna

MODEL_NAME='MyModel' #the name of the class in models.py you want to tune
OPTIMIZERS=['AdamW', 'RMSprop'] #the list of optimizers to sample from
MAX_EPOCHS=100 #the max number of epochs to train for each hyperparameter combination
METRIC='roc_auc' #the metric to optimize for across trials, also used for early stopping within a trial and pruning across trials, full list of possible values in tuningfunctions.Objective
MIN_LR=3e-5 #the mininum learning rate to sample from on a log scale
MAX_LR=3e-3 #the maximum learning rate to sample from on a log scale
PATIENCE=10 #the early stopping patience, can be set to None
SCHEDULER=False #bool for whether to use the learning rate scheduler
STEP=1 #the learning rate scheduler step size, if SCHEDULER=True, cannot be None
GAMMA=0.975 #the learning rate gamma, if SHCEDULER=True, cannot be None

PRUNER=optuna.pruners.NopPruner() #the optuna pruner to use across trials
NUM_TRIALS=5 #the number of different hyperparameter combination to try
DIRECTION='maximize' #the direction of the metric to optimize towards ex: 'loss' = 'minimize', 'roc_auc' = 'maximize'



## MyModel specific parameters
INITIAL_KERNEL_NUM = [4,8,16,32,64]
MIN_DROPOUT = 0
MAX_DROPOUT = 1
CONV1_KERNEL1 = [7,21]
CONV1_KERNEL2 = [1,3]