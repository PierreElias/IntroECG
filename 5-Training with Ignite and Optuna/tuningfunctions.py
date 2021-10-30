import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
# from imblearn.under_sampling import RandomUnderSampler
import cv2
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, \
    average_precision_score
from sklearn.model_selection import train_test_split
import time
import os
from pathlib import Path
from skimage import io
import copy
from torch import optim, cuda
import pandas as pd
import glob
from collections import Counter
# Useful for examining network
from functools import reduce
from operator import __add__
# from torchsummary import summary
import seaborn as sns
import warnings
# warnings.filterwarnings('ignore', category=FutureWarning)
from PIL import Image
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# Useful for examining network
from functools import reduce
from operator import __add__
from torchsummary import summary

# from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns

import warnings
# warnings.filterwarnings('ignore', category=FutureWarning)

# Image manipulations
from PIL import Image

# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt




import optuna
from ignite.engine import Engine
from ignite.engine import create_supervised_evaluator
from ignite.engine import create_supervised_trainer
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss, Precision, Recall, Fbeta
from ignite.contrib.metrics.roc_auc import ROC_AUC
from ignite.handlers import ModelCheckpoint, global_step_from_engine, Checkpoint, DiskSaver
from ignite.handlers.early_stopping import EarlyStopping
from ignite.contrib.handlers import TensorboardLogger

import models



def get_data_loaders(X_train, X_test, y_train, y_test):
  
    batch_size = 10
    dlen = X_train.shape[0]


    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    X_test = TensorDataset(torch.FloatTensor(X_test), y_test)
    test_loader = DataLoader(X_test, batch_size=batch_size, pin_memory=True, shuffle=True)

    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_train = TensorDataset(torch.FloatTensor(X_train), y_train)
    train_loader = DataLoader(X_train, batch_size=batch_size, pin_memory=True, shuffle=True)

    return train_loader, test_loader


def get_criterion(y_train):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Train on: {device}')


    LABEL_WEIGHTS = []

   
    class_counts = np.bincount(y_train).tolist() #y_train.value_counts().tolist()
    weights = torch.tensor(np.array(class_counts) / sum(class_counts))
    # assert weights[0] > weights[1]
    print("CLASS 0: {}, CLASS 1: {}".format(weights[0], weights[1]))
    weights = weights[0] / weights
    print("WEIGHT 0: {}, WEIGHT 1: {}".format(weights[0], weights[1]))
    LABEL_WEIGHTS.append(weights[1])

    print("Label Weights: ", LABEL_WEIGHTS)
    cuda_idx = 0
    LABEL_WEIGHTS = torch.stack(LABEL_WEIGHTS)
    LABEL_WEIGHTS = LABEL_WEIGHTS.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=LABEL_WEIGHTS)
    criterion.to(device)
    
    return criterion

def thresholded_output_transform(output):
            y_pred, y = output
            y_pred = torch.round(torch.sigmoid(y_pred))
            return y_pred, y
def class0_thresholded_output_transform(output):
            y_pred, y = output
            y_pred = torch.round(torch.sigmoid(y_pred))
            y=1-y
            y_pred=1-y_pred
            return y_pred, y
        

class Objective(object):
    def __init__(self, model_name, criterion, train_loader, test_loader, optimizers, lr_lower, lr_upper, metric, max_epochs, early_stopping_patience=None, lr_scheduler=False, step_size=None, gamma=None):
        # Hold this implementation specific arguments as the fields of the class.
        self.model_name=model_name
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.optimizers = optimizers
        self.criterion=criterion
        self.metric = metric
        self.max_epochs=max_epochs
        self.lr_lower=lr_lower
        self.lr_upper=lr_upper
        self.early_stopping_patience=early_stopping_patience
        self.lr_scheduler=lr_scheduler
        self.step_size=step_size
        self.gamma=gamma

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        model = getattr(models, self.model_name)(trial)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            model.cuda(device)

        val_metrics = {
        "accuracy": Accuracy(output_transform=thresholded_output_transform),
        "loss": Loss(self.criterion),
        "roc_auc": ROC_AUC(output_transform=thresholded_output_transform),
        "precision": Precision(output_transform=thresholded_output_transform),
        "precision_0": Precision(output_transform=class0_thresholded_output_transform),
        "recall": Recall(output_transform=thresholded_output_transform),
        "recall_0": Recall(output_transform=class0_thresholded_output_transform),
        }
        val_metrics["f1"]=Fbeta(beta=1.0, average=False, precision=val_metrics['precision'], recall=val_metrics['recall'])
        val_metrics["f1_0"]=Fbeta(beta=1.0, average=False, precision=val_metrics['precision_0'], recall=val_metrics['recall_0'])




        optimizer_name = trial.suggest_categorical("optimizer", self.optimizers)
        learnrate = trial.suggest_loguniform("lr", self.lr_lower, self.lr_upper)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learnrate)

        trainer = create_supervised_trainer(model, optimizer, self.criterion, device=device)
        train_evaluator = create_supervised_evaluator(model, metrics= val_metrics, device=device)
        evaluator = create_supervised_evaluator(model, metrics= val_metrics, device=device)

        # Register a pruning handler to the evaluator.
        pruning_handler = optuna.integration.PyTorchIgnitePruningHandler(trial, self.metric, trainer)
        evaluator.add_event_handler(Events.COMPLETED, pruning_handler)

        def score_fn(engine):
            score = engine.state.metrics[self.metric]
            return score if self.metric!='loss' else -score

        #early stopping
        if self.early_stopping_patience is not None:
            es_handler = EarlyStopping(patience=self.early_stopping_patience, score_function=score_fn, trainer=trainer)
            evaluator.add_event_handler(Events.COMPLETED, es_handler)

        #checkpointing
        to_save = {'model': model}

        checkpointname='checkpoint'
        for key, value in trial.params.items():
          checkpointname+=key+': '+str(value)+', '
        checkpoint_handler = Checkpoint(to_save, DiskSaver(checkpointname, create_dir=True),
                         filename_prefix='best', score_function=score_fn, score_name="val_metric",
                         global_step_transform=global_step_from_engine(trainer))

        evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler)

        #  Add lr scheduler
        if self.lr_scheduler is True:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
            trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: scheduler.step())

        
        
        #print metrics on each epoch completed
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
          train_evaluator.run(self.train_loader)
          metrics = train_evaluator.state.metrics
          print("Training Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f} roc_auc: {:.4f} \n"
              .format(engine.state.epoch, metrics["accuracy"], metrics["loss"], metrics['roc_auc']))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(self.test_loader)
            metrics = evaluator.state.metrics
            print("Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f} ROC_AUC: {:.4f}"
            "\nClass 1 Precision: {:.4f} Class 1 Recall: {:.4f} Class 1 F1: {:.4f}"
            "\nClass 0 Precision: {:.4f} Class 0 Recall: {:.4f} Class 0 F1: {:4f} \n"
              .format(engine.state.epoch, metrics["accuracy"], metrics["loss"], metrics['roc_auc'], 
                      metrics['precision'], metrics['recall'], metrics['f1'], 
                      metrics['precision_0'], metrics['recall_0'], metrics["f1_0"]))

        #Tensorboard logs
        logname=''
        for key, value in trial.params.items():
          logname+=key+': '+str(value)+','
        tb_logger = TensorboardLogger(log_dir=logname)

        for tag, evaluator in [("training", train_evaluator), ("validation", evaluator)]:
          tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),)

        #run the trainer
        trainer.run(self.train_loader, max_epochs=self.max_epochs)

        #load the checkpoint with the best validation metric in the trial
        to_load = to_save
        checkpoint = torch.load(checkpointname+'/'+checkpoint_handler.last_checkpoint)
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

        evaluator.run(self.test_loader)

        tb_logger.close()
        return evaluator.state.metrics[self.metric]
    

def run_trials(objective, pruner, num_trials, direction): 
    pruner = pruner
    study = optuna.create_study(direction=direction, pruner=pruner)
    study.optimize(objective, n_trials=num_trials, gc_after_trial=True)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
          print("    {}: {}".format(key, value))