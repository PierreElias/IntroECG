#!/usr/bin/env python
# coding: utf-8

# In[42]:


# NOTE: General consideration about overfitting:
#   1. Are we facing an imbalance data sceniario? How many classes are there and what is the distribution?
#   2. Are you using regularization?
#   3. Adding noise is not recommended at this stage. It's more of a data augmentation technique than regularization.


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
from torchvision.transforms import Resize,ToTensor,Normalize
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import time
import os
from skimage import io
import copy
from torch import optim, cuda
import pandas as pd
import glob

# Useful for examining network
from functools import reduce
from operator import __add__
from torchsummary import summary

from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Image manipulations
from PIL import Image

# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['font.size'] = 10

# Printing out all outputs
# InteractiveShell.ast_node_interactivity = 'all'


# ## Building the Inception Model for ECG Arrays

# In[43]:


class Multi_2D_CNN_block(nn.Module):
    def __init__(self, in_channels, num_kernel):
        super(Multi_2D_CNN_block, self).__init__()
        conv_block=BasicConv2d
        self.a=conv_block(in_channels,int(num_kernel/3),kernel_size= (1,1))

        self.b=nn.Sequential(
            conv_block(in_channels,int(num_kernel/2),kernel_size= (1,1)),
            conv_block(int(num_kernel/2),int(num_kernel),kernel_size= (3,3))
        )

        self.c=nn.Sequential(
            conv_block(in_channels,int(num_kernel/3),kernel_size= (1,1)),
            conv_block(int(num_kernel/3),int(num_kernel/2),kernel_size= (3,3)),
            conv_block(int(num_kernel/2),int(num_kernel),kernel_size= (3,3))
        )
        self.out_channels=int(num_kernel/3)+int(num_kernel)+int(num_kernel)
        #I get out_channels is total number of out_channels for a/b/c
        self.bn = nn.BatchNorm2d(self.out_channels)

    def get_out_channels():
        return self.out_channels

    def forward(self,x):
        branch1=self.a(x)
        branch2=self.b(x)
        branch3=self.c(x)
        output=[branch1,branch2,branch3]
        return self.bn(torch.cat(output,1)) #BatchNorm across the concatenation of output channels from final layer of Branch 1/2/3
        #,1 refers to the channel dimension
        

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        self.pad = nn.ZeroPad2d(conv_padding)
        # ZeroPad2d Output: :math:`(N, C, H_{out}, W_{out})` H_{out} is H_{in} with the padding to be added to either side of height
        # ZeroPad2d(2) would add 2 to all 4 sides, ZeroPad2d((1,1,2,0)) would add 1 left, 1 right, 2 above, 0 below
        # n_output_features = floor((n_input_features + 2(paddingsize) - convkernel_size) / stride_size) + 1
        # above creates same padding 
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size ,bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x=self.pad(x)
        x=self.conv(x)
        x=F.relu(x,inplace=True)
        x = self.bn(x)
        return x


class MyModel(nn.Module):

    def __init__(self, initial_kernel_num=64):
        super(MyModel, self).__init__()

        multi_2d_cnn = Multi_2D_CNN_block
        conv_block= BasicConv2d
        
        self.conv_1=conv_block(1,64,kernel_size=(7,3),stride=(2,1))  
        
        self.multi_2d_cnn_1a=nn.Sequential(
            multi_2d_cnn(in_channels=64,num_kernel=initial_kernel_num),
            multi_2d_cnn(in_channels=149,num_kernel=initial_kernel_num),
            nn.MaxPool2d(kernel_size = (3,1))
        )

        self.multi_2d_cnn_1b=nn.Sequential(
            multi_2d_cnn(in_channels=149,num_kernel=initial_kernel_num*1.5),
            multi_2d_cnn(in_channels=224,num_kernel=initial_kernel_num*1.5),
            nn.MaxPool2d(kernel_size = (3,1))
        )
        self.multi_2d_cnn_1c=nn.Sequential(
            multi_2d_cnn(in_channels=224,num_kernel=initial_kernel_num*2),
            multi_2d_cnn(in_channels=298,num_kernel=initial_kernel_num*2),
            nn.MaxPool2d(kernel_size = (2,1))
        )

        self.multi_2d_cnn_2a=nn.Sequential(
            multi_2d_cnn(in_channels=298,num_kernel=initial_kernel_num*3),
            multi_2d_cnn(in_channels=448,num_kernel=initial_kernel_num*3),
            multi_2d_cnn(in_channels=448,num_kernel=initial_kernel_num*4),
            nn.MaxPool2d(kernel_size = (2,1))
        )
        self.multi_2d_cnn_2b=nn.Sequential(
            multi_2d_cnn(in_channels=597,num_kernel=initial_kernel_num*5),
            multi_2d_cnn(in_channels=746,num_kernel=initial_kernel_num*6),
            multi_2d_cnn(in_channels=896,num_kernel=initial_kernel_num*7),
            nn.MaxPool2d(kernel_size = (2,1))
        )
        self.multi_2d_cnn_2c=nn.Sequential(
            multi_2d_cnn(in_channels=1045,num_kernel=initial_kernel_num*8),
            multi_2d_cnn(in_channels=1194,num_kernel=initial_kernel_num*8),
            multi_2d_cnn(in_channels=1194,num_kernel=initial_kernel_num*8),
            nn.MaxPool2d(kernel_size = (2,1))
        )
        self.multi_2d_cnn_2d=nn.Sequential(
            multi_2d_cnn(in_channels=1194,num_kernel=initial_kernel_num*12),
            multi_2d_cnn(in_channels=1792,num_kernel=initial_kernel_num*14),
            multi_2d_cnn(in_channels=2090,num_kernel=initial_kernel_num*16),
        )
        self.output=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2389 ,1),
            # nn.Sigmoid()
        )



    def forward(self,x):
        x=self.conv_1(x)
        # N x 1250 x 12 x 64 tensor
        x=self.multi_2d_cnn_1a(x)
        # N x 416 x 12 x 149 tensor
        x=self.multi_2d_cnn_1b(x)
        # N x 138 x 12 x 224 tensor
        x= self.multi_2d_cnn_1c(x)
        # N x 69 x 12 x 298
        x= self.multi_2d_cnn_2a(x)

        x= self.multi_2d_cnn_2b(x)

        x= self.multi_2d_cnn_2c(x)

        x= self.multi_2d_cnn_2d(x)

        x=self.output(x)
        
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):


# Convolution as a whole should span at least 1 beat, preferably more
# Input shape = (12,2500)

    def __init__(self):
        super(Net, self).__init__()
        # kernel
        #first layer picking out most basic pattern, 24 output means = 24 channels and 24 different signals for model to detect 
        #these are the lego blocks that get put together in further layer to detect more detailed features
        #can put into matplotlib and see the features that the first layer is learning 
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=20, kernel_size=15),
            nn.BatchNorm1d(20),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(20, 40, 15),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2))

        self.conv3 = nn.Sequential(
            nn.Conv1d(40, 50, 15), 
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 3, stride = 1))
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(50, 75, 15),
            nn.BatchNorm1d(75),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv1d(75, 90, 15),
            nn.BatchNorm1d(90),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 1))

        self.conv6 = nn.Sequential(
            nn.Conv1d(90, 110, 15),
            nn.BatchNorm1d(110),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 1))

        # NOTE: Why you do not use activations on top of your linear layers?
        # NOTE: Of course the last layer nn.Linear(10,1), does not need activation since you are using BCEWithLogitsLoss.
        # NOTE: Too many layers can increae the chance of overfitting is the data is not large enough. Also, using activations is crucial.
        self.fc1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), #takes 1176 channels and averages to 110
            nn.Flatten(),
            nn.Linear(110, 70),
            nn.Dropout(0.5),
            nn.Linear(70,35),
            nn.Linear(35,10),
            nn.Linear(10,1))

    def forward(self, x):
        # print(x.shape)
#         x = x.view(x.shape[0], 12,-1)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        # print(out.shape)
#         out = out.view(x.shape[0], out.size(1) * out.size(2))
        out = self.fc1(out)
        return out

model = Net()
print(model)
from torchsummary import summary

if torch.cuda.is_available():
    model = model.cuda()

summary(model, input_size=(12,2500))





# model=MyModel()
# if torch.cuda.is_available():
#     model = model.cuda()


# from torchsummary import summary
# summary(model, input_size=(1,2500, 12))
#pytorch always (channel,width,height)


def split_train_eval_test(metadata_df_path):

    df = pd.read_csv(metadata_df_path)

    df = df.dropna(subset=["ANY_AMYLOID"])

    pat_ids = list(set(df['PatientID']))

    permuted_pat_ids = np.random.permutation(pat_ids)

    last_train_idx = int(len(permuted_pat_ids) * .80)

    last_eval_idx = int(last_train_idx +
                        ((len(permuted_pat_ids) - last_train_idx) * .70))

    train_pat_ids = permuted_pat_ids[:last_train_idx]
    eval_pat_ids = permuted_pat_ids[last_train_idx:last_eval_idx]
    test_pat_ids = permuted_pat_ids[last_eval_idx:]

    df_train = df[df['PatientID'].isin(train_pat_ids)]
    df_eval = df[df['PatientID'].isin(eval_pat_ids)]
    df_test = df[df['PatientID'].isin(test_pat_ids)]

    print(
        "Train pts/pct: {}, {} \n Eval pts/pct: {}, {} \n Test pts/pct: {}, {}"
        .format(len(df_train),
                len(df_train) / float(len(df)), len(df_eval),
                len(df_eval) / float(len(df)), len(df_test),
                len(df_test) / float(len(df))))

    df_train.to_csv('/labs/cardiac-amyloid/CADnet/data/train_metadata_5k.csv', index=False)
    df_eval.to_csv('/labs/cardiac-amyloid/CADnet/data/eval_metadata_5k.csv', index=False)
    df_test.to_csv('/labs/cardiac-amyloid/CADnet/data/test_metadata_5k.csv', index=False)

def add_gaussian_noise(signal):
    noise=np.random.normal(0,0.05,signal.shape)
    return (signal+noise)


split_train_eval_test('/labs/cardiac-amyloid/amyloid_simple_run.csv')

train_df = pd.read_csv('/labs/cardiac-amyloid/CADnet/data/train_metadata_5k.csv')
eval_df = pd.read_csv('/labs/cardiac-amyloid/CADnet/data/eval_metadata_5k.csv')

npyfilespath ="/labs/cardiac-amyloid/amyloid_waveforms/"
os.chdir(npyfilespath)
npfiles= glob.glob( "*.npy")
npfiles.sort()
npdf = pd.DataFrame({'filename': npfiles})
print(npdf.head())
print(train_df.head())
train_df_available = train_df.merge(npdf)
eval_df_available = eval_df.merge(npdf)


train_df_available = train_df_available[:5000]
eval_df_available = eval_df_available[:1000]


print('train ECG count:' ,len(train_df_available), '\n eval ECG count:', len(eval_df_available))


# fpath_train ="/labs/cardiac-amyloid/CADnet/data/amyloid_concat_train.npy"
# fpath_eval ="/labs/cardiac-amyloid/CADnet/data/amyloid_concat_eval.npy"
npyfilespath ="/labs/cardiac-amyloid/amyloid_waveforms/"
os.chdir(npyfilespath)
npfiles= glob.glob( "*.npy")
npfiles.sort()
all_arrays_train = []
all_arrays_eval = []

# all_arrays_train = np.zeros((len(train_df_available),2500,12,1))
# all_arrays_eval = np.zeros((len(eval_df_available),2500,12,1))

# all_arrays_train = np.zeros((len(train_df_available),1,2500,12))
# all_arrays_eval = np.zeros((len(eval_df_available),1,2500,12))

filelist_train = train_df_available['filename']
filelist_eval = eval_df_available['filename']

#If trying to test model quickly use smaller total dataset or change dataloader to load npy file batch by batch

for i, npfile in enumerate(filelist_train):
    x = 0
    i = 0
    try:
        file=np.load(npyfilespath + npfile)
        file=np.reshape(file,(2500,12))
        all_arrays_train.append(file)
        x += 1
        i += 1
    except:
        continue
    if i % 1 == 100:
        print("{i} EKGs have been written to array")

for i, npfile in enumerate(filelist_eval):
    x = 0
    i = 0
    try:
        file=np.load(npyfilespath + npfile)
        file=np.reshape(file,(2500,12))
        all_arrays_eval.append(file)
        x += 1
        i += 1
    except:
        continue
    if i % 1 == 100:
        print("{i} EKGs have been written to array")

# for i, npfile in enumerate(filelist_train):
#     x = 0
#     i = 0
#     start = time()
#     try:
#         all_arrays_train[i]=np.load(npyfilespath + npfile)
#         x += 1
#         i += 1
#         # all_arrays_train.append(np.load(npyfilespath + npfile))                                                                                                                           
#     except:
#         i += 1
#     if i % 1 == 0:
#         print(time() - start, time()-start_loop)


print('train array count:' ,len(all_arrays_train), '\n eval array count:', len(all_arrays_eval))

np.save('/labs/cardiac-amyloid/CADnet/data/amyloid_concat_train_1DCNN.npy', all_arrays_train)
np.save('/labs/cardiac-amyloid/CADnet/data/amyloid_concat_eval_1DCNN.npy', all_arrays_eval)

trainData=np.load('/labs/cardiac-amyloid/CADnet/data/amyloid_concat_train_1DCNN.npy') #Load input data. Input data should be compiled as numpy arrays with (sample numbers, time, lead, 1)
evalData=np.load('/labs/cardiac-amyloid/CADnet/data/amyloid_concat_eval_1DCNN.npy')




SHAPE=(1,2500,12)

assert trainData.shape[1:] == (2500,12), "train is not X,2500,12"
assert evalData.shape[1:] == (2500,12), "eval is not X,2500,12"

label_list_train = train_df_available['ANY_AMYLOID']
label_list_eval = eval_df_available['ANY_AMYLOID']

#these next three lines should only be here for when you're actively uploading more ECG data since you last created trainData
label_list_train = label_list_train[:len(trainData)]
label_list_eval = label_list_eval[:len(evalData)]

from collections import Counter
print(*label_list_train[:100],sep=',')
print(*label_list_eval[:100],sep=',')
print('train label count: ', Counter(label_list_train), '\n eval label count: ', Counter(label_list_eval))


X_train=np.array(trainData)
y_train=np.array(label_list_train)
X_test=np.array(evalData)
y_test=np.array(label_list_eval)


SHAPE=(12,2500) #Shape of the input. ECG model takes 2500 points sampled at 4 micro-seconds/point for 12 lead.
batch_size=20 #Define sample numbers per batch. OOM error at 32
dlen=X_train.shape[0]

print("before reshape:",X_train.shape)
X_train = X_train.reshape((dlen,)+SHAPE)
# X_train = X_train.reshape(batch_size,2500,12)
print("after reshape:",X_train.shape)
X_test = X_test.reshape((X_test.shape[0],)+SHAPE)


# NOTE:
# 1. It's good that you are using pin_memory=True instead of returning cuda tensors.
#    But be aware that pin_memory=True may slows down your training due to the overhead.
#    Read more at: https://discuss.pytorch.org/t/issue-with-dataloader-using-pin-memory-true/36643
# 2. As you use shuffle=True, for each epoch, the data is shuffled and you should not worry about it.
y_train = torch.FloatTensor(y_train)
X_train = TensorDataset(torch.from_numpy(X_train), y_train)
train_loader = DataLoader(X_train, batch_size=20, pin_memory=True, shuffle=True)

y_test = torch.FloatTensor(y_test)
X_test= TensorDataset(torch.from_numpy(X_test), y_test)
test_loader = DataLoader(X_test, batch_size=20, pin_memory=True, shuffle=True)


# ## Training the Network

# In[64]:




# Whether to train on a gpu
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')


# if (cuda_idx == 0 or cuda_idx == 1) and torch.cuda.is_available():
#     device = torch.device("cuda:{}".format(cuda_idx))
#     dataloader_kwargs = {'pin_memory': True}
# elif cuda_idx == 2:
#     device = torch.device("cuda")
#     dataloader_kwargs = {}
# elif cuda_idx < 0:
#     device = torch.device("cpu")
#     dataloader_kwargs = {}

# Number of gpus
if train_on_gpu:
    gpu_count = cuda.device_count()
    print(f'{gpu_count} gpus detected.')
    torch.cuda.set_device(1)
    print('Using:', torch.cuda.current_device())
    # if gpu_count > 1:
    #     multi_gpu = True
    # else:
    #     multi_gpu = False

def train(model,
          cuda_idx,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=5,
          n_epochs=20,
          print_every=1):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []
    if (cuda_idx == 0 or cuda_idx == 1) and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(cuda_idx))
        dataloader_kwargs = {'pin_memory': True}
    elif cuda_idx == 2:
        device = torch.device("cuda")
        dataloader_kwargs = {}
    elif cuda_idx < 0:
        device = torch.device("cpu")
        dataloader_kwargs = {}
    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0
        running_corrects=0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            # NOTE: Even if you do not use the train_on_gpu condition, the device=device does the job is no CUDA is available!
            if train_on_gpu:
                data, target = data.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float)
                model = model.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            # output = model(data.float())
            output = model(data)
            # print("Output:",output)

            # Loss and backpropagation of gradients
            loss = criterion(output, target.unsqueeze(1))
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            pred = torch.round(torch.sigmoid(output))
            # print("Predictions:",pred)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        model.epochs += 1

        # Don't need to keep track of gradients
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()

            # Validation loop
            for data, target in valid_loader:
                # Tensors to gpu
                if train_on_gpu:
                    data, target = data.to(device=device, dtype=torch.float), target.to(device=device, dtype=torch.float)

                # Forward pass
                output = model(data)

                # Validation loss
                # NOTE: The criterion looks ok but we need to sit and do some debugging with the real data.
                # I can't say by just looking at it. In practice, we should see what are the outputs.
                loss = criterion(output, target.unsqueeze(1))
                # Multiply average loss times the number of examples in batch
                valid_loss += loss.item() * data.size(0)

                # Calculate validation accuracy
                pred = torch.round(torch.sigmoid(output))
                running_corrects += torch.sum(pred == target.data)

                # print(running_corrects,len(valid_loader.dataset))
                correct_tensor = pred.eq(target.data.view_as(pred))
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples
                valid_acc += accuracy.item() * data.size(0)

            # Calculate average losses
            train_loss = train_loss / len(trainData)
            valid_loss = valid_loss / len(evalData)

            # Calculate average accuracy
            train_acc = train_acc / len(trainData)
            valid_acc = valid_acc / len(evalData)
            epoch_acc = running_corrects.double() / len(evalData)
            

            history.append([train_loss, valid_loss, train_acc, valid_acc])

            # Print training and validation results
            if (epoch + 1) % print_every == 0:
                # print("epoch_acc",epoch_acc)
                print(
                    f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                )
                print(
                    f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                )

            # Save the model if validation loss decreases
            if valid_loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_file_name)
                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = valid_loss
                valid_best_acc = valid_acc
                best_epoch = epoch

            # Otherwise increment count of epochs with no improvement
            else:
                epochs_no_improve += 1
                # Trigger early stopping
                # if epochs_no_improve >= max_epochs_stop:
                #     print(
                #         f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                #     )
                #     total_time = timer() - overall_start
                #     print(
                #         f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                #     )

                #     # Load the best state dict
                #     model.load_state_dict(torch.load(save_file_name))
                #     # Attach the optimizer
                #     model.optimizer = optimizer

                #     # Format history
                #     history = pd.DataFrame(
                #         history,
                #         columns=[
                #             'train_loss', 'valid_loss', 'train_acc',
                #             'valid_acc'
                #         ])
                #     return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    print("\n\n----------------\n\n")
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history


# model = model.float()

from torch import optim
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss() #re-add the sigmoid function in model if using BCE
# optimizer_ft = optim.Adam(model.parameters(), lr=0.00003)
optimizer_ft = optim.AdamW(model.parameters(), lr=0.00003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
#add L2 regularization penalty through weight_decay 
NUM_EPOCHS = 40
save_file_name = 'CADnet.pt'
checkpoint_path = 'CADnet.pth'
cuda_idx = 1


model, history = train(
    model,
    cuda_idx, 
    criterion,
    optimizer_ft,
    train_loader,
    test_loader,
    save_file_name=save_file_name,
    max_epochs_stop=4,
    n_epochs=NUM_EPOCHS,
    print_every=1)


# plt.figure(figsize=(8, 6))
# for c in ['train_acc', 'valid_acc']:
#     plt.plot(
#         100 * history[c],40=c)
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Average Accuracy')
# plt.title('Training and Validation Accuracy')


