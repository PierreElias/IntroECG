from torch import nn
import torch
from functools import reduce
from operator import __add__
import torch.nn.functional as F

from constants import INITIAL_KERNEL_NUM, MIN_DROPOUT, MAX_DROPOUT, CONV1_KERNEL1, CONV1_KERNEL2

## use trial.suggest from optuna to suggest hyperparameters 
## https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        self.pad = nn.ZeroPad2d(conv_padding)
        # ZeroPad2d Output: :math:`(N, C, H_{out}, W_{out})` H_{out} is H_{in} with the padding to be added to either side of height
        # ZeroPad2d(2) would add 2 to all 4 sides, ZeroPad2d((1,1,2,0)) would add 1 left, 1 right, 2 above, 0 below
        # n_output_features = floor((n_input_features + 2(paddingsize) - convkernel_size) / stride_size) + 1
        # above creates same padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.bn(x)
        return x


class Multi_2D_CNN_block(nn.Module):
    def __init__(self, in_channels, num_kernel):
        super(Multi_2D_CNN_block, self).__init__()
        conv_block = BasicConv2d
        self.a = conv_block(in_channels, int(num_kernel / 3), kernel_size=(1, 1))

        self.b = nn.Sequential(
            conv_block(in_channels, int(num_kernel / 2), kernel_size=(1, 1)),
            conv_block(int(num_kernel / 2), int(num_kernel), kernel_size=(3, 3))
        )

        self.c = nn.Sequential(
            conv_block(in_channels, int(num_kernel / 3), kernel_size=(1, 1)),
            conv_block(int(num_kernel / 3), int(num_kernel / 2), kernel_size=(3, 3)),
            conv_block(int(num_kernel / 2), int(num_kernel), kernel_size=(3, 3))
        )
        self.out_channels = int(num_kernel / 3) + int(num_kernel) + int(num_kernel)
        # I get out_channels is total number of out_channels for a/b/c
        self.bn = nn.BatchNorm2d(self.out_channels)

    def get_out_channels(self):
        return self.out_channels

    def forward(self, x):
        branch1 = self.a(x)
        branch2 = self.b(x)
        branch3 = self.c(x)
        output = [branch1, branch2, branch3]
        return self.bn(torch.cat(output,
                                 1))  # BatchNorm across the concatenation of output channels from final layer of Branch 1/2/3
        # ,1 refers to the channel dimension


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        self.pad = nn.ZeroPad2d(conv_padding)
        # ZeroPad2d Output: :math:`(N, C, H_{out}, W_{out})` H_{out} is H_{in} with the padding to be added to either side of height
        # ZeroPad2d(2) would add 2 to all 4 sides, ZeroPad2d((1,1,2,0)) would add 1 left, 1 right, 2 above, 0 below
        # n_output_features = floor((n_input_features + 2(paddingsize) - convkernel_size) / stride_size) + 1
        # above creates same padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.bn(x)
        return x


class MyModel(nn.Module):

    def __init__(self, trial):
        super(MyModel, self).__init__()

        multi_2d_cnn = Multi_2D_CNN_block
        conv_block = BasicConv2d
        
        # Define the values in constants.py and import them
        initial_kernel_num=trial.suggest_categorical("kernel_num", INITIAL_KERNEL_NUM)
        dropout = trial.suggest_float('dropout', MIN_DROPOUT, MAX_DROPOUT)
        conv1kernel1=trial.suggest_categorical("conv_1_1", CONV1_KERNEL1)
        conv1kernel2=trial.suggest_categorical("conv_1_2", CONV1_KERNEL2)

        self.conv_1 = conv_block(1, 64, kernel_size=(conv1kernel1, conv1kernel2), stride=(2, 1)) #kernel_size=(7,1), (21,3), (21,1)....

        self.multi_2d_cnn_1a = nn.Sequential(
            multi_2d_cnn(in_channels=64, num_kernel=initial_kernel_num),
            multi_2d_cnn(in_channels=int(initial_kernel_num / 3) + int(initial_kernel_num) + int(initial_kernel_num), num_kernel=initial_kernel_num),
            nn.MaxPool2d(kernel_size=(3, 1))
        )

        self.multi_2d_cnn_1b = nn.Sequential(
            multi_2d_cnn(in_channels=int(initial_kernel_num / 3) + int(initial_kernel_num) + int(initial_kernel_num), num_kernel=initial_kernel_num * 1.5),
            multi_2d_cnn(in_channels=int(initial_kernel_num * 1.5 / 3) + int(initial_kernel_num * 1.5) + int(initial_kernel_num * 1.5), num_kernel=initial_kernel_num * 1.5),
            nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.multi_2d_cnn_1c = nn.Sequential(
            multi_2d_cnn(in_channels=int(initial_kernel_num * 1.5 / 3) + int(initial_kernel_num * 1.5) + int(initial_kernel_num * 1.5), num_kernel=initial_kernel_num * 2),
            multi_2d_cnn(in_channels=int(initial_kernel_num * 2 / 3) + int(initial_kernel_num * 2) + int(initial_kernel_num * 2), num_kernel=initial_kernel_num * 2),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.multi_2d_cnn_2a = nn.Sequential(
            multi_2d_cnn(in_channels=int(initial_kernel_num * 2 / 3) + int(initial_kernel_num * 2) + int(initial_kernel_num * 2), num_kernel=initial_kernel_num * 3),
            multi_2d_cnn(in_channels=int(initial_kernel_num * 3 / 3) + int(initial_kernel_num * 3) + int(initial_kernel_num * 3), num_kernel=initial_kernel_num * 3),
            multi_2d_cnn(in_channels=int(initial_kernel_num * 3 / 3) + int(initial_kernel_num * 3) + int(initial_kernel_num * 3), num_kernel=initial_kernel_num * 4),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2b = nn.Sequential(
            multi_2d_cnn(in_channels=int(initial_kernel_num * 4 / 3) + int(initial_kernel_num * 4) + int(initial_kernel_num * 4), num_kernel=initial_kernel_num * 5),
            multi_2d_cnn(in_channels=int(initial_kernel_num * 5 / 3) + int(initial_kernel_num * 5) + int(initial_kernel_num * 5), num_kernel=initial_kernel_num * 6),
            multi_2d_cnn(in_channels=int(initial_kernel_num * 6 / 3) + int(initial_kernel_num * 6) + int(initial_kernel_num * 6), num_kernel=initial_kernel_num * 7),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2c = nn.Sequential(
            multi_2d_cnn(in_channels=int(initial_kernel_num * 7 / 3) + int(initial_kernel_num * 7) + int(initial_kernel_num * 7), num_kernel=initial_kernel_num * 8),
            multi_2d_cnn(in_channels=int(initial_kernel_num * 8 / 3) + int(initial_kernel_num * 8) + int(initial_kernel_num * 8), num_kernel=initial_kernel_num * 8),
            multi_2d_cnn(in_channels=int(initial_kernel_num * 8 / 3) + int(initial_kernel_num * 8) + int(initial_kernel_num * 8), num_kernel=initial_kernel_num * 8),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2d = nn.Sequential(
            multi_2d_cnn(in_channels=int(initial_kernel_num * 8 / 3) + int(initial_kernel_num * 8) + int(initial_kernel_num * 8), num_kernel=initial_kernel_num * 12),
            multi_2d_cnn(in_channels=int(initial_kernel_num * 12 / 3) + int(initial_kernel_num * 12) + int(initial_kernel_num * 12), num_kernel=initial_kernel_num * 14),
            multi_2d_cnn(in_channels=int(initial_kernel_num * 14 / 3) + int(initial_kernel_num * 14) + int(initial_kernel_num * 14), num_kernel=initial_kernel_num * 16),
        )
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(int(initial_kernel_num * 16 / 3) + int(initial_kernel_num * 16) + int(initial_kernel_num * 16), 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_1(x)
        # N x 1250 x 12 x 64 tensor
        x = self.multi_2d_cnn_1a(x)
        # N x 416 x 12 x 149 tensor
        x = self.multi_2d_cnn_1b(x)
        # N x 138 x 12 x 224 tensor
        x = self.multi_2d_cnn_1c(x)
        # N x 69 x 12 x 298
        x = self.multi_2d_cnn_2a(x)

        x = self.multi_2d_cnn_2b(x)

        x = self.multi_2d_cnn_2c(x)

        x = self.multi_2d_cnn_2d(x)

        x = self.output(x)

        return x