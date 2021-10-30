class Inception_Tabular_7_1(nn.Module):
    def __init__(self, tabular_model, len_tabular_model_output, initial_kernel_num=64, initialize_parameters=False):
        super(Inception_Tabular_7_1, self).__init__()

        self.conv_1 = BasicConv2d(in_channels=1, out_channels=64, kernel_size=(7, 1), stride=(2, 1))

        self.multi_2d_cnn_1a = nn.Sequential(
          Multi_2D_CNN_block(in_channels=64, num_kernel=initial_kernel_num),
          Multi_2D_CNN_block(in_channels=149, num_kernel=initial_kernel_num),
          nn.MaxPool2d(kernel_size=(3, 1))
        )

        self.multi_2d_cnn_1b = nn.Sequential(
          Multi_2D_CNN_block(in_channels=149, num_kernel=initial_kernel_num * 1.5),
          Multi_2D_CNN_block(in_channels=224, num_kernel=initial_kernel_num * 1.5),
          nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.multi_2d_cnn_1c = nn.Sequential(
          Multi_2D_CNN_block(in_channels=224, num_kernel=initial_kernel_num * 2),
          Multi_2D_CNN_block(in_channels=298, num_kernel=initial_kernel_num * 2),
          nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.multi_2d_cnn_2a = nn.Sequential(
          Multi_2D_CNN_block(in_channels=298, num_kernel=initial_kernel_num * 3),
          Multi_2D_CNN_block(in_channels=448, num_kernel=initial_kernel_num * 3),
          Multi_2D_CNN_block(in_channels=448, num_kernel=initial_kernel_num * 4),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2b = nn.Sequential(
          Multi_2D_CNN_block(in_channels=597, num_kernel=initial_kernel_num * 5),
          Multi_2D_CNN_block(in_channels=746, num_kernel=initial_kernel_num * 6),
          Multi_2D_CNN_block(in_channels=896, num_kernel=initial_kernel_num * 7),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2c = nn.Sequential(
          Multi_2D_CNN_block(in_channels=1045, num_kernel=initial_kernel_num * 8),
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 8),
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 8),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2d = nn.Sequential(
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 12),
          Multi_2D_CNN_block(in_channels=1792, num_kernel=initial_kernel_num * 14),
          Multi_2D_CNN_block(in_channels=2090, num_kernel=initial_kernel_num * 16),
        )
        self.output_before_fc = nn.Sequential(
          nn.AdaptiveAvgPool2d((1, 1)),
          nn.Flatten(),
          nn.Dropout(0.5),
          # nn.Sigmoid()
        )

        self.tabular_model = tabular_model
        self.len_tabular_model_output = len_tabular_model_output

        # the extra features are for the output for the tabular data model
        self.output = nn.Linear(2389 + len_tabular_model_output, 1)

        if initialize_parameters:
          for m in self.modules():
            model_utils.initialize_parameters(m)

      def forward(self, x_and_tabular):
        """
        Parameters
        ----------
        x_and_tabular tuple of (image input, tabular features)
        where image input and tabular features are both tensors of size BATCHSIZE s.t.
        BATCHSIZE is a batch size > 1.
        Returns
        -------
        tensor of BATCHSIZE with forward pass results
        """
        x, tabular = x_and_tabular
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

        x = self.output_before_fc(x)

        tabular_model_output = self.tabular_model(tabular)
        len_tabular_model_output = tabular_model_output.shape[1]
        assert len_tabular_model_output == self.len_tabular_model_output, \
          f"Unexpected length {len_tabular_model_output}"
        # see https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462/2
        x = torch.cat((x, tabular_model_output), dim=1)
        x = self.output(x)

        # WARNING: Be careful about .squeeze(). We expect for a given batch, the output is
        # [BATCHSIZE, 1] and BATCHSIZE > 1. If the tensor has a batch dimension of size 1,
        # then squeeze(input) will also unexpectedly remove the batch dimension!
        return x.squeeze()


class Inception_Tabular_15_3(nn.Module):

    def __init__(self, tabular_model, len_tabular_model_output, initial_kernel_num=64, initialize_parameters=False):
        super(Inception_Tabular_15_3, self).__init__()

        self.conv_1 = BasicConv2d(in_channels=1, out_channels=64, kernel_size=(15, 3), stride=(2, 1))

        self.multi_2d_cnn_1a = nn.Sequential(
          Multi_2D_CNN_block(in_channels=64, num_kernel=initial_kernel_num),
          Multi_2D_CNN_block(in_channels=149, num_kernel=initial_kernel_num),
          nn.MaxPool2d(kernel_size=(3, 1))
        )

        self.multi_2d_cnn_1b = nn.Sequential(
          Multi_2D_CNN_block(in_channels=149, num_kernel=initial_kernel_num * 1.5),
          Multi_2D_CNN_block(in_channels=224, num_kernel=initial_kernel_num * 1.5),
          nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.multi_2d_cnn_1c = nn.Sequential(
          Multi_2D_CNN_block(in_channels=224, num_kernel=initial_kernel_num * 2),
          Multi_2D_CNN_block(in_channels=298, num_kernel=initial_kernel_num * 2),
          nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.multi_2d_cnn_2a = nn.Sequential(
          Multi_2D_CNN_block(in_channels=298, num_kernel=initial_kernel_num * 3),
          Multi_2D_CNN_block(in_channels=448, num_kernel=initial_kernel_num * 3),
          Multi_2D_CNN_block(in_channels=448, num_kernel=initial_kernel_num * 4),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2b = nn.Sequential(
          Multi_2D_CNN_block(in_channels=597, num_kernel=initial_kernel_num * 5),
          Multi_2D_CNN_block(in_channels=746, num_kernel=initial_kernel_num * 6),
          Multi_2D_CNN_block(in_channels=896, num_kernel=initial_kernel_num * 7),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2c = nn.Sequential(
          Multi_2D_CNN_block(in_channels=1045, num_kernel=initial_kernel_num * 8),
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 8),
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 8),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2d = nn.Sequential(
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 12),
          Multi_2D_CNN_block(in_channels=1792, num_kernel=initial_kernel_num * 14),
          Multi_2D_CNN_block(in_channels=2090, num_kernel=initial_kernel_num * 16),
        )
        self.output_before_fc = nn.Sequential(
          nn.AdaptiveAvgPool2d((1, 1)),
          nn.Flatten(),
          nn.Dropout(0.5),
          # nn.Sigmoid()
        )

        self.tabular_model = tabular_model
        self.len_tabular_model_output = len_tabular_model_output

        # the extra features are for the output for the tabular data model
        self.output = nn.Linear(2389 + len_tabular_model_output, 1)

        if initialize_parameters:
          for m in self.modules():
            model_utils.initialize_parameters(m)

      def forward(self, x_and_tabular):
        """
        Parameters
        ----------
        x_and_tabular tuple of (image input, tabular features)
        where image input and tabular features are both tensors of size BATCHSIZE s.t.
        BATCHSIZE is a batch size > 1.
        Returns
        -------
        tensor of BATCHSIZE with forward pass results
        """
        x, tabular = x_and_tabular
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

        x = self.output_before_fc(x)

        tabular_model_output = self.tabular_model(tabular)
        len_tabular_model_output = tabular_model_output.shape[1]
        assert len_tabular_model_output == self.len_tabular_model_output, \
          f"Unexpected length {len_tabular_model_output}"
        # see https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462/2
        x = torch.cat((x, tabular_model_output), dim=1)
        x = self.output(x)

        # WARNING: Be careful about .squeeze(). We expect for a given batch, the output is
        # [BATCHSIZE, 1] and BATCHSIZE > 1. If the tensor has a batch dimension of size 1,
        # then squeeze(input) will also unexpectedly remove the batch dimension!
        return x.squeeze()


class Inception_Tabular_15_1(nn.Module):

    def __init__(self, tabular_model, len_tabular_model_output, initial_kernel_num=64, initialize_parameters=False):
        super(Inception_Tabular_15_1, self).__init__()

        self.conv_1 = BasicConv2d(in_channels=1, out_channels=64, kernel_size=(15, 1), stride=(2, 1))

        self.multi_2d_cnn_1a = nn.Sequential(
          Multi_2D_CNN_block(in_channels=64, num_kernel=initial_kernel_num),
          Multi_2D_CNN_block(in_channels=149, num_kernel=initial_kernel_num),
          nn.MaxPool2d(kernel_size=(3, 1))
        )

        self.multi_2d_cnn_1b = nn.Sequential(
          Multi_2D_CNN_block(in_channels=149, num_kernel=initial_kernel_num * 1.5),
          Multi_2D_CNN_block(in_channels=224, num_kernel=initial_kernel_num * 1.5),
          nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.multi_2d_cnn_1c = nn.Sequential(
          Multi_2D_CNN_block(in_channels=224, num_kernel=initial_kernel_num * 2),
          Multi_2D_CNN_block(in_channels=298, num_kernel=initial_kernel_num * 2),
          nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.multi_2d_cnn_2a = nn.Sequential(
          Multi_2D_CNN_block(in_channels=298, num_kernel=initial_kernel_num * 3),
          Multi_2D_CNN_block(in_channels=448, num_kernel=initial_kernel_num * 3),
          Multi_2D_CNN_block(in_channels=448, num_kernel=initial_kernel_num * 4),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2b = nn.Sequential(
          Multi_2D_CNN_block(in_channels=597, num_kernel=initial_kernel_num * 5),
          Multi_2D_CNN_block(in_channels=746, num_kernel=initial_kernel_num * 6),
          Multi_2D_CNN_block(in_channels=896, num_kernel=initial_kernel_num * 7),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2c = nn.Sequential(
          Multi_2D_CNN_block(in_channels=1045, num_kernel=initial_kernel_num * 8),
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 8),
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 8),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2d = nn.Sequential(
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 12),
          Multi_2D_CNN_block(in_channels=1792, num_kernel=initial_kernel_num * 14),
          Multi_2D_CNN_block(in_channels=2090, num_kernel=initial_kernel_num * 16),
        )
        self.output_before_fc = nn.Sequential(
          nn.AdaptiveAvgPool2d((1, 1)),
          nn.Flatten(),
          nn.Dropout(0.5),
          # nn.Sigmoid()
        )

        self.tabular_model = tabular_model
        self.len_tabular_model_output = len_tabular_model_output

        # the extra features are for the output for the tabular data model
        self.output = nn.Linear(2389 + len_tabular_model_output, 1)

        if initialize_parameters:
          for m in self.modules():
            model_utils.initialize_parameters(m)

      def forward(self, x_and_tabular):
        """
        Parameters
        ----------
        x_and_tabular tuple of (image input, tabular features)
        where image input and tabular features are both tensors of size BATCHSIZE s.t.
        BATCHSIZE is a batch size > 1.
        Returns
        -------
        tensor of BATCHSIZE with forward pass results
        """
        x, tabular = x_and_tabular
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

        x = self.output_before_fc(x)

        tabular_model_output = self.tabular_model(tabular)
        len_tabular_model_output = tabular_model_output.shape[1]
        assert len_tabular_model_output == self.len_tabular_model_output, \
          f"Unexpected length {len_tabular_model_output}"
        # see https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462/2
        x = torch.cat((x, tabular_model_output), dim=1)
        x = self.output(x)

        # WARNING: Be careful about .squeeze(). We expect for a given batch, the output is
        # [BATCHSIZE, 1] and BATCHSIZE > 1. If the tensor has a batch dimension of size 1,
        # then squeeze(input) will also unexpectedly remove the batch dimension!
        return x.squeeze()


class Inception_Tabular_5_Linears(nn.Module):

    def __init__(self, tabular_model, len_tabular_model_output, initial_kernel_num=64, initialize_parameters=False):
        
        self.conv_1 = BasicConv2d(in_channels=1, out_channels=64, kernel_size=(7, 3), stride=(2, 1))

        self.multi_2d_cnn_1a = nn.Sequential(
          Multi_2D_CNN_block(in_channels=64, num_kernel=initial_kernel_num),
          Multi_2D_CNN_block(in_channels=149, num_kernel=initial_kernel_num),
          nn.MaxPool2d(kernel_size=(3, 1))
        )

        self.multi_2d_cnn_1b = nn.Sequential(
          Multi_2D_CNN_block(in_channels=149, num_kernel=initial_kernel_num * 1.5),
          Multi_2D_CNN_block(in_channels=224, num_kernel=initial_kernel_num * 1.5),
          nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.multi_2d_cnn_1c = nn.Sequential(
          Multi_2D_CNN_block(in_channels=224, num_kernel=initial_kernel_num * 2),
          Multi_2D_CNN_block(in_channels=298, num_kernel=initial_kernel_num * 2),
          nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.multi_2d_cnn_2a = nn.Sequential(
          Multi_2D_CNN_block(in_channels=298, num_kernel=initial_kernel_num * 3),
          Multi_2D_CNN_block(in_channels=448, num_kernel=initial_kernel_num * 3),
          Multi_2D_CNN_block(in_channels=448, num_kernel=initial_kernel_num * 4),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2b = nn.Sequential(
          Multi_2D_CNN_block(in_channels=597, num_kernel=initial_kernel_num * 5),
          Multi_2D_CNN_block(in_channels=746, num_kernel=initial_kernel_num * 6),
          Multi_2D_CNN_block(in_channels=896, num_kernel=initial_kernel_num * 7),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2c = nn.Sequential(
          Multi_2D_CNN_block(in_channels=1045, num_kernel=initial_kernel_num * 8),
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 8),
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 8),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2d = nn.Sequential(
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 12),
          Multi_2D_CNN_block(in_channels=1792, num_kernel=initial_kernel_num * 14),
          Multi_2D_CNN_block(in_channels=2090, num_kernel=initial_kernel_num * 16),
        )
        self.output_before_fc = nn.Sequential(
          nn.AdaptiveAvgPool2d((1, 1)),
          nn.Flatten(),
          nn.Dropout(0.5),
          # nn.Sigmoid()
        )

        self.tabular_model = tabular_model
        self.len_tabular_model_output = len_tabular_model_output

        # the extra features are for the output for the tabular data model
        self.output = nn.Sequential(
            nn.Linear(2389 + len_tabular_model_output, 1200),
            nn.Linear(1200, 400),
            nn.Linear(400, 100),
            nn.Linear(100, 20),
            nn.Linear(20, 1)
            ),

        if initialize_parameters:
          for m in self.modules():
            model_utils.initialize_parameters(m)

      def forward(self, x_and_tabular):
        """
        Parameters
        ----------
        x_and_tabular tuple of (image input, tabular features)
        where image input and tabular features are both tensors of size BATCHSIZE s.t.
        BATCHSIZE is a batch size > 1.
        Returns
        -------
        tensor of BATCHSIZE with forward pass results
        """
        x, tabular = x_and_tabular
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

        x = self.output_before_fc(x)

        tabular_model_output = self.tabular_model(tabular)
        len_tabular_model_output = tabular_model_output.shape[1]
        assert len_tabular_model_output == self.len_tabular_model_output, \
          f"Unexpected length {len_tabular_model_output}"
        # see https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462/2
        x = torch.cat((x, tabular_model_output), dim=1)
        x = self.output(x)

        # WARNING: Be careful about .squeeze(). We expect for a given batch, the output is
        # [BATCHSIZE, 1] and BATCHSIZE > 1. If the tensor has a batch dimension of size 1,
        # then squeeze(input) will also unexpectedly remove the batch dimension!
        return x.squeeze()

class Inception_Tabular_5_Linears_BatchNorm(nn.Module):

    def __init__(self, tabular_model, len_tabular_model_output, initial_kernel_num=64, initialize_parameters=False):
        
        self.conv_1 = BasicConv2d(in_channels=1, out_channels=64, kernel_size=(7, 3), stride=(2, 1))

        self.multi_2d_cnn_1a = nn.Sequential(
          Multi_2D_CNN_block(in_channels=64, num_kernel=initial_kernel_num),
          Multi_2D_CNN_block(in_channels=149, num_kernel=initial_kernel_num),
          nn.MaxPool2d(kernel_size=(3, 1))
        )

        self.multi_2d_cnn_1b = nn.Sequential(
          Multi_2D_CNN_block(in_channels=149, num_kernel=initial_kernel_num * 1.5),
          Multi_2D_CNN_block(in_channels=224, num_kernel=initial_kernel_num * 1.5),
          nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.multi_2d_cnn_1c = nn.Sequential(
          Multi_2D_CNN_block(in_channels=224, num_kernel=initial_kernel_num * 2),
          Multi_2D_CNN_block(in_channels=298, num_kernel=initial_kernel_num * 2),
          nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.multi_2d_cnn_2a = nn.Sequential(
          Multi_2D_CNN_block(in_channels=298, num_kernel=initial_kernel_num * 3),
          Multi_2D_CNN_block(in_channels=448, num_kernel=initial_kernel_num * 3),
          Multi_2D_CNN_block(in_channels=448, num_kernel=initial_kernel_num * 4),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2b = nn.Sequential(
          Multi_2D_CNN_block(in_channels=597, num_kernel=initial_kernel_num * 5),
          Multi_2D_CNN_block(in_channels=746, num_kernel=initial_kernel_num * 6),
          Multi_2D_CNN_block(in_channels=896, num_kernel=initial_kernel_num * 7),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2c = nn.Sequential(
          Multi_2D_CNN_block(in_channels=1045, num_kernel=initial_kernel_num * 8),
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 8),
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 8),
          nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.multi_2d_cnn_2d = nn.Sequential(
          Multi_2D_CNN_block(in_channels=1194, num_kernel=initial_kernel_num * 12),
          Multi_2D_CNN_block(in_channels=1792, num_kernel=initial_kernel_num * 14),
          Multi_2D_CNN_block(in_channels=2090, num_kernel=initial_kernel_num * 16),
        )
        self.output_before_fc = nn.Sequential(
          nn.AdaptiveAvgPool2d((1, 1)),
          nn.Flatten(),
          nn.Dropout(0.5),
          # nn.Sigmoid()
        )

        self.tabular_model = tabular_model
        self.len_tabular_model_output = len_tabular_model_output

        # the extra features are for the output for the tabular data model
        self.output = nn.Sequential(
            nn.Linear(2389 + len_tabular_model_output, 1200),
            nn.BatchNorm1d(1200),
            nn.Linear(1200, 400),
            nn.BatchNorm1d(400),
            nn.Linear(400, 100),
            nn.BatchNorm1d(100),
            nn.Linear(100, 20),
            nn.BatchNorm1d(20),
            nn.Linear(20, 1)
            ),

        if initialize_parameters:
          for m in self.modules():
            model_utils.initialize_parameters(m)

      def forward(self, x_and_tabular):
        """
        Parameters
        ----------
        x_and_tabular tuple of (image input, tabular features)
        where image input and tabular features are both tensors of size BATCHSIZE s.t.
        BATCHSIZE is a batch size > 1.
        Returns
        -------
        tensor of BATCHSIZE with forward pass results
        """
        x, tabular = x_and_tabular
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

        x = self.output_before_fc(x)

        tabular_model_output = self.tabular_model(tabular)
        len_tabular_model_output = tabular_model_output.shape[1]
        assert len_tabular_model_output == self.len_tabular_model_output, \
          f"Unexpected length {len_tabular_model_output}"
        # see https://discuss.pytorch.org/t/concatenate-layer-output-with-additional-input-data/20462/2
        x = torch.cat((x, tabular_model_output), dim=1)
        x = self.output(x)

        # WARNING: Be careful about .squeeze(). We expect for a given batch, the output is
        # [BATCHSIZE, 1] and BATCHSIZE > 1. If the tensor has a batch dimension of size 1,
        # then squeeze(input) will also unexpectedly remove the batch dimension!
        return x.squeeze()
