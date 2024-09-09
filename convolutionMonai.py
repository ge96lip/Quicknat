# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from multiprocessing import pool
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers.factories import Conv


class Convolution(nn.Sequential):
    """
    Constructs a convolution with normalization, optional dropout, and optional activation layers::

        -- (Norm -- Acti) -- (Conv|ConvTrans) -- 

    if ``conv_only`` set to ``True``::

        -- (Conv|ConvTrans) --

    For example:

    .. code-block:: python

        from monai.networks.blocks import Convolution

        conv = Convolution(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            adn_ordering="ADN",
            act=("prelu", {"init": 0.2}),
            dropout=0.1,
            norm=("layer", {"normalized_shape": (10, 10, 10)}),
        )
        print(conv)

    output::

        Convolution(
        (adn): ADN(
            (A): PReLU(num_parameters=1)
            (D): Dropout(p=0.1, inplace=False)
            (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
          )
        (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          
        )

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the spatial dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zeroes out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zeroes out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no larger than the value of `spatial_dims`.
        dilation: dilation rate. Defaults to 1.
        groups: controls the connections between inputs and outputs. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        conv_only: whether to use the convolutional layer only. Defaults to False.
        is_transposed: if True uses ConvTrans instead of Conv. Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.
        output_padding: controls the additional size added to one side of the output shape.
            Defaults to None.

    See also:

        :py:class:`monai.networks.layers.Conv`
        :py:class:`monai.networks.blocks.ADN`

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        # Dropout only done for complete Denseblock but not for single convolution blocks
        adn_ordering: str = "NA",
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        # defines a spacing between the values in a kernel. 
        # A 3x3 kernel with a dilation rate of 2 will have the same field of view as a 5x5 kernel, while only using 9 parameters.
        dilation: Union[Sequence[int], int] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        output_padding: Optional[Union[Sequence[int], int]] = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed

        if padding is None:
            padding = same_padding(kernel_size, dilation)
        conv_type = Conv[Conv.CONVTRANS if is_transposed else Conv.CONV, self.spatial_dims]

        conv: nn.Module
        if is_transposed:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, strides)
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias,
                dilation=dilation,
            )
        else:
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )     

        if conv_only or act is None and norm is None and dropout is None:
            self.add_module("conv", conv)
            return 

        self.add_module(
            "na",
            ADN(
                ordering=adn_ordering,
                in_channels=in_channels,
                act=act,
                norm=norm,
                norm_dim=2,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )
        self.add_module("conv", conv)
        


class DenseBlock(nn.Module):
    """
    Dense module with 3 convolutions and a residual connection between each convolution.
    Every convolutional layer is preceded by a batchnormalization layer and a Rectifier Linear Unit (ReLU) layer

    For example:

    .. code-block:: python

        from monai.networks.blocks import ResidualUnit

        convs = ResidualUnit(
            spatial_dims=2,
            in_channels=1,
            subunits=3,
            out_channels=32,
            adn_ordering="NA",
        )
        print(convs)

    output::

    ResidualUnit(
        (conv): Sequential(
          (unit0): Convolution(
            (adn): ADN(
              (N): InstanceNorm2d(1, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (A): PReLU(num_parameters=1)
            )
            (conv): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
          )
          (unit1): Convolution(
            (adn): ADN(
              (N): InstanceNorm2d(33, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (A): PReLU(num_parameters=1)
            )
            (conv): Conv2d(33, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
          )
          (unit2): Convolution(
            (adn): ADN(
              (N): InstanceNorm2d(65, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
              (A): PReLU(num_parameters=1)
            )
            (conv): Conv2d(65, 32, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (residual): Identity()
      )

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels. Input channel size is variable, depending on the number of dense connections
        out_channels: number of output channels. Output at each convolution layer is default set to 64, which acts as a bottleneck for feature map selectivity
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 5. 
        subunits: number of convolutions. Defaults to 2.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zero out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zero out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no larger than the value of `dimensions`.
        dilation: dilation rate. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        last_conv_only: for the last subunit, whether to use the convolutional layer only.
            Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.

    See also:

        :py:class:`monai.networks.blocks.Convolution`

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        pool: int, 
        stride_pool: int, 
        out_channels: int = 64,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 5,
        subunits: int = 2,
        adn_ordering: str = "NA",
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        layer : int = -1,
        indices : int = 0
    ) -> None:
        super().__init__()
        self.pool = pool
        self.stride_pool = stride_pool
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        self.residual = nn.Identity()
        if not padding:
            padding = same_padding(kernel_size, dilation)
        subunits = max(1, subunits)
        self.layer = layer

        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = Convolution(
                self.spatial_dims,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=padding,
            )
            
            self.conv.add_module(f"unit{su:d}", unit)

            # after first loop set channels and strides to what they should be for subsequent units
            # because residual identity is applied to every conv block, schannels need to grow with every unit
            in_channels = in_channels + out_channels 
            # I think strides should stay the same: sstrides = 1
        # third conv layer also preceded by batch normalization and ReLU but with a (1,1) kerne size to compress feature map size to 64
        conv_lastUnit = Convolution(
            self.spatial_dims, 
            in_channels, 
            out_channels, 
            strides=strides, 
            kernel_size = (1,1), 
            adn_ordering=adn_ordering,
            act=act,
            norm=norm,
            dropout=dropout,
            dropout_dim=dropout_dim,
            dilation=dilation,
            bias=bias,
            conv_only=conv_only,
            padding=0,
        )
        #last conv layer
        self.conv.add_module(f"unit2", conv_lastUnit)

        # maxpool and unpool: 
        self.maxPool = nn.MaxPool2d(
            kernel_size=pool,
            stride=stride_pool,
            return_indices=True,
            ceil_mode=True,
        )
        self.unPool = nn.MaxUnpool2d(
            kernel_size=pool,
            stride=stride_pool
        )
        self.indices = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        res: torch.Tensor =  self.residual(x) # create the additive residual from x
        x1 = x
        for module in self.conv:
            x = module(x)
            if module == self.conv.get_submodule("unit0"): 
                # TODO:
                #x = x1 = x + res # add the residual to the output does not work anymore 
                x1 = x 
                x = torch.cat((x, res), dim = 1)

            elif module == self.conv.get_submodule("unit1"): 
                x = torch.cat((x1, x, res), dim = 1) # add the residual to the output
        return x
