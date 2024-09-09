#TODO: do i need a is_top? 

import torch
import torch.nn as nn
from typing import Sequence, Union, Tuple
from monai.networks.blocks import squeeze_and_excitation as se
from squeeze_and_excitation import squeeze_and_excitation as se1
from monai.networks.layers.factories import Act, Norm

from monai.networks.layers.simplelayers import SkipConnection
from convolutionMonai import DenseBlock
from monai.utils import alias, deprecated_arg, export
import torch.nn.functional as F
import numpy as np

LAYER = 0
@alias("Quicknat")
class QuickNAT(nn.Module):
    """
    NETWORK: 
        num_class = 33
        num_channels = 1
        num_filters = 64
        kernel_h = 5
        kernel_w = 5
        kernel_c = 1
        stride_conv = 1
        pool = 2
        stride_pool = 2
        #Valid options : NONE, CSE, SSE, CSSE
        se_block = "None"
        drop_out = 0.5
        act = PReLu
    """
    def __init__(
        self,
        num_classes: int = 33,
        num_channels: int = 1,
        num_filters: int = 64,
        kernel_h: int = 5,
        kernel_w: int = 5,
        kernel_c: int = 1,
        stride_conv: int = 1,
        pool:int = 2,
        stride_pool: int = 2,
        #Valid options : NONE, CSE, SSE, CSSE
        se_block: str = "None",
        drop_out: float = 0.5, 
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        adn_ordering: str = "NA", 
    ) -> None:

        params = {
            "num_channels" : num_channels,
            "num_filters" : num_filters,
            "kernel_h" : kernel_h,
            "kernel_w" : kernel_w,
            "kernel_c" : kernel_c,
            "stride_conv" : stride_conv,
            "pool" : pool,
            "stride_pool": stride_pool,
            "num_classes" : num_classes,
            "se_block" : se_block,
            "drop_out" : drop_out,
            "act" : act,
            "norm" : norm,
            "adn_ordering" : adn_ordering,   
        }

        super(QuickNAT, self).__init__()
        self.conv = nn.Sequential()
        self.layers = 4
        # params for encoder / decoder path 
        # 4 layers -> never less right? 
        # channels for encoder from top layer to bottom layer: 
        channels = [params["num_channels"], params["num_filters"], params["num_filters"], params["num_filters"], params["num_filters"]]

        # sequence of convolutional strides not needed as they are always params["stride_conv"]
        def _create_block(
            inc: int, outc: int, channels: Sequence[int], params: dict, is_top: bool, layer: int
        ) -> nn.Module:
            
            c = channels[0]

            subblock: nn.Module
              
            if len(channels) > 2:
                # recursion for layer down 
                # not sure about input/output channels 
                subblock = _create_block(outc, outc, channels[1:], params, False, layer+1)
            
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                # need for bottolneck layer after bottom layer -> check this  
                # in_channels = params[num_filters]
                # out_channels = params [num_filters]
                # Do not need an adapted forward method -> for bottleneck layer can just call the DenseBlock
                subblock = self._get_BottleneckBlock(c, channels[1], params, layer+1)
            down = self._getEncoderBlock(inc, params["num_filters"], params, is_top, layer)  # create layer in downsampling path
            self.conv.add_module(f"Encoder{layer:d}", down)
            up = self._getDecoderBlock(params["num_filters"]*2, params["num_filters"], params, is_top, layer)  # create layer in upsampling path
            self.conv.add_module(f"Decoder{layer:d}", up)
            # The classifier block is added after the last decoder layer -> need to first make encoder & decoder path and afterwards add the classifier layer
            if is_top: 
                connection = self._get_connection_block(down, up, subblock, layer)
                return self._getClassifier(params, connection)
            else: 
                
                return self._get_connection_block(down, up, subblock, layer)

        self.model = _create_block(params["num_channels"], params["num_filters"], channels, params, True,0)
    
    
    
    def _getClassifier(self, params, mod: nn.Module): 
        # in_channel: params["num_filters"] 
        # out_channel 2   
        #monai implementation? Can not use Monai Convolution implementation because I do not need PReLU and normalization 
        clas = nn.Conv2d(
            params["num_filters"],
            params["num_classes"],
            params["kernel_c"],
            params["stride_conv"],
        )
        def _new_forward(input : torch.Tensor, weights=None): 
            batch_size, channel, a, b = input.size()
            # copied from original classifier block 
            if weights is not None:
                weights, _ = torch.max(weights, dim=0)
                weights = weights.view(1, channel, 1, 1)
                out_conv = F.conv2d(input, weights)
            else:
                # is this correct?
                out_conv = nn.Conv2d.forward(clas, input)
            return out_conv

        clas.forward = _new_forward 
        self.conv.add_module(f"Classifier", clas)
        return nn.Sequential(mod, clas) 
        

    
    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module, layer: int) -> nn.Module:
        """
        Returns the block object defining a layer of the QuickNAT structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        
        mod = nn.Sequential(down_path, SkipConnection(subblock), up_path)
        #mod.forward = _forward_with_unpool
        """self.conv.add_module(f"Encoder{layer:d}", down_path)
        self.conv.add_module(f"Skip{layer:d}", SkipConnection(subblock))
        self.conv.add_module(f"Decoder{layer:d}", up_path)"""
        self.conv.add_module(f"Skip{layer:d}", SkipConnection(subblock))

        return mod
    

    def getSELayer(self, n_filters, se_block_type = "None"): 
            if se_block_type == 'CSE':
                return se.ChannelSELayer(n_filters)
            # not implemented in squeeze_and_excitation in monai 
            elif se_block_type == 'SSE':
                return se1.SpatialSELayer(n_filters)

            elif se_block_type == 'CSSE':
                # not implemented in monai
                return se1.ChannelSpatialSELayer(n_filters)
            else:
                return None

    def _get_DenseBlock(self, inc, outc, params, layer) -> nn.Module: 
    
        mod = DenseBlock(
            2,
            inc,
            pool = params["pool"],
            stride_pool = params["stride_pool"],
            strides=params["stride_conv"],
            kernel_size=(params["kernel_h"], params["kernel_w"]),
            subunits=2,
            act=params["act"],
            norm=params["norm"],
            dropout=params["drop_out"],
            adn_ordering=params["adn_ordering"],
            layer = layer,
            
        )
        return mod

    def _get_BottleneckBlock(self, inc: int, outc: int, params: dict, layer: int): 
        mod : DenseBlock
        mod = self._get_DenseBlock(inc, outc, params, layer)

        def _forward_bot(input : torch.Tensor, weights = None): 
            
            input, indices = mod.maxPool(input)
            (self.conv.get_submodule("Encoder" + str(layer - 1))).indices = indices
            out_block = DenseBlock.forward(mod, input)
            #indices = (self.conv.get_submodule("Encoder" + str(layer - 1))).indices
            out_block = mod.unPool(input, indices)
            return out_block

        mod.forward = _forward_bot

        return mod 

    def _getEncoderBlock(self, inc : int, outc : int, params : dict, is_top : bool, layer : int) -> nn.Module:
        
        mod : DenseBlock
        mod = self._get_DenseBlock(inc, outc, params, layer)
        drop_out = None 
        if params["drop_out"] > 0:
            # can't find monai implementation 
            drop_out = nn.Dropout2d(params["drop_out"])
    
        def _forward_enc(input : torch.Tensor, weights=None): 
            # first encoder does not have a maxpool 
            if (layer != 0) : 
                input, indices = mod.maxPool(input)
                (self.conv.get_submodule("Encoder" + str(layer - 1))).indices = indices

            out_block = DenseBlock.forward(mod, input)
            seLayer = self.getSELayer(params["num_filters"], params["se_block"])
            
            if seLayer != None:
                out_block = seLayer(out_block, weights)

            if drop_out != None:
                out_block = drop_out(out_block)
            
            return out_block

        # override the foward function
        mod.forward = _forward_enc
        return mod

    def _getDecoderBlock(self, inc : int, outc : int, params, is_top, layer) -> nn.Module: 
        mod : DenseBlock
        mod = self._get_DenseBlock(inc, outc, params, layer)
        
        drop_out = None 
        if params["drop_out"] > 0:
            # can't find monai implementation 
            drop_out = nn.Dropout2d(params["drop_out"])
         
        # input should already be the torch.cat((out_block, unpool), dim=1)
        def _forward_dec(input : torch.Tensor, weights = None): 
            
            # out_block = mod.forward(unpool_Block)
            out_block = DenseBlock.forward(mod, input)
            seLayer = self.getSELayer(params["num_filters"], params["se_block"]) 
            if seLayer != None:
                out_block = seLayer(out_block, weights)

            if drop_out != None:
                out_block = drop_out(out_block)

            # unpool 
            if layer != 0: 
                indices = (self.conv.get_submodule("Encoder" + str(layer - 1))).indices
                out_block = mod.unPool(out_block, indices)
            
            return out_block

        mod.forward = _forward_dec

        return mod

    @property      
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda


    def forward(self, input: torch.Tensor) -> torch.Tensor: 
        
        input = self.model(input)
        return input

    def predict(self, X, device=0, enable_dropout=False):
        """
        Predicts the output after the model is trained.
        Inputs:
        - X: Volume to be predicted
        """
        self.eval()
        print("tensor size before transformation", X.shape)

        if type(X) is np.ndarray:
            # X = torch.tensor(X, requires_grad=False).type(torch.FloatTensor)
            X = (
                torch.tensor(X, requires_grad=False)
                .type(torch.FloatTensor)
                #.cuda(device, non_blocking=True)
            )
        elif type(X) is torch.Tensor and not X.is_cuda:
            X = X.type(torch.FloatTensor).cuda(device, non_blocking=True)


        if enable_dropout:
            self.enable_test_dropout()

        with torch.no_grad():
            out = self.forward(X)

        max_val, idx = torch.max(out, 1)
        idx = idx.data.cpu().numpy()
        prediction = np.squeeze(idx)
        print("prediction shape", prediction.shape)
        del X, out, idx, max_val
        
        return prediction

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with '*.model'.

        Inputs:
        - path: path string
        """
        print("Saving model... %s" % path)
        torch.save(self.state_dict(), path)

        