import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor

from voxelnet.utils.registry import MODELS
from voxelnet.modules.pool import Pool
from voxelnet.modules.functional import fapply 


class BatchNorm(nn.BatchNorm1d):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)

class Dropout(nn.Dropout):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,
                   kernel_size=1,
                   stride=1,
                   dilation=1,
                   transposed=False,
                   norm_layer=BatchNorm,
                   conv_layer = spnn.Conv3d,
                   activate_layer=spnn.ReLU) -> None:
        super().__init__()
        self.conv = conv_layer(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,dilation=dilation,transposed=transposed)
        if norm_layer is not None:
            # if get_cfg().rank>=0:
            #     self.norm = SyncNorm(out_channels)
            # else:
            #     self.norm = norm_layer(out_channels)
            self.norm = norm_layer(out_channels)
        else:
            self.norm = nn.Identity()
        if activate_layer is not None:
            self.act = activate_layer(True)
        else:
            self.act = nn.Identity()
    
    def forward(self,x):
        return self.act(self.norm(self.conv(x))) 


class SwitchModule(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv = spnn.Conv3d(channels,channels*3,kernel_size=1)

    def forward(self,input,x1,x2,x3):
        scale = self.conv(input).feats
        scale = scale.reshape(scale.shape[0],3,-1)
        f2 = torch.stack([x1.feats,x2.feats,x3.feats],dim=1)
        f2 = (f2*scale).sum(dim=1)
        output = SparseTensor(coords=input.coords, feats=f2, stride=input.stride)
        output.cmaps = input.cmaps
        output.kmaps = input.kmaps
        return output

class LRPBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels!=out_channels:
            self.lin = spnn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            
        else:
            self.lin = nn.Identity()
        self.conv0 = nn.Sequential(
            ConvBlock(in_channels,out_channels,kernel_size=kernel_size),
            ConvBlock(out_channels,out_channels,kernel_size=kernel_size)
        )
        
        self.pool1 = Pool(kernel_size=kernel_size)
        self.pool2 = Pool(kernel_size=kernel_size,dilation=3)
        self.pool3 = Pool(kernel_size=kernel_size,dilation=9)

        self.scale0 = Scale(out_channels,1.)
        self.scale1 = Scale(out_channels,1e-2)
        p = 0.15

        self.dropout = Dropout(p)
        self.switch = SwitchModule(out_channels)

    def forward(self,x):
        x0 = self.conv0(x)
        x0 = x0+self.lin(x)

        x1 = self.pool1(x0)
        x2 = self.pool2(x1)
        x3 = self.pool3(x2)
        y = self.scale0(x0) + self.dropout(self.scale1(self.switch(x0,x1,x2,x3)))
        return y 
 


class Scale(nn.Module):
    def __init__(self,channels,layer_scale_init_value=1e-2) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels)*layer_scale_init_value)

    def forward(self,input):
        feats = input.feats 
        feats = self.scale.unsqueeze(0)*feats
        output = SparseTensor(coords=input.coords, feats=feats, stride=input.stride)
        output.cmaps = input.cmaps
        output.kmaps = input.kmaps
        return output

@MODELS.register_module()
class LRPNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=20,
                 encoder_channels=[32,64,96,128,128,128,128],
                 decoder_channels=[128,128,128,128,128,128,128]):
        
        super().__init__()
        
        
        assert len(encoder_channels) == len(decoder_channels) and encoder_channels[-1] == decoder_channels[0]
        self.depth = len(encoder_channels)-1
        
        block = LRPBlock
       

        self.in_channels = in_channels
   
        # Unet-like structure
        #-------------------------------- Input ----------------------------------------
        self.input_conv = nn.Sequential(
            ConvBlock(in_channels,encoder_channels[0],3,1),
            block(encoder_channels[0],encoder_channels[0],3)
        )
        #-------------------------------- Encoder ----------------------------------------
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.depth):
            self.downsamples.append(ConvBlock(encoder_channels[i],encoder_channels[i+1],2,2))
            self.encoders.append(block(encoder_channels[i+1],encoder_channels[i+1],3))
        
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(self.depth):
            self.upsamples.append(ConvBlock(decoder_channels[i],decoder_channels[i+1],2,2,transposed=True))
            self.decoders.append(block(decoder_channels[i+1]+encoder_channels[self.depth-i-1],decoder_channels[i+1],3))

        # Linear head
        self.pred = spnn.Conv3d(decoder_channels[-1], out_channels, kernel_size=1, stride=1, bias=True)



    def forward(self, features, indices,vertices):
        if self.in_channels==6:
            features = torch.cat([features,vertices],dim=1)
        # input
        indices = indices.int()
        x = SparseTensor(features, indices)

        en0 = self.input_conv(x)

        en_tensors = [en0]
        for i in range(self.depth):
            x = self.downsamples[i](en_tensors[-1])
            x = self.encoders[i](x)
            if i<self.depth-1:
                en_tensors.append(x)
        
        for i in range(self.depth):
            x = self.upsamples[i](x)
            x = torchsparse.cat([x,en_tensors[self.depth-i-1]])
            x = self.decoders[i](x)
            
        #-------------------------------- output ----------------------------------------
        output = self.pred(x)

        return output.F