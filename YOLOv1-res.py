import torch
import torch.nn as nn
import torch.nn.functional as F



class Unsupported_Activation(Exception):
    def __init__(self,activation):
        message=f"Activation {activation} not one of relu,gelu,selu, or silu"
        super().__init__(message)

class Identity_block(nn.Module):
    def __init__(self, channels_in,kernel_size, pool=False,activation='gelu'):
        super().__init__()

        # Batch Norm Layers
        self.bn1=nn.BatchNorm2d(channels_in)
        self.bn2=nn.BatchNorm2d(channels_in)

        #Convolutional Layers
        self.conv1=nn.Conv2d(channels_in,channels_in,kernel_size,padding='same')
        self.conv2=nn.Conv2d(channels_in,channels_in,kernel_size,padding='same')

        self.pool=pool

        #Choose Activation
        if (activation=='relu'):
            self.activ=nn.ReLU()
        elif (activation=='gelu'):
            self.activ=nn.GELU()
        elif (activation=='selu'):
            self.activ=nn.SELU()
        elif (activation=='silu'):
            self.activ=nn.SiLU()
        else: 
            raise Unsupported_Activation(activation)
        
    def forward(self, input):
        x=self.conv1(input)
        x=self.bn1(x)
        x=self.activ(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.activ(x)

        #Skip connection
        x+=input
        if (self.pool==True):
            return F.max_pool2d(x,2,2)
        else:
            return x
        
class Bottleneck_block(nn.Module):
    def __init__(self, channels_in,channels_out,kernel_size, pool=False,activation='gelu'):
        super().__init__()

        # Batch Norm Layers
        self.bn1=nn.BatchNorm2d(channels_in)
        self.bn2=nn.BatchNorm2d(channels_in)
        self.bn3=nn.BatchNorm2d(channels_out)

        #Convolutional Layers
        self.skip=nn.Conv2d(channels_in,channels_out,1,padding='same')
        self.conv1=nn.Conv2d(channels_in,channels_in,1,padding='same')
        self.conv2=nn.Conv2d(channels_in,channels_in,kernel_size,padding='same')
        self.conv3=nn.Conv2d(channels_in,channels_out,1,padding='same')
        self.pool=pool

        #Choose Activation
        if (activation=='relu'):
            self.activ=nn.ReLU()
        elif (activation=='gelu'):
            self.activ=nn.GELU()
        elif (activation=='selu'):
            self.activ=nn.SELU()
        elif (activation=='silu'):
            self.activ=nn.SiLU()
        else: 
            raise Unsupported_Activation(activation)
        
    def forward(self, input):
        x=self.conv1(input)
        x=self.bn1(x)
        x=self.activ(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.activ(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x=self.activ(x)

        #Skip connection
        x+=self.skip(input)
        if (self.pool==True):
            return F.max_pool2d(x,2,2)
        else:
            return x
 # Sub_block containing a few identity blocks with a bottleneck block at the end
class ResNet_Sub_Block(nn.Module):
    def __init__(self,conv_in,conv_out,no_identity,kernel_size,activation='gelu'):
        super().__init__()
        self.id_blocks=nn.ModuleList([Identity_block(conv_in,kernel_size,False,activation) for i in range(no_identity)])
        self.bottleneck=Bottleneck_block(conv_in,conv_out,kernel_size,True,activation)
    def forward(self,input):
        x=input
        for id_block in self.id_blocks:
            x=id_block(x)
        return self.bottleneck(x)
    
 # Complete ResNet  
class ResNet(nn.Module):
    def __init__(self,out_channels:int,channel_sizes:list,block_sizes:list,kernel_sizes: list):
        # Ensure number of residual_sub_blocks equals number of kernel_sizes
        assert len(block_sizes) == len(kernel_sizes), "Number of blocks not equal to number of kernels given"
        super().__init__()
        self.out_channels=out_channels
        self.in_channels=channel_sizes[0]
        self.res_sub_blocks=nn.ModuleList()
        for i,length in enumerate(block_sizes[:-1]):
            self.res_sub_blocks.add_module(str(i),ResNet_Sub_Block(channel_sizes[i],channel_sizes[i+1],length,kernel_sizes[i]))
        l=len(self.res_sub_blocks)
        self.res_sub_blocks.add_module(str(l),ResNet_Sub_Block(channel_sizes[-1],out_channels,block_sizes[-1],kernel_sizes[-1]))
    def forward(self,input):
        x=input
        for sub_block in self.res_sub_blocks:
            x=sub_block(x)
        return x
    
    # Simple Multi-Layer Perceptron
class MLP(nn.Module):
    def __init__(self,in_no,out_no,activation='gelu', layer_sizes:list =[]):
        super().__init__()
        self.layers=nn.ModuleList()
        l=len(layer_sizes)
        if l==0:
            self.layers.add_module(str(0),nn.Linear(in_no,out_no))
        elif l==1:
            self.layers.add_module(str(0),nn.Linear(in_no,layer_sizes[0]))
            self.layers.add_module(str(1),nn.Linear(layer_sizes[0],out_no))
        elif l==2:
            self.layers.add_module(str(0),nn.Linear(in_no,layer_sizes[0]))
            self.layers.add_module(str(1),nn.Linear(layer_sizes[0],layer_sizes[1]))
            self.layers.add_module(str(2),nn.Linear(layer_sizes[1],out_no))
        elif l>2:
            self.layers.add_module(str(0),nn.Linear(in_no,layer_sizes[0]))
            for i in range(l-2):
                self.layers.add_module(str(i+1),nn.Linear(layer_sizes[i],layer_sizes[i+1]))
            self.layers.add_module(str(l),nn.Linear(layer_sizes[-1],out_no))
        if (activation=='relu'):
            self.activ=nn.ReLU()
        elif (activation=='gelu'):
            self.activ=nn.GELU()
        elif (activation=='selu'):
            self.activ=nn.SELU()
        elif (activation=='silu'):
            self.activ=nn.SiLU()
        else: 
            raise Unsupported_Activation(activation)
    def forward(self, input):
        x=input
        for layer in self.layers[:-1]:
            x=layer(x)
            x=self.activ(x)
        return self.layers[-1](x)
    
class YOLOmodel(nn.Module):
    def __init__(self,feature_backbone,num_divisions,num_bounding_boxes,num_classes):
        super().__init__()
        #Pass ResNet backbone to the model for feature extraction
        self.backbone= feature_backbone
        conv_out=self.backbone.out_channels
        conv_in=self.backbone.in_channels
        # Final size is shape of output
        final_size=(num_divisions*num_divisions)*(num_bounding_boxes*5+num_classes)
        self.input_conv=nn.Conv2d(3,conv_in,(7,7),2)
        self.fully_connected=MLP(conv_out,final_size)
    def forward(self,input):
        x=self.input_conv(input)
        x=self.backbone(x)
        x=self.fully_connected(x)
        return x
