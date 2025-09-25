from monai.networks.nets import AutoEncoder
import warnings
from typing import Optional, Sequence, Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch import sigmoid
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export

import monai.networks.blocks.dynunet_block
from monai.networks.blocks.dynunet_block import get_padding, get_output_padding

# Redefinition of UnetBasicBlock class from Monai to add bias to convolutions.
def modified_get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Optional[Union[Tuple, str]] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = True, # Adding bias in every convolutional layer ...
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


class InterAutoEncoder(AutoEncoder):
        
    def forward(self, x):
        x_inter = self.encode(x)
        x_out = self.intermediate(x_inter)
        x_out = self.decode(x_out)
        return x_out, x_inter
    
    def forward_inter(self, x):
        x_out = self.decode(x)
        return x_out
    
    
class InterAutoEncoderSourceClassifRegularizer(AutoEncoder):
        
    def forward(self, x):
        x_inter = self.encode(x)
        x_out = self.intermediate(x_inter)
        x_out = self.decode(x_out)
        return x_out, x_inter


# FedSpotunet forward functions
def forward_connection_with_policy(x, global_block, local_block, action_mask, i):
    
    # Forward the first block of the given sequence
    out = global_block[0](x)*action_mask[..., i] + local_block[0](x)*(1 - action_mask[..., i])
    i += 1
    
    # If the given block is not the bottom one, recursively call forward with policy and forward with the 
    if len(global_block) > 1 and type(global_block[1]) is SkipConnection:    
        # Forward the middle block with policy
        out_middle, i = forward_connection_with_policy(out, global_block[1].submodule, local_block[1].submodule, action_mask, i)
        
        # We can't juste use the forward of the SkipConnection class, have to concatenate the results ourselves
        out = torch.cat([out, out_middle], dim=1)
        
        # Forward the last module of the block with policy
        out = global_block[2](out)*action_mask[..., i] + local_block[2](out)*(1 - action_mask[..., i])
        i += 1
    
    return out, i


def forward_with_policy(x, global_net, local_net, policy):
    
    action = policy.contiguous()  # [batch_size, nb_blocks]
    action_mask = action.view(-1, 1, 1, 1, action.shape[1]) # [batch_size, 1, 1, 1, nb_blocks]
    
    out, i = forward_connection_with_policy(x, global_net.model, local_net.model, action_mask, 0)
    
    return out


class CNN_Ditto(nn.Module):
    """CNN Model used for leaf dataset in Ditto paper."""
    
    def __init__(self):
        super(CNN_Ditto, self).__init__()
        
        # Defining a 2D convolution layer
        self.cnn_layer1 = Sequential(
            
            Conv2d(1, 16, kernel_size=5, stride=1, padding="same"),
            ReLU(inplace=True)
        )
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2)
        
        # Defining another 2D convolution layer
        self.cnn_layer2 = Sequential(
            Conv2d(16, 32, kernel_size=5, stride=1, padding="same"),
            ReLU(inplace=True)
        )
        self.maxpool2 = MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = Dropout(p=0.25)
        
        # First dense layer
        self.linear_layer1 = Sequential(
            Linear(32 * 7 * 7, 128),
            ReLU(inplace=True)
        )
        self.dropout2 = Dropout(p=0.5)
        
        # Final dense layer
        self.linear_layer2 = Linear(128, 62)

    
    def forward(self, x):
        
        x = self.cnn_layer1(x)
        x = self.maxpool1(x)
        x = self.cnn_layer2(x)
        x = self.maxpool2(x)
        x = self.dropout1(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.linear_layer1(x)
        x = self.dropout2(x)
        x = self.linear_layer2(x)
        
        return x


class CNN_FedOpt(nn.Module):
    """CNN Model used for leaf dataset in FedOpt paper."""
    
    def __init__(self):
        super(CNN_FedOpt, self).__init__()
        
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 32, kernel_size=3, stride=1, padding="same"),
            # Defining another 2D convolution layer
            Conv2d(32, 64, kernel_size=3, stride=1, padding="same"),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Dropout
            Dropout(p=0.25)
        )
        
        self.linear_layers = Sequential(
            Linear(64 * 14 * 14, 128),
            ReLU(inplace=True),
            Dropout(p=0.5),
            Linear(128, 62)
        )
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
   
    
class CNN_Policy_Ditto(nn.Module):
    """ Policy net to be used with the adaptation model of Ditto CNN 
        Gives 4 outputs for the 4 layers of Ditto's CNN. """
    
    def __init__(self):
        super(CNN_Policy_Ditto, self).__init__()
        
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 16, kernel_size=5, stride=1, padding="same"),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(16, 32, kernel_size=5, stride=1, padding="same"),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Dropout
            Dropout(p=0.25)
        )
        
        self.linear_layers = Sequential(
            Linear(32 * 7 * 7, 128),
            ReLU(inplace=True),
            Dropout(p=0.5),
            Linear(128, 4)
        )
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
        
    
class CNN_FedSpot(nn.Module):
    """ Class englobing the local copy of the global model, the personalized model and the policy model for a CNN like Ditto's one """
    
    def __init__(self, dropout=True):
        
        super(CNN_FedSpot, self).__init__()
        
        self.global_model = CNN_Ditto()
        self.perso_model = CNN_Ditto()
        self.policy_model = CNN_Policy_Ditto()
        self.dropout = dropout
        
    def replace_global_model(self, new_model):
        for old_p, new_p in zip(self.global_model.parameters(), new_model.parameters()):
            old_p.data = new_p.data.clone()
            
    def replace_local_model(self, new_model):
        for old_p, new_p in zip(self.perso_model.parameters(), new_model.parameters()):
            old_p.data = new_p.data.clone()
            
    def forward(self, x):
        policy = self.policy_model(x)
        action = sigmoid(policy) # [batch_size, nb_layers]
        
        action = action.contiguous()
        action_mask = action.view(-1, 1, 1, 1, action.shape[1]) # [batch_size, 1, 1, 1, nb_layers]

        x = self.global_model.cnn_layer1(x)*action_mask[..., 0] + self.perso_model.cnn_layer1(x)*(1 - action_mask[..., 0])
        x = self.perso_model.maxpool1(x)
        
        x = self.global_model.cnn_layer2(x)*action_mask[..., 1] + self.perso_model.cnn_layer2(x)*(1 - action_mask[..., 1])
        x = self.perso_model.maxpool2(x)
        
        if self.dropout:
            x = self.perso_model.dropout1(x)
        
        x = x.view(x.size(0), -1)
        action_mask = action_mask.view(action_mask.size(0), 1, -1)
        
        x = self.global_model.linear_layer1(x)*action_mask[..., 2] + self.perso_model.linear_layer1(x)*(1 - action_mask[..., 2])
        
        if self.dropout:
            x = self.perso_model.dropout2(x)
            
        x = self.global_model.linear_layer2(x)*action_mask[..., 3] + self.perso_model.linear_layer2(x)*(1 - action_mask[..., 3])  
        
        return x, action
    
    def forward_policy(self, x):
        policy = self.policy_model(x)
        return sigmoid(policy)
    
    def aggregate_models(self, weights):
        for (idx, (perso_param, global_param)) in enumerate(zip(self.perso_model.parameters(), self.global_model.parameters())):
            perso_param.data = global_param.data.clone()*weights[idx//2] + perso_param.data.clone()*(1-weights[idx//2])
        
# --------------- Hypernetworks for FedHN ------------------
class CNNHyper(nn.Module):
    """ Server-side Hypernetwork used in FedHN with leaf """
    
    def __init__(
            self, n_nodes, embedding_dim, in_channels=1, out_dim=62, 
            hidden_dim=100, n_hidden=1):
        
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        for _ in range(n_hidden):
            layers.append(
                nn.Linear(hidden_dim, hidden_dim),
            )
            layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*layers)

        self.c1_weights = nn.Linear(hidden_dim, 16 * 1 * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, 16)
        self.c2_weights = nn.Linear(hidden_dim, 16 * 32 * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 32)
        self.l1_weights = nn.Linear(hidden_dim, 32 * 7 * 7 * 128)
        self.l1_bias = nn.Linear(hidden_dim, 128)
        self.l2_weights = nn.Linear(hidden_dim, self.out_dim * 128)
        self.l2_bias = nn.Linear(hidden_dim, self.out_dim)
        
    def forward(self, idx):
    
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            "cnn_layer1.0.weight": self.c1_weights(features).view(16, 1, 5, 5),
            "cnn_layer1.0.bias": self.c1_bias(features).view(-1),
            "cnn_layer2.0.weight": self.c2_weights(features).view(32, 16, 5, 5),
            "cnn_layer2.0.bias": self.c2_bias(features).view(-1),
            "linear_layer1.0.weight": self.l1_weights(features).view(128, 7 * 7 * 32),
            "linear_layer1.0.bias": self.l1_bias(features).view(-1),
            "linear_layer2.weight": self.l2_weights(features).view(self.out_dim, 128),
            "linear_layer2.bias": self.l2_bias(features).view(-1)
        })
        
        return weights, emd
            
    
class BraTS_Hyper(nn.Module):
    
    def __init__(self, n_nodes, embedding_dim, hidden_dim=124, n_hidden=1):

        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        
        layers = [
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        for _ in range(n_hidden):
            layers.append(
                nn.Linear(hidden_dim, hidden_dim),
            )
            layers.append(nn.ReLU(inplace=True))
        
        self.mlp = nn.Sequential(*layers)
        
        self.c1_weights = nn.Linear(hidden_dim, 16 * 4 * 3 * 3 * 3)
        self.c1_bias = nn.Linear(hidden_dim, 16)
        self.c2_weights = nn.Linear(hidden_dim, 32 * 16 * 3 * 3 * 3)
        self.c2_bias = nn.Linear(hidden_dim, 32)
        self.c3_weights = nn.Linear(hidden_dim, 64 * 32 * 3 * 3 * 3)
        self.c3_bias = nn.Linear(hidden_dim, 64)
        self.c4_weights = nn.Linear(hidden_dim, 128 * 64 * 3 * 3 * 3)
        self.c4_bias = nn.Linear(hidden_dim, 128)
        self.c5_weights = nn.Linear(hidden_dim, 256 * 128 * 3 * 3 * 3)
        self.c5_bias = nn.Linear(hidden_dim, 256)
        self.c6_weights = nn.Linear(hidden_dim, 384 * 64 * 3 * 3 * 3)
        self.c6_bias = nn.Linear(hidden_dim, 64)
        self.c7_weights = nn.Linear(hidden_dim, 128 * 32 * 3 * 3 * 3)
        self.c7_bias = nn.Linear(hidden_dim, 32)
        self.c8_weights = nn.Linear(hidden_dim, 64 * 16 * 3 * 3 * 3)
        self.c8_bias = nn.Linear(hidden_dim, 16)
        self.c9_weights = nn.Linear(hidden_dim, 32 * 3 * 3 * 3 * 3)
        self.c9_bias = nn.Linear(hidden_dim, 3)
        
    def forward(self, idx):
        
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            "model.0.conv.weight": self.c1_weights(features).view(16, 4, 3, 3, 3),
            "model.0.conv.bias": self.c1_bias(features).view(-1),
            "model.1.submodule.0.conv.weight": self.c2_weights(features).view(32, 16, 3, 3, 3),
            "model.1.submodule.0.conv.bias": self.c2_bias(features).view(-1),
            "model.1.submodule.1.submodule.0.conv.weight": self.c3_weights(features).view(64, 32, 3, 3, 3),
            "model.1.submodule.1.submodule.0.conv.bias": self.c3_bias(features).view(-1),
            "model.1.submodule.1.submodule.1.submodule.0.conv.weight": self.c4_weights(features).view(128, 64, 3, 3, 3),
            "model.1.submodule.1.submodule.1.submodule.0.conv.bias": self.c4_bias(features).view(-1),
            "model.1.submodule.1.submodule.1.submodule.1.submodule.conv.weight": self.c5_weights(features).view(256, 128, 3, 3, 3),
            "model.1.submodule.1.submodule.1.submodule.1.submodule.conv.bias": self.c5_bias(features).view(-1),
            "model.1.submodule.1.submodule.1.submodule.2.conv.weight": self.c6_weights(features).view(384, 64, 3, 3, 3),
            "model.1.submodule.1.submodule.1.submodule.2.conv.bias": self.c6_bias(features).view(-1),
            "model.1.submodule.1.submodule.2.conv.weight": self.c7_weights(features).view(128, 32, 3, 3, 3),
            "model.1.submodule.1.submodule.2.conv.bias": self.c7_bias(features).view(-1),
            "model.1.submodule.2.conv.weight": self.c8_weights(features).view(64, 16, 3, 3, 3),
            "model.1.submodule.2.conv.bias": self.c8_bias(features).view(-1),
            "model.2.conv.weight": self.c9_weights(features).view(32, 3, 3, 3, 3),
            "model.2.conv.bias": self.c9_bias(features).view(-1)
        })
        
        return weights, emd
    
    def get_last_layer_bias_as_model(self):
        
        weights = OrderedDict({
            "model.0.conv.weight": self.c1_weights.state_dict()["bias"].view(16, 4, 3, 3, 3),
            "model.0.conv.bias": self.c1_bias.state_dict()["bias"].view(-1),
            "model.1.submodule.0.conv.weight": self.c2_weights.state_dict()["bias"].view(32, 16, 3, 3, 3),
            "model.1.submodule.0.conv.bias": self.c2_bias.state_dict()["bias"].view(-1),
            "model.1.submodule.1.submodule.0.conv.weight": self.c3_weights.state_dict()["bias"].view(64, 32, 3, 3, 3),
            "model.1.submodule.1.submodule.0.conv.bias": self.c3_bias.state_dict()["bias"].view(-1),
            "model.1.submodule.1.submodule.1.submodule.0.conv.weight": self.c4_weights.state_dict()["bias"].view(128, 64, 3, 3, 3),
            "model.1.submodule.1.submodule.1.submodule.0.conv.bias": self.c4_bias.state_dict()["bias"].view(-1),
            "model.1.submodule.1.submodule.1.submodule.1.submodule.conv.weight": self.c5_weights.state_dict()["bias"].view(256, 128, 3, 3, 3),
            "model.1.submodule.1.submodule.1.submodule.1.submodule.conv.bias": self.c5_bias.state_dict()["bias"].view(-1),
            "model.1.submodule.1.submodule.1.submodule.2.conv.weight": self.c6_weights.state_dict()["bias"].view(384, 64, 3, 3, 3),
            "model.1.submodule.1.submodule.1.submodule.2.conv.bias": self.c6_bias.state_dict()["bias"].view(-1),
            "model.1.submodule.1.submodule.2.conv.weight": self.c7_weights.state_dict()["bias"].view(128, 32, 3, 3, 3),
            "model.1.submodule.1.submodule.2.conv.bias": self.c7_bias.state_dict()["bias"].view(-1),
            "model.1.submodule.2.conv.weight": self.c8_weights.state_dict()["bias"].view(64, 16, 3, 3, 3),
            "model.1.submodule.2.conv.bias": self.c8_bias.state_dict()["bias"].view(-1),
            "model.2.conv.weight": self.c9_weights.state_dict()["bias"].view(32, 3, 3, 3, 3),
            "model.2.conv.bias": self.c9_bias.state_dict()["bias"].view(-1)
        })
        
        return weights
    
   
# Hypernetwork only outputting the first 2 layers of a UNet
class Partial_BraTS_Hyper(nn.Module):
    
    def __init__(self, n_nodes, embedding_dim, hidden_dim=1024, n_hidden=2):

        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        
        layers = [
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        for _ in range(n_hidden):
            layers.append(
                nn.Linear(hidden_dim, hidden_dim),
            )
            layers.append(nn.ReLU(inplace=True))
        
        self.mlp = nn.Sequential(*layers)
        
        self.c1_weights = nn.Linear(hidden_dim, 16 * 4 * 3 * 3 * 3)
        self.c1_bias = nn.Linear(hidden_dim, 16)
        self.c2_weights = nn.Linear(hidden_dim, 32 * 16 * 3 * 3 * 3)
        self.c2_bias = nn.Linear(hidden_dim, 32)
        
    def forward(self, idx):
        
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            "model.0.conv.weight": self.c1_weights(features).view(16, 4, 3, 3, 3),
            "model.0.conv.bias": self.c1_bias(features).view(-1),
            "model.1.submodule.0.conv.weight": self.c2_weights(features).view(32, 16, 3, 3, 3),
            "model.1.submodule.0.conv.bias": self.c2_bias(features).view(-1)
        })
        
        return weights, emd
    
# Hypernetwork only outputting some of the first layers of a DynUNet
class NVIDIA_Partial_BraTS_Hyper(nn.Module):
    
    def __init__(self, n_nodes, embedding_dim, hidden_dim=1024, n_hidden=2):

        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        
        layers = [
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        for _ in range(n_hidden):
            layers.append(
                nn.Linear(hidden_dim, hidden_dim),
            )
            layers.append(nn.ReLU(inplace=True))
        
        self.mlp = nn.Sequential(*layers)
        
        self.c1_weights = nn.Linear(hidden_dim, 16 * 16 * 3 * 3 * 3)
        self.c1_bias = nn.Linear(hidden_dim, 16)
        self.c2_weights = nn.Linear(hidden_dim, 32 * 32 * 3 * 3 * 3)
        self.c2_bias = nn.Linear(hidden_dim, 32)
        self.c3_weights = nn.Linear(hidden_dim, 64 * 64 * 3 * 3 * 3)
        self.c3_bias = nn.Linear(hidden_dim, 64)
        
    def forward(self, idx):
        
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            "input_block.conv2.conv.weight": self.c1_weights(features).view(16, 16, 3, 3, 3),
            "input_block.conv2.conv.bias": self.c1_bias(features).view(-1),
            "downsamples.0.conv2.conv.weight": self.c2_weights(features).view(32, 32, 3, 3, 3),
            "downsamples.0.conv2.conv.bias": self.c2_bias(features).view(-1),
            "downsamples.1.conv2.conv.weight": self.c3_weights(features).view(64, 64, 3, 3, 3),
            "downsamples.1.conv2.conv.bias": self.c3_bias(features).view(-1),
        })
        
        return weights, emd
