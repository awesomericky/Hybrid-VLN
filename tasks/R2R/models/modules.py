import math
import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_mlp(input_dim, hidden_dims, output_dim=None,
              use_batchnorm=False, dropout=0, fc_bias=True, relu=True):
    layers = []
    D = input_dim
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    if hidden_dims:
        for dim in hidden_dims:
            layers.append(nn.Linear(D, dim, bias=fc_bias))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            if relu:
                layers.append(nn.ReLU(inplace=True))
            D = dim
    if output_dim:
        layers.append(nn.Linear(D, output_dim, bias=fc_bias))
    return nn.Sequential(*layers)


class SoftAttention(nn.Module):
    """Soft-Attention without learnable parameters
    """

    def __init__(self, extra_mask_needed=0, batch_size=None, device=None):
        super(SoftAttention, self).__init__()
         
        self.softmax = nn.Softmax(dim=1)
        self.extra_mask_needed = extra_mask_needed
        if extra_mask_needed:
           self.stop_idx_mask = torch.ones(batch_size).unsqueeze(-1).float().to(device)

    def forward(self, h, proj_context, context=None, mask=None, reverse_attn=False):
        """Propagate h through the network.

        h: batch x dim (concat(img, action))
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        """
        # Get attention
        attn = torch.bmm(proj_context, h.unsqueeze(2)).squeeze(2)  # batch x seq_len

        if reverse_attn:
            attn = -attn
        
        if mask is not None:
            if self.extra_mask_needed:
               mask = torch.cat((mask, self.stop_idx_mask), dim=1)
            attn.data.masked_fill_((mask == 0).data, -float('inf'))
        attn = self.softmax(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        if context is not None:
            weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        else:
            weighted_context = torch.bmm(attn3, proj_context).squeeze(1)  # batch x dim

        return weighted_context, attn

class PositionalEncoding(nn.Module):
    """Implement the PE function to introduce the concept of relative position"""

    def __init__(self, d_model, dropout, max_len=80):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 2i + 1
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)  #10
        return self.dropout(x)  #10

class Dynamic_conv2d_attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_filters, temperature=30, init_weight=True):
        super(Dynamic_conv2d_attention, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, num_filters, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            for n in m.modules():
               if isinstance(n, nn.Linear):
                   nn.init.kaiming_normal_(n.weight, mode='fan_out', nonlinearity='relu')
                   if n.bias is not None:
                      nn.init.constant_(n.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_filters, kernel_size, input_channel, stride, padding, dilation, \
                output_channel=1, groups=1, bias=True, temperature=30, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.num_filters = num_filters
        self.attention = Dynamic_conv2d_attention(input_dim, hidden_dim, num_filters, temperature)

        # self.weight = nn.Parameter(torch.randn(num_filters, output_channel//groups, input_channel//groups, kernel_size, kernel_size), requires_grad=True)
        self.weights = []
        self.bias = []
        for _ in range(num_filters):
            weight = torch.randn(output_channel, input_channel, kernel_size, kernel_size)
            self.weights.append(weight)
            if bias:
                self.bias.append(torch.zeros(output_channel))

        if init_weight:
            self._initialize_weights()
        
        self._parametrize()

    def _initialize_weights(self):
        for i in range(self.num_filters):
            nn.init.kaiming_uniform_(self.weights[i])
    
    def _parametrize(self):
        self.weights = nn.Parameter(torch.stack(self.weights), requires_grad=True)
        self.bias = nn.Parameter(torch.stack(self.bias), requires_grad=True)

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, filter_condition_input, model_input):
        softmax_attention = self.attention(filter_condition_input)
        batch_size, input_channel, height, width = model_input.size()
        weight = self.weights.view(self.num_filters, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.input_channel, self.kernel_size, self.kernel_size)
        outputs = []
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias)
            
            for i in range(batch_size):
                output = F.conv2d(model_input[i].view(self.groups, -1, height, width), weight=aggregate_weight[i].unsqueeze(0), bias=aggregate_bias[i], stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, groups=1)
                outputs.append(output.squeeze(1))
        else:
            for i in range(batch_size):
                output = F.conv2d(model_input[i].view(self.groups, -1, height, width), weight=aggregate_weight[i].unsqueeze(0), bias=None, stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, groups=1)
                outputs.append(output.squeeze(1))
        outputs = torch.stack(outputs).squeeze(1)

        return outputs, softmax_attention

def create_mask(batchsize, max_length, length):
    """Given the length create a mask given a padded tensor"""
    tensor_mask = torch.zeros(batchsize, max_length)
    for idx, row in enumerate(tensor_mask):
        row[:length[idx]] = 1
    return tensor_mask.to(device)

def create_new_mask(batchsize, max_length, batch_indexs):
    """Given the index create a mask given a padded tensor"""
    tensor_mask = torch.zeros(batchsize, max_length)
    for batch_num, indexs in enumerate(batch_indexs):
        for index in indexs:
            tensor_mask[batch_num, index] = 1
    return tensor_mask.to(device)

def proj_masking(feat, projector, mask=None):
    """Universal projector and masking"""
    proj_feat = projector(feat.view(-1, feat.size(2)))
    proj_feat = proj_feat.view(feat.size(0), feat.size(1), -1)
    if mask is not None:
        return proj_feat * mask.unsqueeze(2).expand_as(proj_feat)
    else:
        return proj_feat
