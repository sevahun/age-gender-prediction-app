### AGE AND GENDER PREDICTION MODEL ####
# Age and gender prediction model is implemented by us in another repository. You can check the original implementation
# of model here: https://github.com/SeoulTech-HCIRLab/Relative-Age-Position-Learning.git


import torch
import torch.nn as nn


def load_state_dict(self, state_dict):
    model_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in model_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        model_state[name].copy_(param)


class IR50_EVR_AgeRM_GP(nn.Module):
    def __init__(self, age_num):
        super().__init__()
        self.age_num = age_num
        self.backbone = Backbone()
        self.fc_gender = nn.Linear(256, 2)
        self.fc_age = nn.Linear(256, self.age_num)
        self.fc_feature = nn.Linear(self.age_num, 1)
        self.fc_pos = nn.Linear(256, self.age_num)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = torch.nn.Softmax(dim=1)
        self.parameter = Parameter(torch.Tensor(self.age_num))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)  # [b, 256, 14, 14]
        x = torch.flatten(self.avgpool(x), 1)  # [b, 256]

        age_out = self.softmax(self.fc_age(x))  # [b, 256] --> [b, n]
        gender_out = self.fc_gender(x)  # [b, 256] --> [b, 2]

        return age_out, gender_out


#### IR50 MODEL THAT IS USED AS BACKBONE ####

# Original Arcface Model:
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/iresnet.py
# THIS implementation of IR50 is from:
# https://github.com/Talented-Q/POSTER_V2/blob/89d72bda05736880663672b7a530383372a5a6d2/models/ir50.py


from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
from collections import namedtuple


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''
    # print('50')


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks1 = [
            get_block(in_channel=64, depth=64, num_units=3)
        ]
        blocks2 = [
            get_block(in_channel=64, depth=128, num_units=4)
        ]
        blocks3 = [
            get_block(in_channel=128, depth=256, num_units=14),
        ]

    return blocks1, blocks2, blocks3


class Backbone(Module):
    def __init__(self, num_layers=50, drop_ratio=0.0, mode='ir'):
        super(Backbone, self).__init__()
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks1, blocks2, blocks3 = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, 512),
                                       BatchNorm1d(512))
        modules1 = []
        for block in blocks1:
            for bottleneck in block:
                modules1.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))

        modules2 = []
        for block in blocks2:
            for bottleneck in block:
                modules2.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))

        modules3 = []
        for block in blocks3:
            for bottleneck in block:
                modules3.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))

        self.body1 = Sequential(*modules1)
        self.body2 = Sequential(*modules2)
        self.body3 = Sequential(*modules3)

    def forward(self, x):
        x = F.interpolate(x, size=112)
        x = self.input_layer(x)
        x1 = self.body1(x)
        x2 = self.body2(x1)
        x3 = self.body3(x2)
        return x3
