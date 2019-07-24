from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet_trident_101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
#CLASS torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    def __init__(self, inplanes, planes, stride=1, downsample=None):#inplanes输入channel，planes输出channel
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, # change
                 padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class trident_block(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, padding=[1, 2, 3], dilate=[1, 2, 3]):
        super(trident_block, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilate = dilate
        self.downsample = downsample
        self.share_weight4conv1 = nn.Parameter(torch.randn(planes, inplanes, 1, 1))
        self.share_weight4conv2 = nn.Parameter(torch.randn(planes, planes, 3, 3))
        self.share_weight4conv3 = nn.Parameter(torch.randn(planes * self.expansion, planes, 1, 1))#1*1/64, 3*3/64, 1*1/256

        self.bn11 = nn.BatchNorm2d(planes)#bn层
        self.bn12 = nn.BatchNorm2d(planes)
        self.bn13 = nn.BatchNorm2d(planes * self.expansion)

        self.bn21 = nn.BatchNorm2d(planes)
        self.bn22 = nn.BatchNorm2d(planes)
        self.bn23 = nn.BatchNorm2d(planes * self.expansion)

        self.bn31 = nn.BatchNorm2d(planes)
        self.bn32 = nn.BatchNorm2d(planes)
        self.bn33 = nn.BatchNorm2d(planes * self.expansion)

        self.relu1 = nn.ReLU(inplace=True)#relu层
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward_for_small(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weight4conv1, bias=None)
        out = self.bn11(out)
        out = self.relu1(out)

        out = nn.functional.conv2d(out, self.share_weight4conv2, bias=None, stride=self.stride, padding=self.padding[0], dilation=self.dilate[0])

        out = self.bn12(out)
        out = self.relu1(out)

        out = nn.functional.conv2d(out, self.share_weight4conv3, bias=None)
        out = self.bn13(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu1(out)

        return out

    def forward_for_middle(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weight4conv1, bias=None)
        out = self.bn21(out)
        out = self.relu2(out)

        out = nn.functional.conv2d(out, self.share_weight4conv2, bias=None, stride=self.stride, padding=self.padding[1],dilation=self.dilate[1])

        out = self.bn22(out)
        out = self.relu2(out)

        out = nn.functional.conv2d(out, self.share_weight4conv3, bias=None)
        out = self.bn23(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        print(out.shape)
        print(residual.shape)

        out += residual
        out = self.relu2(out)

        return out

    def forward_for_big(self, x):
        residual = x
        out = nn.functional.conv2d(x, self.share_weight4conv1, bias=None)
        out = self.bn31(out)
        out = self.relu3(out)

        out = nn.functional.conv2d(out, self.share_weight4conv2, bias=None, stride=self.stride, padding=self.padding[2], dilation=self.dilate[2])

        out = self.bn32(out)
        out = self.relu3(out)

        out = nn.functional.conv2d(out, self.share_weight4conv3, bias=None)#对输入平面实施2D卷积
        out = self.bn33(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out

    def forward(self, x):
        xm=x
        base_feat=[]#重新定义数组
        if self.downsample is not None:#衔接段需要downsample
            x1 = self.forward_for_small(x)
            base_feat.append(x1)
            x2 = self.forward_for_middle(x)
            base_feat.append(x2)
            x3 = self.forward_for_big(x)
            base_feat.append(x3)
        else:
            x1 = self.forward_for_small(xm[0])
            base_feat.append(x1)
            x2 = self.forward_for_middle(xm[1])
            base_feat.append(x2)
            x3 = self.forward_for_big(xm[2])
            base_feat.append(x3)            
        return base_feat #三个分支

class ResNet(nn.Module):
#  def __init__(self, block, layers, num_classes=1000):#layers数组，units个数
    def __init__(self, block = Bottleneck, block1 = trident_block, layers = [3, 4, 6, 3], num_classes=1000):#layers数组，units个数
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,#通道输入3，输出64
                 bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # 3*3 maxpooling
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#    self.layer3 = trident_block(1024,256)#加入14个trident-block,输出为1024维，3个分支，feature map大小一样

        self.layer3= self._make_layer1(block1, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#需要修改
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7)#resnet自身的
        self.fc = nn.Linear(512 * block.expansion, num_classes)#全连接分类

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),#shortcut用1*1卷积
      )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))#衔接段会出现通道不匹配，需要借助downsample
        self.inplanes = planes * block.expansion#维度保持一致
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))#堆叠的block

        return nn.Sequential(*layers)#一个resnet-unit卷积

    def _make_layer1(self, block1, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block1.expansion:
            downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block1.expansion,
              kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block1.expansion),#shortcut用1*1卷积
      )

        layers = []
        layers.append(block1(self.inplanes, planes, stride, downsample))#衔接段会出现通道不匹配，需要借助downsample
        self.inplanes = planes * block1.expansion#维度保持一致
        for i in range(1, blocks):
            layers.append(block1(self.inplanes, planes))#堆叠的block

        return nn.Sequential(*layers)#一个trident-block卷积

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)#三个分支输出(进入RPN)-base feat=1*3，feature map大小一样
#在这需要分三个分支，参数共享 
        result = np.array(x)
        print(result[0].shape)
        print(result[1].shape)
        print(result[2].shape)
        x = torch.cat((x[0],x[1],x[2]),0)
#        x = x[0]+x[1]+x[2]
        print(x.shape)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.fc(x)
        return x# 1*3输出


def resnet18(pretrained=False):

    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False):

    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False):

    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False):

    model = ResNet(Bottleneck, trident_block, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def trident_50():

#  model = ResNet(Bottleneck, trident_block, [3, 4, 6, 3])#论文采用15个trident-block
    model = ResNet()
    return model

def resnet152(pretrained=False):

    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

