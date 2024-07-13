import logging

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Pre_Net(nn.Module):
    def __init__(self):
        super(Pre_Net, self).__init__()
        self.res = resnet50_backbone(pretrained=True)
        self.quality_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(7, 7)),
        )
        self.distortion_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(7, 7)),
        )

        self.fc1 = nn.Sequential(  # 2分类
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.fc2 = nn.Sequential(  # 6分类
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, img1, img2):
        res_out_1 = self.res(img1)  # 2048,7,7
        res_out_2 = self.res(img2)
        # input features for 2 class
        two_class_input = res_out_1 - res_out_2
        Conv1_out = self.quality_conv(two_class_input).squeeze(3).squeeze(2)
        two_class_out = self.fc1(Conv1_out)

        # input features for 6 class
        img1_Conv2_out = self.distortion_conv(res_out_1).squeeze(3).squeeze(2)
        img2_Conv2_out = self.distortion_conv(res_out_2).squeeze(3).squeeze(2)

        img1_six_class_out = self.fc2(img1_Conv2_out)
        img2_six_class_out = self.fc2(img2_Conv2_out)

        out = {}
        out['two_class_out'] = two_class_out
        out['img1_six_class_out'] = img1_six_class_out
        out['img2_six_class_out'] = img2_six_class_out

        return out


class PINet(nn.Module):
    def __init__(self):
        super(PINet, self).__init__()
        self.res = resnet50_backbone(pretrained=True)
        # 图像质量差异感知器
        self.quality_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(7, 7)),
        )
        # 图像是失真类型感知器
        self.distortion_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(7, 7)),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, img):
        res_out = self.res(img)  # b,2048,7,7

        q1 = self.quality_conv[0](res_out)
        d1 = self.distortion_conv[0](res_out)
        attention1 = torch.sigmoid(q1 @ d1)
        q2 = self.quality_conv[1](q1 * attention1)
        d2 = self.distortion_conv[1](d1 * attention1)
        attention2 = torch.sigmoid(q2 @ d2)
        q3 = self.quality_conv[2](q2 * attention2)
        d3 = self.distortion_conv[2](d2 * attention2)
        attention3 = torch.sigmoid(q3 @ d3)
        q4 = self.quality_conv[3](q3 * attention3)
        d4 = self.distortion_conv[3](d3 * attention3)
        out = torch.cat([q4, d4], dim=1)
        out = F.relu(out)
        out = F.avg_pool2d(out, [7, 7]).squeeze(3).squeeze(2)

        out = self.fc(out)

        # import numpy as np
        # img_name = 'img44'
        # for i in range(1, 5):
        #     path = '../vector/二维/fastfading/中/'+img_name+'_q' + str(i) + '.txt'
        #     data = {'q1': q1.squeeze(0).cpu().detach().numpy(),
        #             'q2': q2.squeeze(0).cpu().detach().numpy(),
        #             'q3': q3.squeeze(0).cpu().detach().numpy(),
        #             'q4': q4.squeeze(0).cpu().detach().numpy()}
        #     with open(path, 'w') as outfile:
        #         for slice_2d in data['q' + str(i)]:
        #             np.savetxt(outfile, slice_2d, fmt='%f', delimiter=',')
        #
        #     path = '../vector/二维/fastfading/中/'+img_name+'_d' + str(i) + '.txt'
        #     data = {'d1': d1.squeeze(0).cpu().detach().numpy(),
        #             'd2': d2.squeeze(0).cpu().detach().numpy(),
        #             'd3': d3.squeeze(0).cpu().detach().numpy(),
        #             'd4': d4.squeeze(0).cpu().detach().numpy()}
        #     with open(path, 'w') as outfile:
        #         for slice_2d in data['d' + str(i)]:
        #             np.savetxt(outfile, slice_2d, fmt='%f', delimiter=',')

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
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


class ResNetBackbone(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.path_conv1 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        self.path_conv2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(64, 512, kernel_size=(1, 1), stride=(1, 1)),
        )
        self.path_conv3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(128, 1024, kernel_size=(1, 1), stride=(1, 1)),
        )
        self.path_conv4 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(256, 2048, kernel_size=(1, 1), stride=(1, 1)),
        )

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
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f1 = self.maxpool(x)  # b,64,56,56
        f2 = self.layer1(f1)  # b,256,56,56
        f3 = self.layer2(f2)  # b,512,28,28
        f4 = self.layer3(f3)  # b,1024,14,14
        f5 = self.layer4(f4)  # b,2048,7,7

        f1_1 = self.path_conv1(f1) + f2
        f1_2 = self.path_conv2(f1_1) + f3
        f1_3 = self.path_conv3(f1_2) + f4
        f1_4 = self.path_conv4(f1_3)

        f2_2 = self.path_conv2(f2) + f3
        f2_3 = self.path_conv3(f2_2) + f4
        f2_4 = self.path_conv4(f2_3)

        f3_3 = self.path_conv3(f3) + f4
        f3_4 = self.path_conv4(f3_3)

        f4_4 = self.path_conv4(f4)

        out = f5 + f1_4 + f2_4 + f3_4 + f4_4
        return out


def resnet50_backbone(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_hyper.构造一个ResNet-50model_hyper

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
            预训练(bool)：如果为True，则在ImageNet上返回预训练的model_hyper
    """
    model = ResNetBackbone(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
