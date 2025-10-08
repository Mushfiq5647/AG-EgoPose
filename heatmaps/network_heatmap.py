from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


def make_conv_layer(in_channels, out_channels, kernel_size, stride, padding, with_bn=True):
    """Improved conv layer with proper initialization"""
    conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding)
    
    # Proper initialization for ReLU/LeakyReLU (use fan_out for stability)
    torch.nn.init.kaiming_uniform_(conv.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')
    torch.nn.init.constant_(conv.bias, 0)
    
    bn = torch.nn.BatchNorm2d(num_features=out_channels)
    relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    if with_bn:
        return torch.nn.Sequential(conv, bn, relu)
    else:
        return torch.nn.Sequential(conv, relu)


def make_deconv_layer(in_channels, out_channels, kernel_size, stride, padding, with_bn=True):
    """Improved deconv layer with proper initialization"""
    conv = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding)
    
    # Proper initialization for transposed conv (use fan_out for stability)
    torch.nn.init.kaiming_uniform_(conv.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')
    torch.nn.init.constant_(conv.bias, 0)
    
    bn = torch.nn.BatchNorm2d(num_features=out_channels)
    relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    if with_bn:
        return torch.nn.Sequential(conv, bn, relu)
    else:
        return torch.nn.Sequential(conv, relu)


def make_fc_layer(in_feature, out_feature, with_relu=True, with_bn=True):
    """Improved FC layer with proper initialization"""
    modules = OrderedDict()
    fc = torch.nn.Linear(in_feature, out_feature)
    
    # Proper initialization for linear layers
    torch.nn.init.kaiming_uniform_(fc.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
    torch.nn.init.constant_(fc.bias, 0)
    
    modules['fc'] = fc
    bn = torch.nn.BatchNorm1d(num_features=out_feature)
    relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    if with_bn is True:
        modules['bn'] = bn

    if with_relu is True:
        modules['relu'] = relu

    return torch.nn.Sequential(modules)


def convrelu(in_channels, out_channels, kernel, padding):
    """Improved conv+relu with proper initialization"""
    conv = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
    torch.nn.init.kaiming_uniform_(conv.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')
    torch.nn.init.constant_(conv.bias, 0)
    
    return nn.Sequential(
        conv,
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class HeatMap_Network(nn.Module):
    def __init__(self, opt, model_name='resnet50'):
        super(HeatMap_Network, self).__init__()
        self.backbone = HeatMap_UnrealEgo_Shared_Backbone(opt, model_name=model_name)
        self.after_backbone = HeatMap_UnrealEgo_AfterBackbone(opt, model_name=model_name)

    def forward(self, input_rgb):
        x = self.backbone(input_rgb)
        output = self.after_backbone(x)
        return output


class HeatMap_UnrealEgo_Shared_Backbone(nn.Module):
    def __init__(self, opt, model_name='resnet50'):
        super(HeatMap_UnrealEgo_Shared_Backbone, self).__init__()
        self.backbone = Encoder_Block(opt, model_name=model_name)

    def forward(self, input_rgb):
        output = self.backbone(input_rgb)
        return output


class Encoder_Block(nn.Module):
    def __init__(self, opt, model_name='resnet50'):
        super(Encoder_Block, self).__init__()

        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=opt.init_ImageNet)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=opt.init_ImageNet)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=opt.init_ImageNet)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=opt.init_ImageNet)
        else:
            raise NotImplementedError('model type [%s] is invalid', model_name)

        self.base_layers = list(self.backbone.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

    def forward(self, input):
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        output = [input, layer0, layer1, layer2, layer3, layer4]
        return output


class HeatMap_UnrealEgo_AfterBackbone(nn.Module):
    def __init__(self, opt, model_name="resnet50"):
        super(HeatMap_UnrealEgo_AfterBackbone, self).__init__()

        if model_name == 'resnet18':
            feature_scale = 1
        elif model_name == "resnet34":
            feature_scale = 1
        elif model_name == "resnet50":
            feature_scale = 2
        elif model_name == "resnet101":
            feature_scale = 2
        else:
            raise NotImplementedError('model type [%s] is invalid', model_name)

        self.num_heatmap = opt.num_heatmap

        if model_name in ("resnet18", "resnet34"):
            c1, c2, c3, c4 = 64, 128, 256, 512
        elif model_name in ("resnet50", "resnet101"):
            c1, c2, c3, c4 = 256, 512, 1024, 2048
        else:
            raise NotImplementedError(f"model type [{model_name}] is invalid")

        # 1x1 convolutions for channel reduction
        self.layer0_1x1 = convrelu(c1, c1, 1, 0)
        self.layer1_1x1 = convrelu(c1, c1, 1, 0)
        self.layer2_1x1 = convrelu(c2, c2, 1, 0)
        self.layer3_1x1 = convrelu(c3, c3, 1, 0)
        self.layer4_1x1 = convrelu(c4, c4, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Improved FPN-style decoder with better feature fusion
        self.conv_up3 = convrelu(c4 + c3, c3, 3, 1)
        self.conv_up2 = convrelu(c3 + c2, c2, 3, 1)
        self.conv_up1 = convrelu(c2 + c1, c1, 3, 1)
        self.conv_up0 = convrelu(c1 + c1, c1, 3, 1)

        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
        
        # Final heatmap head - adjusted for sigma=2 heatmaps (wider, lower intensity)
        self.conv_heatmap = nn.Conv2d(c1, self.num_heatmap, 1)
        nn.init.kaiming_normal_(self.conv_heatmap.weight, nonlinearity='linear')
        nn.init.constant_(self.conv_heatmap.bias, 0.0)  # sigmoid(-4.6) ≈ 0.01 background prior

    def forward(self, list_rgb_features):
        input = list_rgb_features[0]
        layer0 = list_rgb_features[1]
        layer1 = list_rgb_features[2]
        layer2 = list_rgb_features[3]
        layer3 = list_rgb_features[4]
        layer4 = list_rgb_features[5]

        # Process through FPN decoder
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        # print("X shape after convup1", x.shape)

        # x = self.upsample(x)
        # layer0 = self.layer0_1x1(layer0)
        # x = torch.cat([x, layer0], dim=1)
        # x = self.conv_up0(x)
        # # print("X shape after convup0", x.shape)

        output = self.conv_heatmap(x)
        # print("output shape after conv heatmap",output.shape)
        
        return output
