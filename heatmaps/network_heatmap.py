from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


def make_conv_layer(in_channels, out_channels, kernel_size, stride, padding, with_bn=True):
    """Improved conv layer with proper initialization"""
    conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding)
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
    """Stereo heatmap network following the UnrealEgo shared-backbone design.

    Architecture:
      - Shared encoder (same weights) processes left and right views independently.
      - Decoder channel-concatenates left+right feature maps at every FPN stage,
        enabling stereo feature fusion inside the decoder.
      - Output: (B, num_heatmap * 2, H, W)
          first  num_heatmap channels → left  heatmaps (logits)
          second num_heatmap channels → right heatmaps (logits)
    """
    def __init__(self, opt, model_name='resnet50'):
        super(HeatMap_Network, self).__init__()
        self.backbone = HeatMap_UnrealEgo_Shared_Backbone(opt, model_name=model_name)
        self.after_backbone = HeatMap_UnrealEgo_AfterBackbone(opt, model_name=model_name)

    def forward(self, input_left, input_right):
        """
        Args:
            input_left:  (B, 3, H, W) — left-eye images
            input_right: (B, 3, H, W) — right-eye images
        Returns:
            (B, num_heatmap * 2, H_hm, W_hm) — left then right heatmap logits
        """
        feats_left, feats_right = self.backbone(input_left, input_right)
        output = self.after_backbone(feats_left, feats_right)
        return output


class HeatMap_UnrealEgo_Shared_Backbone(nn.Module):
    """Runs both views through the same encoder weights independently."""
    def __init__(self, opt, model_name='resnet50'):
        super(HeatMap_UnrealEgo_Shared_Backbone, self).__init__()
        self.backbone = Encoder_Block(opt, model_name=model_name)

    def forward(self, input_left, input_right):
        feats_left  = self.backbone(input_left)
        feats_right = self.backbone(input_right)
        return feats_left, feats_right


class Encoder_Block(nn.Module):
    def __init__(self, opt, model_name='resnet50'):
        super(Encoder_Block, self).__init__()

        if model_name == 'convnext_tiny':
            backbone = models.convnext_tiny(weights='DEFAULT')
            # ConvNeXt-Tiny feature stages: 96 → 192 → 384 → 768
            self.layer0 = backbone.features[0]                                          # stem: C=96, stride 4
            self.layer1 = backbone.features[1]                                          # stage 1: C=96, H/4
            self.layer2 = nn.Sequential(backbone.features[2], backbone.features[3])     # downsample + stage 2: C=192, H/8
            self.layer3 = nn.Sequential(backbone.features[4], backbone.features[5])     # downsample + stage 3: C=384, H/16
            self.layer4 = nn.Sequential(backbone.features[6], backbone.features[7])     # downsample + stage 4: C=768, H/32
        elif model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=opt.init_ImageNet)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=opt.init_ImageNet)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=opt.init_ImageNet)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=opt.init_ImageNet)
        else:
            raise NotImplementedError('model type [%s] is invalid', model_name)

        if model_name != 'convnext_tiny':
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
        return [input, layer0, layer1, layer2, layer3, layer4]


class HeatMap_UnrealEgo_AfterBackbone(nn.Module):
    """FPN decoder with stereo feature fusion.

    At each FPN stage the left and right feature maps are concatenated along the
    channel dimension (matching the original UnrealEgo paper).  All conv layers
    therefore operate on 2× the single-view channel count.

    Output: (B, num_heatmap * 2, H_hm, W_hm)
        [:, :J]  — left  heatmap logits
        [:, J:]  — right heatmap logits
    """
    def __init__(self, opt, model_name="resnet50"):
        super(HeatMap_UnrealEgo_AfterBackbone, self).__init__()

        self.num_heatmap = opt.num_heatmap

        if model_name == 'convnext_tiny':
            c1, c2, c3, c4 = 96, 192, 384, 768
        elif model_name in ("resnet18", "resnet34"):
            c1, c2, c3, c4 = 64, 128, 256, 512
        elif model_name in ("resnet50", "resnet101"):
            c1, c2, c3, c4 = 256, 512, 1024, 2048
        else:
            raise NotImplementedError(f"model type [{model_name}] is invalid")

        # All 1×1 convs operate on stereo-concatenated features → 2× channels in
        self.layer1_1x1 = convrelu(c1 * 2, c1 * 2, 1, 0)
        self.layer2_1x1 = convrelu(c2 * 2, c2 * 2, 1, 0)
        self.layer3_1x1 = convrelu(c3 * 2, c3 * 2, 1, 0)
        self.layer4_1x1 = convrelu(c4 * 2, c4 * 2, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # FPN merge convs: upsampled + skip connection (both are stereo-doubled)
        self.conv_up3 = convrelu(c4 * 2 + c3 * 2, c3 * 2, 3, 1)
        self.conv_up2 = convrelu(c3 * 2 + c2 * 2, c2 * 2, 3, 1)
        self.conv_up1 = convrelu(c2 * 2 + c1 * 2, c1 * 2, 3, 1)

        self.dropout = nn.Dropout2d(0.1)

        # Joint head: outputs left heatmaps followed by right heatmaps
        self.conv_heatmap = nn.Conv2d(c1 * 2, self.num_heatmap * 2, 1)
        nn.init.kaiming_normal_(self.conv_heatmap.weight, nonlinearity='linear')
        nn.init.constant_(self.conv_heatmap.bias, 0.0)

    def forward(self, list_left, list_right):
        """
        Args:
            list_left:  [input, layer0, layer1, layer2, layer3, layer4] from left view
            list_right: [input, layer0, layer1, layer2, layer3, layer4] from right view
        """
        # Channel-wise stereo fusion at every FPN stage (paper eq.)
        # list indices: 0=raw_input, 1=layer0, 2=layer1, 3=layer2, 4=layer3, 5=layer4
        layer1 = torch.cat([list_left[2], list_right[2]], dim=1)   # (B, c1*2, H/4,  W/4)
        layer2 = torch.cat([list_left[3], list_right[3]], dim=1)   # (B, c2*2, H/8,  W/8)
        layer3 = torch.cat([list_left[4], list_right[4]], dim=1)   # (B, c3*2, H/16, W/16)
        layer4 = torch.cat([list_left[5], list_right[5]], dim=1)   # (B, c4*2, H/32, W/32)

        # FPN decoder
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

        x = self.dropout(x)
        output = self.conv_heatmap(x)   # (B, num_heatmap * 2, H_hm, W_hm)
        return output
