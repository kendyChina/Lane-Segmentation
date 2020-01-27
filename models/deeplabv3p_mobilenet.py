import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG


class SeperateConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, dilation=1):
        super(SeperateConv, self).__init__(
            # depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      stride, groups=in_channels, bias=False,
                      padding=(kernel_size // 2) * dilation,
                      dilation=dilation),
            # nn.BatchNorm2d(in_channels),
            # nn.ReLU(inplace=True),
            # pointwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=1, groups=1, bias=False,
                      dilation=1, padding=0),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )


class MobileNet(nn.Module):

    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1) # os=2
        self.conv2 = SeperateConv(32, 64, 3, 1)
        self.conv3 = SeperateConv(64, 128, 3, 2) # os=4
        self.conv4 = SeperateConv(128, 128, 3, 1)
        self.conv5 = SeperateConv(128, 256, 3, 2) # os=8
        self.conv6 = SeperateConv(256, 256, 3, 1)
        stride = 1
        if CONFIG.OUTPUT_STRIDE == 8:
            stride = 1
        elif CONFIG.OUTPUT_STRIDE == 16:
            stride = 2
        self.conv7 = SeperateConv(256, 512, 3, stride) # os=8 or 16
        self.blocks8 = nn.Sequential(*[SeperateConv(512, 512, 3, 1) for _ in range(5)])
        self.conv9 = SeperateConv(512, 1024, 3, 1, dilation=2)
        self.conv10 = SeperateConv(1024, 1024, 3, 1, dilation=2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        decode_shortcut = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.blocks8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        return decode_shortcut, x



def mobilenetv2(pretrained=True):
    model = MobileNet()

    return model


class ASPP(nn.Module):

    def __init__(self, in_chans, out_chans, rate=1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, 1, padding=0, dilation=rate, bias=True),
            # nn.BatchNorm2d(out_chans),
            nn.GroupNorm(CONFIG.GROUPS_FOR_NORM, out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            # nn.BatchNorm2d(out_chans),
            nn.GroupNorm(CONFIG.GROUPS_FOR_NORM, out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            # nn.BatchNorm2d(out_chans),
            nn.GroupNorm(CONFIG.GROUPS_FOR_NORM, out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            # nn.BatchNorm2d(out_chans),
            nn.GroupNorm(CONFIG.GROUPS_FOR_NORM, out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch5_avg = nn.AdaptiveAvgPool2d(1)
        self.branch5_conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=True)
        # self.branch5_bn = nn.BatchNorm2d(out_chans)
        self.branch5_gn = nn.GroupNorm(CONFIG.GROUPS_FOR_NORM, out_chans)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_chans * 5, out_chans, 1, 1, padding=0, bias=True),
            # nn.BatchNorm2d(out_chans),
            nn.GroupNorm(CONFIG.GROUPS_FOR_NORM, out_chans),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, h, w = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = self.branch5_avg(x)
        global_feature = self.branch5_relu(self.branch5_gn(self.branch5_conv(global_feature)))
        global_feature = F.interpolate(global_feature, (h, w), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

class MobileNetDeeplabV3Plus(nn.Module):

    def __init__(self, pretrained=True):
        super(MobileNetDeeplabV3Plus, self).__init__()
        self.backbone = mobilenetv2(pretrained=pretrained)
        input_channel = 1024
        self.aspp = ASPP(in_chans=input_channel, out_chans=CONFIG.ASPP_OUTDIM, rate=16 // CONFIG.OUTPUT_STRIDE)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=CONFIG.OUTPUT_STRIDE // 4)

        indim = 128
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, CONFIG.SHORTCUT_DIM, CONFIG.SHORTCUT_KERNEL, 1, padding=CONFIG.SHORTCUT_KERNEL // 2,
                      bias=False),
            # nn.BatchNorm2d(CONFIG.SHORTCUT_DIM),
            nn.GroupNorm(12, CONFIG.SHORTCUT_DIM),
            nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(CONFIG.ASPP_OUTDIM + CONFIG.SHORTCUT_DIM, CONFIG.ASPP_OUTDIM, 3, 1, padding=1, bias=False),
            # nn.BatchNorm2d(CONFIG.ASPP_OUTDIM),
            nn.GroupNorm(CONFIG.GROUPS_FOR_NORM, CONFIG.ASPP_OUTDIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(CONFIG.ASPP_OUTDIM, CONFIG.ASPP_OUTDIM, 3, 1, padding=1, bias=False),
            # nn.BatchNorm2d(CONFIG.ASPP_OUTDIM),
            nn.GroupNorm(CONFIG.GROUPS_FOR_NORM, CONFIG.ASPP_OUTDIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(CONFIG.ASPP_OUTDIM, CONFIG.NUM_CLASSES, 1, 1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        layers = self.backbone(x)
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)
        return result


if __name__ == '__main__':
    from torchsummary import summary
    model = MobileNetDeeplabV3Plus()
    summary(model, (3, 256, 96))
    x = torch.randn((1, 3, 256, 96))
    model.eval()
    y = model(x)
    print(x.size())
    print(y.size())