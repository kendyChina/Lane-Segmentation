import torch
import torch.nn as nn
from config import CONFIG


class MixConv(nn.Module):

    def __init__(self, in_chnl, out_chnl, kernels, stride=1, **kwargs):
        super(MixConv, self).__init__()
        self.kernels = kernels
        self.groups = in_chnl // len(kernels)
        self.depth_blocks = nn.ModuleList()
        for k in kernels:
            self.depth_blocks.append(nn.Conv2d(self.groups, self.groups, k, stride=stride,
                                               groups=self.groups, padding=(k // 2), **kwargs))
        if in_chnl % len(kernels) != 0:  # prevent in_chnl from being divisible by len(kernels)
            remainder = in_chnl % len(kernels)  # E.g. in_chnl=40, len(kernels)=3
            self.depth_blocks.append(nn.Conv2d(remainder, remainder, kernels[-1], stride=stride,
                                               groups=remainder, padding=(kernels[-1] // 2), **kwargs))
        self.point_conv = nn.Conv2d(in_chnl, out_chnl, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        tensors = torch.split(x, self.groups, dim=1)
        y = []
        for i, t in enumerate(tensors):
            y.append(self.depth_blocks[i](t))
        x = torch.cat(y, dim=1)
        return self.point_conv(x)


def conv_bn_relu(in_c, out_c, k, s=1, p=0, b=False):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=b),
                         nn.BatchNorm2d(out_c), nn.ReLU())


def mixconv_bn_relu(in_c, out_c, k, s=1, b=False):
    return nn.Sequential(MixConv(in_c, out_c, k, stride=s, bias=b),
                         nn.BatchNorm2d(out_c), nn.ReLU())


class MixNet(nn.Module):

    def __init__(self):
        super(MixNet, self).__init__()
        self.conv1 = conv_bn_relu(3, 24, 3, 2, 1)  # os=2
        self.conv2 = conv_bn_relu(24, 24, 3, 1, 1)  # in, out, k, s, p, b
        self.conv3 = mixconv_bn_relu(24, 32, [3, 5, 7], 2)  # os=4
        self.conv4 = conv_bn_relu(32, 32, 3, 1, 1)
        self.conv5 = mixconv_bn_relu(32, 40, [3, 5, 7, 9], 2)  # os=8
        self.conv6 = nn.Sequential(*[mixconv_bn_relu(40, 40, [3, 5]) for _ in range(3)])
        self.conv7 = mixconv_bn_relu(40, 80, [3, 5, 7])
        self.conv8 = nn.Sequential(*[mixconv_bn_relu(80, 80, [3, 5, 7, 9]) for _ in range(3)])
        self.conv9 = conv_bn_relu(80, 120, 3, 2, 1)  # os=16
        self.conv10 = nn.Sequential(*[mixconv_bn_relu(120, 120, [3, 5, 7, 9]) for _ in range(3)])
        self.conv11 = mixconv_bn_relu(120, 200, [3, 5, 7, 9], 2)  # os=32
        self.conv12 = nn.Sequential(*[mixconv_bn_relu(200, 200, [3, 5, 7, 9]) for _ in range(3)])

    def forward(self, x):
        bridges = []
        x = self.conv1(x)
        bridges.append(x)  # [200, 120, 40, 32, 24]
        x = self.conv2(x)
        x = self.conv3(x)
        bridges.append(x)
        x = self.conv4(x)
        x = self.conv5(x)
        bridges.append(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        bridges.append(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        bridges.append(x)
        return bridges


class UNetConvBlock(nn.Module):
    def __init__(self, in_chnl, out_chnl, padding, batch_norm, bias):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_chnl, out_chnl, kernel_size=3, padding=padding, bias=bias))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chnl))
        block.append(nn.ReLU(inplace=True))

        block.append(nn.Conv2d(out_chnl, out_chnl, kernel_size=3, padding=padding, bias=bias))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chnl))
        block.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = self.block(x)
        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_chnl, out_chnl, up_mode, padding, batch_norm, bias):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_chnl, out_chnl, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_chnl, out_chnl, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(out_chnl*2, out_chnl, padding=padding,
                                        batch_norm=batch_norm, bias=bias)

    def center_crop(self, layer, target_size):
        _, _, layer_h, layer_w = layer.size()
        delta_h = (layer_h - target_size[0]) // 2
        delta_w = (layer_w - target_size[1]) // 2

        return layer[:, :, delta_h: delta_h + target_size[0],
               delta_w: delta_w + target_size[1]]

    def forward(self, x, bridge):
        x = self.up(x)
        bridge = self.center_crop(bridge, x.shape[-2:])
        x = torch.cat([bridge, x], 1)
        x = self.conv_block(x)
        return x


class MixNetUNet(nn.Module):
    def __init__(self, in_chnl=3, base_chnl=32, n_classes=8, depth=5,
                 padding=1, batch_norm=True, bias=False, up_mode="upconv", **kwargs):
        super(MixNetUNet, self).__init__()
        assert up_mode in ("upconv", "upsample")
        self.backbone = MixNet()
        self.depth = depth

        chnls = [24, 32, 40, 120, 200]

        self.up_path = nn.ModuleList()
        for i, in_chnl in enumerate(reversed(chnls[1:])):
            self.up_path.append(UNetUpBlock(in_chnl, chnls[-2-i], up_mode,
                                            padding, batch_norm, bias=bias))
        self.last1 = nn.ConvTranspose2d(chnls[0], chnls[0], kernel_size=2, stride=2)
        self.last2 = nn.Conv2d(chnls[0], n_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        bridges = self.backbone(x)

        x = bridges[-1]
        for i, up in enumerate(self.up_path):
            x = up(x, bridges[-2 - i])
        x = self.last2(self.last1(x))

        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = MixNetUNet()
    model.eval()
    x = torch.randn((1, 3, 256, 96))
    y = model(x)
    y = torch.argmax(torch.softmax(y, dim=1), dim=1)
    print(y)
    print(torch.max(y))
    print(y.shape)
    # x = torch.randn((3, 256, 96))
    # summary(model, (3, 256, 96))
    # model = MixConv(24, 24, [3, 5])
    # x = torch.randn((1, 24, 256, 96))
    # model.eval()
    # model(x)
