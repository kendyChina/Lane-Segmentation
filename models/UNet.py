import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class UNetConvBlock(nn.Module):
    def __init__(self, in_chnl, out_chnl, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_chnl, out_chnl, kernel_size=3, padding=padding))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chnl))
        block.append(nn.ReLU(inplace=True))

        block.append(nn.Conv2d(out_chnl, out_chnl, kernel_size=3, padding=padding))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chnl))
        block.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = self.block(x)
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_chnl, out_chnl, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_chnl, out_chnl, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_chnl, out_chnl, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_chnl, out_chnl, padding=padding,
                                        batch_norm=batch_norm)

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

class UNet(nn.Module):
    def __init__(self, in_chnl=3, base_chnl=64, n_classes=8, depth=5,
                 padding=1, batch_norm=False, up_mode="upconv", **kwargs):
        super(UNet, self).__init__()
        assert up_mode in ("upconv", "upsample")
        self.depth = depth

        out_chnl = base_chnl
        self.down_path = nn.ModuleList()
        for _ in range(depth):
            out_chnl *= 2
            # To keep i' == i    padding = (kernel_size - 1) //2
            self.down_path.append(UNetConvBlock(in_chnl, out_chnl,
                                                padding=padding, batch_norm=batch_norm))
            in_chnl = out_chnl

        self.up_path = nn.ModuleList()
        for _ in range(depth - 1):
            in_chnl //= 2
            self.up_path.append(UNetUpBlock(out_chnl, in_chnl, up_mode,
                                            padding, batch_norm))
            out_chnl = in_chnl
        self.last = nn.Conv2d(in_chnl, n_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print(x.shape)
        bridges = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != self.depth - 1:
                bridges.append(x)
                x = F.max_pool2d(x, kernel_size=2, stride=2)

        for i, up in enumerate(self.up_path):
            x = up(x, bridges[-1 - i])

        x = self.last(x)

        return x

if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    # x = Image.open(r"E:\code\cv\baidu_data_set\baidu_Image_Data\Road02\ColorImage_road02\ColorImage\Record001\Camera 5\170927_063811892_Camera_5.jpg")
    # x = transforms.ToTensor()(x)

    model = UNet()
    model.eval()
    summary(model, (3, 1024, 384))
    # y = model(x)
    # print(x.shape)
    # print(y.shape)
