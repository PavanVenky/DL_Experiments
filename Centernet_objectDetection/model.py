"""
CenterNet: Objects as Points (Zhou et al., 2019)
https://arxiv.org/abs/1904.07850

Architecture recap:
    Backbone (ResNet-18, ImageNet-pretrained)
        -> Neck (3x transposed-conv upsampling, stride 32 -> stride 4)
        -> Head (3 parallel conv branches: heatmap, offset, size)

Key idea: detect objects as a single point (their bbox center) on a
heatmap, then regress box size + sub-pixel offset at that point.
No anchors, no NMS-heavy multi-scale matching -- just a keypoint
detection problem.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CenterNet(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # ---------------- Backbone ----------------
        # Strip the avgpool + fc layers off a standard classification
        # backbone. We keep everything up through layer4 (stride 32).
        backbone = getattr(models, backbone_name)(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )  # stride 4
        self.layer1 = backbone.layer1  # stride 4
        self.layer2 = backbone.layer2  # stride 8
        self.layer3 = backbone.layer3  # stride 16
        self.layer4 = backbone.layer4  # stride 32

        # ResNet18/34 use BasicBlock (out=512), ResNet50+ use Bottleneck (out=2048)
        backbone_out_channels = 512 if backbone_name in ("resnet18", "resnet34") else 2048

        # ---------------- Neck ----------------
        # 3 transposed-conv (deconv) layers: stride 32 -> 16 -> 8 -> 4.
        # This is what turns a classification backbone into something
        # that can output a dense, high-resolution heatmap.
        self.deconv_layers = self._make_deconv_layers(
            in_channels=backbone_out_channels,
            num_layers=3,
            num_filters=[256, 128, 64],
        )
        head_in_channels = 64

        # ---------------- Heads ----------------
        # Every head is a small 2-conv branch on the shared upsampled
        # feature map. Output spatial size = input_size / 4.

        # 1) Heatmap head: one channel per class. Each pixel = probability
        #    that an object center of that class is here.
        self.heatmap_head = self._make_head(head_in_channels, num_classes, fill_bias=True)

        # 2) Offset head: 2 channels (dx, dy). Corrects the quantization
        #    error introduced by the stride-4 downsampling (since the
        #    true center rarely lands exactly on an integer grid cell).
        self.offset_head = self._make_head(head_in_channels, 2, fill_bias=False)

        # 3) Size head: 2 channels (w, h) of the box, regressed directly
        #    in input-image pixel scale.
        self.size_head = self._make_head(head_in_channels, 2, fill_bias=False)

    @staticmethod
    def _make_deconv_layers(in_channels, num_layers, num_filters):
        layers = []
        for i in range(num_layers):
            out_ch = num_filters[i]
            layers += [
                nn.ConvTranspose2d(
                    in_channels, out_ch, kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_ch
        return nn.Sequential(*layers)

    @staticmethod
    def _make_head(in_channels, out_channels, fill_bias: bool):
        head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1),
        )
        if fill_bias:
            # Standard CenterNet/RetinaNet trick: initialize the heatmap
            # bias so that at the start of training every pixel predicts
            # a LOW foreground probability (~0.01) instead of 0.5.
            # Without this, focal loss is unstable for the first many
            # iterations because of the huge background/foreground
            # imbalance (thousands of background pixels per object).
            import math
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            head[-1].bias.data.fill_(bias_value)
        return head

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)          # (B, C, H/32, W/32)

        feat = self.deconv_layers(x)  # (B, 64, H/4, W/4)

        heatmap = torch.sigmoid(self.heatmap_head(feat))  # (B, num_classes, H/4, W/4)
        offset = self.offset_head(feat)                    # (B, 2, H/4, W/4)
        size = self.size_head(feat)                         # (B, 2, H/4, W/4)

        return {"heatmap": heatmap, "offset": offset, "size": size}


if __name__ == "__main__":
    model = CenterNet(num_classes=20, pretrained=False)
    dummy = torch.randn(2, 3, 512, 512)
    out = model(dummy)
    for k, v in out.items():
        print(k, v.shape)
    # Expected with 512x512 input, stride 4 output:
    # heatmap torch.Size([2, 20, 128, 128])
    # offset  torch.Size([2, 2, 128, 128])
    # size    torch.Size([2, 2, 128, 128])
