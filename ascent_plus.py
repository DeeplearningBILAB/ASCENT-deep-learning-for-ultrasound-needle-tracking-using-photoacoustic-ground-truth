import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, resnet101, mobilenet_v3_large
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, MobileNet_V3_Large_Weights


class ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPPModule, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride=16, dilation_rates=None, aspp_channels=256, drop_rate=0.2):
        super(ASPP, self).__init__()
        ASPP_CONFIGS = {
            '1': [1],
            '1_3': [1, 3],
            '1_3_6': [1, 3, 6],
            '1_3_6_12': [1, 3, 6, 12],
            '1_3_6_12_18': [1, 3, 6, 12, 18],
            '1_6_12_18': [1, 6, 12, 18],
            '3_6_12_18': [3, 6, 12, 18],
        }
        if output_stride == 16:
            if aspp_channels <= 128:
                default_dilations = [1, 3, 6]  # ASPP (1,3,6) for smaller networks
            else:
                default_dilations = [1, 3, 6, 12, 18]  # ASPP (1,3,6,12,18) for larger networks
                
        elif output_stride == 8:
            if aspp_channels <= 128:
                default_dilations = [1, 6, 12, 18]  # ASPP (1,6,12,18) for smaller networks at OS=8
            else:
                default_dilations = [3, 6, 12, 18]  # ASPP (3,6,12,18) for larger networks at OS=8
        else:
            raise NotImplementedError(f"Unsupported output stride: {output_stride}")

        dilations = dilation_rates if dilation_rates is not None else default_dilations
        self.dilations = dilations
        self.aspp_channels = aspp_channels
        self.aspp_modules = nn.ModuleList([ASPPModule(inplanes, aspp_channels, rate=r) for r in dilations])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, aspp_channels, 1, stride=1, bias=False),
            nn.ReLU(inplace=True)
        )

        concat_channels = aspp_channels * (len(dilations) + 1)
        self.conv1 = nn.Conv2d(concat_channels, aspp_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(aspp_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_rate)
        self._init_weight()

    def forward(self, x):
        branch_outputs = [m(x) for m in self.aspp_modules]
        x6 = self.global_avg_pool(x)
        x6 = F.interpolate(x6, size=branch_outputs[-1].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat(branch_outputs + [x6], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_inplanes=256, aspp_channels=256, low_level_channels=48, drop_rate=0.2):
        super(Decoder, self).__init__()
        self.aspp_channels = aspp_channels
        self.low_level_channels = low_level_channels
        fusion_channels = aspp_channels + low_level_channels

        self.conv1 = nn.Conv2d(low_level_inplanes, low_level_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(low_level_channels)
        self.relu = nn.ReLU(inplace=True)

        self.last_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, aspp_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Conv2d(aspp_channels, aspp_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.side1 = nn.Conv2d(aspp_channels, num_classes, kernel_size=1)
        self.side2 = nn.Conv2d(aspp_channels, num_classes, kernel_size=1)
        self.side3 = nn.Conv2d(aspp_channels, num_classes, kernel_size=1)
        self.side4 = nn.Conv2d(aspp_channels, num_classes, kernel_size=1)
        self.side5 = nn.Conv2d(aspp_channels, num_classes, kernel_size=1)
        self.side6 = nn.Conv2d(aspp_channels, num_classes, kernel_size=1)

        self.outconv = nn.Conv2d(6 * num_classes, num_classes, 1)

        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        # Upsample the high-level features to match low-level features
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)

        # Concatenate low-level and high-level features
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        # Generate multiple side outputs
        d1 = self.side1(x)
        d2 = self.side2(x)
        d3 = self.side3(x)
        d4 = self.side4(x)
        d5 = self.side5(x)
        d6 = self.side6(x)

        # Final output
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return d0, d1, d2, d3, d4, d5, d6

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AdaptiveContrastBoostModule(nn.Module):
    """
    Adaptive Contrast Boost Module.
    Contrast-aware attention mechanism with residual connections.
    """

    def __init__(self, in_channels):
        super(AdaptiveContrastBoostModule, self).__init__()

        # Step 1: Contrast-Aware Feature Enhancement
        self.contrast_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()  # Attention weights
        )

        # Step 2: Noise Suppression
        self.noise_suppression = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self._init_weights()

    def forward(self, x):
        """
        Forward pass implementing the two-stage enhancement process.

        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            Enhanced feature map [B, C, H, W]
        """
        # Step 1: Contrast-Aware Feature Enhancement
        attention_weights = self.contrast_attention(x)

        # Residual Feature Recalibration: F_enhanced
        enhanced_features = x * attention_weights + x

        # Step 2: Noise Suppression
        output = self.noise_suppression(enhanced_features)

        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SpatialPatchAttentionModule(nn.Module):
    """
    Applies patch-wise pooling to low-level features
    using local context, preserving needle-relevant spatial cues while suppressing speckle.
    """
    def __init__(self, in_channels, reduction_ratio=4, patch_size=8):
        super(SpatialPatchAttentionModule, self).__init__()
        self.patch_size = patch_size
        reduced = max(in_channels // reduction_ratio, 8)
        self.conv_reduce = nn.Conv2d(in_channels, reduced, 1)
        self.bn = nn.BatchNorm2d(reduced)
        self.conv_restore = nn.Conv2d(reduced, in_channels, 1)
        self._init_weights()

    def forward(self, x):
        B, C, H, W = x.shape
        h_p, w_p = self.patch_size, self.patch_size
        Hp, Wp = max(H // h_p, 1), max(W // w_p, 1)
        # Patch-wise pooling: descriptor map Z
        z = F.adaptive_avg_pool2d(x, (Hp, Wp))
        # Bottleneck gating: reduce -> ReLU -> restore -> sigmoid
        a = self.conv_reduce(z)
        a = F.relu(self.bn(a), inplace=True)
        a = torch.sigmoid(self.conv_restore(a))
        # Upsample to original spatial size
        a = F.interpolate(a, size=(H, W), mode='bilinear', align_corners=True)
        # Residual: X' = X + X * A
        return x + x * a

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabv3Plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, output_stride=16, pretrained=True, backbone_name='resnet50',
                 dilation_rates=None, aspp_channels=256, low_level_channels=48, drop_rate=0.2):
        super(DeepLabv3Plus, self).__init__()

        self.backbone_name = backbone_name
        self.aspp_channels = aspp_channels
        self.low_level_channels = low_level_channels

        self.backbone, backbone_features = self._create_backbone(backbone_name, pretrained, in_channels)

        if output_stride == 16 and 'resnet' in backbone_name:
            for n, m in self.backbone.named_modules():
                if 'layer4' in n:
                    if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                        m.stride = (1, 1)
                    elif isinstance(m, nn.MaxPool2d) and m.stride == 2:
                        m.stride = 1

        self.aspp = ASPP(backbone_features, output_stride, dilation_rates=dilation_rates, aspp_channels=aspp_channels, drop_rate=drop_rate)

        low_level_features = self._get_low_level_features(backbone_name)
        self.decoder = Decoder(num_classes, low_level_features, aspp_channels, low_level_channels, drop_rate=drop_rate)

    def _create_backbone(self, backbone_name, pretrained, in_channels):
        """Create backbone networke"""
        if backbone_name == 'resnet50':
            if pretrained:
                backbone = resnet50(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                backbone = resnet50(weights=None)
            if in_channels != 3:
                backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone_features = 2048

        elif backbone_name == 'resnet101':
            if pretrained:
                backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
            else:
                backbone = resnet101(weights=None)

            if in_channels != 3:
                backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone_features = 2048

        elif backbone_name == 'mobilenetv3':
            if pretrained:
                backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            else:
                backbone = mobilenet_v3_large(weights=None)

            if in_channels != 3:
                backbone.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)

            backbone = backbone.features
            backbone_features = 960

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        return backbone, backbone_features

    def _get_low_level_features(self, backbone_name):
        """Get low-level feature dimensions for each backbone"""
        if backbone_name == 'resnet50':
            return 64  # ResNet18 layer1
        if backbone_name in ['resnet101']:
            return 256  # ResNet layer1 output channels
        elif backbone_name == 'mobilenetv3':
            return 24
        else:
            return 256  # Default

        # Initialize weights
        self._init_weight()

    def forward(self, x):
        input_size = x.size()[2:]

        # Extract features
        if 'resnet' in self.backbone_name:
            for i, layer in enumerate(self.backbone):
                x = layer(x)
                if i == 4:  # After layer1 (low-level features)
                    low_level_features = x
        elif self.backbone_name == 'mobilenetv3':
            for i, layer in enumerate(self.backbone):
                x = layer(x)
                if i == 3:
                    low_level_features = x
        else:
            raise ValueError(f"Forward pass not implemented for backbone: {self.backbone_name}")

        # Apply ASPP
        x = self.aspp(x)

        # Decoder with skip connections
        x, d1, d2, d3, d4, d5, d6 = self.decoder(x, low_level_features)

        # Upsample
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        d2 = F.interpolate(d2, size=input_size, mode='bilinear', align_corners=True)
        d3 = F.interpolate(d3, size=input_size, mode='bilinear', align_corners=True)
        d4 = F.interpolate(d4, size=input_size, mode='bilinear', align_corners=True)
        d5 = F.interpolate(d5, size=input_size, mode='bilinear', align_corners=True)
        d6 = F.interpolate(d6, size=input_size, mode='bilinear', align_corners=True)

        # Return logits (sigmoid applied in loss / inference)
        return x, d1, d2, d3, d4, d5, d6

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AscentPlus(DeepLabv3Plus):
    def __init__(self, in_channels=3, num_classes=1, output_stride=16, pretrained=True, backbone_name='resnet50',
                 use_contrast_boost=True, use_spam=True, dilation_rates=None, aspp_channels=256, low_level_channels=48, drop_rate=0.2):
        super(AscentPlus, self).__init__(in_channels, num_classes, output_stride, pretrained, backbone_name,
                                         dilation_rates=dilation_rates, aspp_channels=aspp_channels,
                                         low_level_channels=low_level_channels, drop_rate=drop_rate)

        self.use_contrast_boost = use_contrast_boost
        self.use_spam = use_spam
        fusion_channels = aspp_channels + low_level_channels

        if use_contrast_boost:
            self.adaptive_contrast_boost = AdaptiveContrastBoostModule(fusion_channels)
        else:
            self.adaptive_contrast_boost = None

        if use_spam:
            low_level_inplanes = self.decoder.conv1.in_channels
            self.spam = SpatialPatchAttentionModule(low_level_inplanes, reduction_ratio=4, patch_size=8)
        else:
            self.spam = None

    def forward(self, x):
        input_size = x.size()[2:]

        if 'resnet' in self.backbone_name:
            for i, layer in enumerate(self.backbone):
                x = layer(x)
                if i == 4:  # After layer1 (low-level features)
                    low_level_features = x
        elif self.backbone_name == 'mobilenetv3':
            for i, layer in enumerate(self.backbone):
                x = layer(x)
                if i == 3:
                    low_level_features = x
        else:
            raise ValueError(f"Forward pass not implemented for backbone: {self.backbone_name}")

        # ASPP on high-level features
        x = self.aspp(x)

        if self.spam is not None:
            low_level_features = self.spam(low_level_features)

        # Reduce low-level: 1x1 conv
        low_level_features = self.decoder.conv1(low_level_features)
        low_level_features = self.decoder.bn1(low_level_features)
        low_level_features = self.decoder.relu(low_level_features)

        # Upsample high-level to match low-level spatial dims
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        # Concat
        x = torch.cat((x, low_level_features), dim=1)

        if self.adaptive_contrast_boost is not None:
            x = self.adaptive_contrast_boost(x)

        # Feature fusion + side outputs
        x = self.decoder.last_conv(x)
        d1 = self.decoder.side1(x)
        d2 = self.decoder.side2(x)
        d3 = self.decoder.side3(x)
        d4 = self.decoder.side4(x)
        d5 = self.decoder.side5(x)
        d6 = self.decoder.side6(x)
        x = self.decoder.outconv(torch.cat((d1, d2, d3, d4, d5, d6), dim=1))

        # Upsample to original image size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        d2 = F.interpolate(d2, size=input_size, mode='bilinear', align_corners=True)
        d3 = F.interpolate(d3, size=input_size, mode='bilinear', align_corners=True)
        d4 = F.interpolate(d4, size=input_size, mode='bilinear', align_corners=True)
        d5 = F.interpolate(d5, size=input_size, mode='bilinear', align_corners=True)
        d6 = F.interpolate(d6, size=input_size, mode='bilinear', align_corners=True)

        # Return logits (sigmoid applied in loss / inference)
        return x, d1, d2, d3, d4, d5, d6