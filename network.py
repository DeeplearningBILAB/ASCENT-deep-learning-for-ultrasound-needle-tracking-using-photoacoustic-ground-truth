import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, mobilenet_v3_large, ResNet50_Weights, ResNet101_Weights, \
    MobileNet_V3_Large_Weights


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
    def __init__(self, inplanes, output_stride=16):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPModule(inplanes, 256, rate=dilations[0])
        self.aspp2 = ASPPModule(inplanes, 256, rate=dilations[1])
        self.aspp3 = ASPPModule(inplanes, 256, rate=dilations[2])
        self.aspp4 = ASPPModule(inplanes, 256, rate=dilations[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

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
    def __init__(self, num_classes, low_level_inplanes=256):
        super(Decoder, self).__init__()

        # For ResNet50
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        # Fusion of low-level and high-level features
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Multi-scale outputs
        self.side1 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.side2 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.side3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.side4 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.side5 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.side6 = nn.Conv2d(256, num_classes, kernel_size=1)

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


class FeaturePyramidAttention(nn.Module):
    def __init__(self, in_channels, lambda_val=0.1):
        super(FeaturePyramidAttention, self).__init__()
        self.lambda_val = lambda_val

        # Pyramid pooling branches
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool5 = nn.AdaptiveAvgPool2d(5)

        # 1x1 convs for dimension reduction
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv3 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv5 = nn.Conv2d(in_channels, in_channels // 4, 1)

        # Fusion conv
        self.fusion = nn.Conv2d(in_channels, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Self-attention components
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)

        self._init_weights()

    def forward(self, x):
        # Pyramid pooling
        p1 = F.interpolate(self.conv1(self.pool1(x)), size=x.shape[2:], mode='bilinear', align_corners=True)
        p2 = F.interpolate(self.conv2(self.pool2(x)), size=x.shape[2:], mode='bilinear', align_corners=True)
        p3 = F.interpolate(self.conv3(self.pool3(x)), size=x.shape[2:], mode='bilinear', align_corners=True)
        p5 = F.interpolate(self.conv5(self.pool5(x)), size=x.shape[2:], mode='bilinear', align_corners=True)

        # Fuse pyramid features (concat + conv)
        fused = torch.cat([p1, p2, p3, p5], dim=1)
        fpa_out = self.relu(self.bn(self.fusion(fused)))

        # Self-attention
        B, C, H, W = fpa_out.shape
        Q = self.query(fpa_out).view(B, C, -1)
        K = self.key(fpa_out).view(B, C, -1)
        V = self.value(fpa_out).view(B, C, -1)

        attention = torch.softmax(torch.bmm(Q.transpose(1, 2), K) / (C ** 0.5), dim=-1)
        context = torch.bmm(V, attention.transpose(1, 2))
        context = context.view(B, C, H, W)

        # Residual connection
        output = fpa_out + self.lambda_val * context

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


class DeepLabv3Plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, output_stride=16, pretrained=True, backbone_name='resnet50'):
        super(DeepLabv3Plus, self).__init__()

        self.backbone_name = backbone_name

        self.backbone, backbone_features = self._create_backbone(backbone_name, pretrained, in_channels)

        if output_stride == 16 and 'resnet' in backbone_name:
            for n, m in self.backbone.named_modules():
                if 'layer4' in n:
                    if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                        m.stride = (1, 1)
                    elif isinstance(m, nn.MaxPool2d) and m.stride == 2:
                        m.stride = 1

        # ASPP module with backbone-specific feature channels
        self.aspp = ASPP(backbone_features, output_stride)

        # Decoder with backbone-specific low-level features
        low_level_features = self._get_low_level_features(backbone_name)
        self.decoder = Decoder(num_classes, low_level_features)

    def _create_backbone(self, backbone_name, pretrained, in_channels):
        """Create backbone networke"""
        if backbone_name == 'resnet50':
            if pretrained:
                backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                backbone = resnet50(weights=None)

            # Adjust input channels
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
        if backbone_name in ['resnet50', 'resnet101']:
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

        # Apply sigmoid for binary segmentation
        return torch.sigmoid(x), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AscentPlus(DeepLabv3Plus):
    def __init__(self, in_channels=3, num_classes=1, output_stride=16, pretrained=True, backbone_name='resnet50'):
        super(AscentPlus, self).__init__(in_channels, num_classes, output_stride, pretrained, backbone_name)

        # Apply to decoder features (256 channels)
        self.adaptive_contrast_boost = AdaptiveContrastBoostModule(256)

        self.fpa = FeaturePyramidAttention(256)  # FPA with default lambda=0.1

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

        # Apply ASPP
        x = self.aspp(x)

        # Apply Adaptive Contrast Boost Module to high-level features
        x = self.adaptive_contrast_boost(x)
        x = self.fpa(x)  # Apply FPA after contrast boost

        # Decoder with skip connections
        x, d1, d2, d3, d4, d5, d6 = self.decoder(x, low_level_features)

        # Upsample to original image size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)
        d2 = F.interpolate(d2, size=input_size, mode='bilinear', align_corners=True)
        d3 = F.interpolate(d3, size=input_size, mode='bilinear', align_corners=True)
        d4 = F.interpolate(d4, size=input_size, mode='bilinear', align_corners=True)
        d5 = F.interpolate(d5, size=input_size, mode='bilinear', align_corners=True)
        d6 = F.interpolate(d6, size=input_size, mode='bilinear', align_corners=True)

        # Apply sigmoid for binary segmentation
        return torch.sigmoid(x), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)
