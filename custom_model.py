import torch
import torch.nn as nn
import torch.nn.functional as F


#######################################
# 1. MODEL CƠ BẢN: Basic CNN
#######################################
class BasicConvBlock(nn.Module):
    """
    Basic Convolutional Block: Conv2D → BatchNorm → ReLU → MaxPooling
    LeNet, AlexNet, VGG
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=True):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        return x


class BasicCNN(nn.Module):
    """Mô hình CNN cơ bản với block đơn giản nhưng hiệu quả"""
    def __init__(self, num_classes=10, in_channels=3):
        super(BasicCNN, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            BasicConvBlock(in_channels, 32),          # 32x16x16
            BasicConvBlock(32, 64),                   # 64x8x8
            BasicConvBlock(64, 128),                  # 128x4x4
            BasicConvBlock(128, 256, pool=False),     # 256x4x4
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                  # Global average pooling: 256x1x1
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


#######################################
# 2. MODEL BÌNH THƯỜNG: Intermediated CNN
#######################################
class ResidualBlock(nn.Module):
    """
    Residual Block (ResNet): Conv → BN → ReLU → Conv → BN → (shortcut) → ReLU
    Skip connection để giải quyết vanishing gradient
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out


class IntermediateCNN(nn.Module):
    """Mô hình CNN bình thường sử dụng một số block nâng cao"""
    def __init__(self, num_classes=10, in_channels=3):
        super(IntermediateCNN, self).__init__()
        
        # Entry flow
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Feature extraction with residual blocks
        self.layer1 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        # First block may change dimensions
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks keep dimensions
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.entry(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x


#######################################
# 3. MODEL NÂNG CAO: Advanced CNN
#######################################
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (SENet): Squeeze (Global Pooling) → Excitation (FC → ReLU → FC → Sigmoid) → Scale
    Tạo trọng số cho các channel feature map
    Nâng cao khả năng tập trung vào channel có thông tin quan trọng
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class InvertedResidualBlock(nn.Module):
    """
    Inverted Residual Block (MobileNetV2): 1x1 Conv (expand) → Depthwise 3x3 → 1x1 Conv (project)
    Mở rộng chiều trước khi depthwise convolution 
    Hiệu quả về mặt tính toán, phù hợp cho thiết bị di động
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio=6, use_se=True):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.use_residual = in_channels == out_channels and stride == 1
        
        hidden_dim = int(in_channels * expand_ratio)
        
        # Expansion phase
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ) if expand_ratio != 1 else nn.Identity()
        
        # Depthwise conv
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        # SE block
        self.se = SEBlock(hidden_dim, reduction=hidden_dim // 4) if use_se else nn.Identity()
        
        # Pointwise conv
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        
        if self.use_residual:
            return x + identity
        return x


class AdvancedCNN(nn.Module):
    """Mô hình CNN nâng cao sử dụng nhiều block hiện đại"""
    def __init__(self, num_classes=10, in_channels=3, width_mult=1.0):
        super(AdvancedCNN, self).__init__()
        
        # Initial conv layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, int(32 * width_mult), 3, 2, 1, bias=False),
            nn.BatchNorm2d(int(32 * width_mult)),
            nn.ReLU6(inplace=True)
        )
        
        # Inverted Residual Blocks
        self.features = nn.Sequential(
            InvertedResidualBlock(int(32 * width_mult), int(16 * width_mult), 1, 1, False),
            InvertedResidualBlock(int(16 * width_mult), int(24 * width_mult), 2, 6, True),
            InvertedResidualBlock(int(24 * width_mult), int(24 * width_mult), 1, 6, True),
            InvertedResidualBlock(int(24 * width_mult), int(32 * width_mult), 2, 6, True),
            InvertedResidualBlock(int(32 * width_mult), int(32 * width_mult), 1, 6, True),
            InvertedResidualBlock(int(32 * width_mult), int(64 * width_mult), 2, 6, True),
            InvertedResidualBlock(int(64 * width_mult), int(64 * width_mult), 1, 6, True),
            InvertedResidualBlock(int(64 * width_mult), int(96 * width_mult), 1, 6, True),
            InvertedResidualBlock(int(96 * width_mult), int(96 * width_mult), 1, 6, True),
            InvertedResidualBlock(int(96 * width_mult), int(160 * width_mult), 2, 6, True),
            InvertedResidualBlock(int(160 * width_mult), int(320 * width_mult), 1, 6, True),
        )
        
        # Last conv layer
        self.last_conv = nn.Sequential(
            nn.Conv2d(int(320 * width_mult), int(1280 * width_mult), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(1280 * width_mult)),
            nn.ReLU6(inplace=True)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.8),
            nn.Linear(int(1280 * width_mult), num_classes)
        )
    
    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.last_conv(x)
        x = self.classifier(x)
        return x


#######################################
# 4. MODEL TỐI ƯU: Optimized CNN
#######################################
class ConvBlock(nn.Module):
    """Block Conv cơ bản với normalization và activation cho tái sử dụng"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 groups=1, activation=True, norm_layer=nn.BatchNorm2d):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.norm = norm_layer(out_channels)
        self.activation = nn.SiLU(inplace=True) if activation else nn.Identity()
    
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class SEApply(nn.Module):
    """
    Áp dụng Squeeze-Excitation (SE) module vào một đầu vào đã được chuẩn bị.
    """
    def __init__(self, se_module):
        super(SEApply, self).__init__()
        self.se_module = se_module
    
    def forward(self, x):
        return x * self.se_module(x)


class MBConvBlock(nn.Module):
    """
    MBConv Block (EfficientNet): 1x1 Conv → Depthwise Conv → SE → 1x1 Conv
    Kết hợp Inverted Residual và Squeeze-Excitation
    Cân bằng hiệu suất và độ chính xác
    """
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6, kernel_size=3, se_ratio=0.25, drop_path_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        self.drop_path_rate = drop_path_rate
        
        hidden_dim = int(in_channels * expand_ratio)
        
        # Layers
        layers = []
        
        # Expansion
        if expand_ratio != 1:
            layers.append(ConvBlock(in_channels, hidden_dim, kernel_size=1, padding=0))
        
        # Depthwise
        layers.append(ConvBlock(
            hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, 
            padding=kernel_size//2, groups=hidden_dim
        ))
        
        # SE
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            se_module = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, se_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, hidden_dim, 1),
                nn.Sigmoid()
            )
            layers.append(SEApply(se_module))
        
        # Project
        layers.append(ConvBlock(hidden_dim, out_channels, kernel_size=1, padding=0, activation=False))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            if self.training and self.drop_path_rate > 0:
                return x + self.drop_path(self.layers(x), self.drop_path_rate)
            else:
                return x + self.layers(x)
        else:
            return self.layers(x)
    
    def drop_path(self, x, drop_prob):
        """Stochastic Depth để regularization"""
        if drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Tạo attention map từ các giá trị lớn nhất và nhỏ nhất theo chiều channel
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


class OptimizedCNN(nn.Module):
    """Mô hình tối ưu kết hợp nhiều loại block hiệu quả"""
    def __init__(self, num_classes=10, in_channels=3, width_mult=1.0, depth_mult=1.0):
        super(OptimizedCNN, self).__init__()
        
        # Stem
        self.stem = ConvBlock(in_channels, int(32 * width_mult), kernel_size=3, stride=2)
        
        # Main blocks configuration
        block_configs = [
            # e: expand_ratio, c: channels, n: num_blocks, s: stride, k: kernel_size
            {'e': 1, 'c': 16, 'n': int(1 * depth_mult), 's': 1, 'k': 3},
            {'e': 6, 'c': 24, 'n': int(2 * depth_mult), 's': 2, 'k': 3},
            {'e': 6, 'c': 40, 'n': int(2 * depth_mult), 's': 2, 'k': 5},
            {'e': 6, 'c': 80, 'n': int(3 * depth_mult), 's': 2, 'k': 3},
            {'e': 6, 'c': 112, 'n': int(3 * depth_mult), 's': 1, 'k': 5},
            {'e': 6, 'c': 192, 'n': int(4 * depth_mult), 's': 2, 'k': 5},
            {'e': 6, 'c': 320, 'n': int(1 * depth_mult), 's': 1, 'k': 3},
        ]
        
        # Build blocks
        self.blocks = nn.Sequential()
        in_ch = int(32 * width_mult)
        
        for i, cfg in enumerate(block_configs):
            for j in range(cfg['n']):
                stride = cfg['s'] if j == 0 else 1
                out_ch = int(cfg['c'] * width_mult)
                drop_rate = 0.2 * (i + j) / (len(block_configs) + cfg['n'])
                
                self.blocks.add_module(
                    f'block_{i}_{j}',
                    MBConvBlock(
                        in_ch, out_ch, stride=stride, expand_ratio=cfg['e'],
                        kernel_size=cfg['k'], drop_path_rate=drop_rate
                    )
                )
                in_ch = out_ch
        
        # Head
        final_ch = int(1280 * width_mult)
        self.head = nn.Sequential(
            ConvBlock(in_ch, final_ch, kernel_size=1, padding=0),
            SpatialAttention(),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),  # Dropout with lower rate
            nn.Linear(final_ch, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.classifier(x)
        return x


# Helper function để đếm số tham số của model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Tạo và thử nghiệm các model
if __name__ == "__main__":
    import torch
    
    # Kiểm tra device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # Tạo dữ liệu đầu vào giả
    batch_size = 32
    channels = 3
    height, width = 224, 224
    x = torch.randn(batch_size, channels, height, width).to(device)
    
    # Thử nghiệm từng model
    models = [
        ('BasicCNN', BasicCNN(num_classes=1000).to(device)),
        ('IntermediateCNN', IntermediateCNN(num_classes=1000).to(device)),
        ('AdvancedCNN', AdvancedCNN(num_classes=1000, width_mult=0.75).to(device)),
        ('OptimizedCNN', OptimizedCNN(num_classes=1000, width_mult=0.75, depth_mult=0.9).to(device))
    ]
    
    for name, model in models:
        # Forward pass
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            output = model(x)
            end.record()
            
            # Đồng bộ CUDA để đo thời gian
            torch.cuda.synchronize()
            
            elapsed_time = start.elapsed_time(end)
        
        print(f"Model: {name}")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Số tham số: {count_parameters(model):,}")
        print(f"Thời gian inference: {elapsed_time:.2f} ms")
        print("-" * 50)