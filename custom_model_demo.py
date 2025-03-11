import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNNMinus(nn.Module):
    """
    Mô hình CNN Cơ bản tùy chỉnh để phân loại hình ảnh.

    Mô hình này bao gồm một loạt các khối tích chập, sau đó là Global Average Pooling
    và các lớp Fully Connected để phân loại.

    Args:
        num_classes (int): Số lượng lớp cần phân loại.
        dropout_rate (float, optional): Tỷ lệ dropout để áp dụng cho lớp Fully Connected. Mặc định là 0.5.

    Attributes:
        conv1 (nn.Sequential): Khối tích chập đầu tiên.
        conv2 (nn.Sequential): Khối tích chập thứ hai.
        conv3 (nn.Sequential): Khối tích chập thứ ba.
        conv4 (nn.Sequential): Khối tích chập thứ tư.
        gap (nn.AdaptiveAvgPool2d): Lớp Global Average Pooling.
        classifier (nn.Sequential): Các lớp Fully Connected để phân loại.

    Methods:
        forward(x): Thực hiện chuyển tiếp (forward pass) cho mô hình.
    """
    def __init__(self, num_classes, dropout_rate=0.5):
        super(CustomCNNMinus, self).__init__()

        # Input: 224x224x3

        # Conv Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 3->64 channels
            nn.BatchNorm2d(64),  # Normalize
            nn.ReLU(inplace=True),  # Activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce size /2
        )

        # Output: 112x112x64

        # Conv Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64->128 channels
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce size /2
        )

        # Output: 56x56x128

        # Conv Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 128->256 channels
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce size /2
        )

        # Output: 28x28x256

        # Conv Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 256->512 channels
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce size /2
        )

        # Output: 14x14x512

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # Reduce spatial dimensions to 1x1x512

        # Fully Connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # Prevent overfitting
            nn.Linear(512, num_classes),  # Final classification
        )

    def forward(self, x):
        # Conv blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Global average pooling
        x = self.gap(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.classifier(x)

        return x

class CustomCNN(nn.Module):
    """
    Mạng Nơ-ron Tích Chập (CNN) tùy chỉnh cho bài toán phân loại ảnh.

    Mô hình này bao gồm bốn khối tích chập để trích xuất đặc trưng từ ảnh đầu vào,
    tiếp theo là cơ chế chú ý giúp tập trung vào các vùng quan trọng, và cuối cùng là
    bộ phân loại với các lớp kết nối đầy đủ.

    Args:
        num_classes (int, tùy chọn): Số lượng lớp đầu ra. Mặc định là 10.
        dropout_rate (float, tùy chọn): Tỷ lệ dropout để tránh overfitting. Mặc định là 0.3.

     Attributes:
        conv1 (nn.Sequential): Khối tích chập đầu tiên để trích xuất đặc trưng cơ bản.
        conv2 (nn.Sequential): Khối tích chập thứ hai để trích xuất đặc trưng kết cấu.
        conv3 (nn.Sequential): Khối tích chập thứ ba để trích xuất đặc trưng phức tạp.
        conv4 (nn.Sequential): Khối tích chập thứ tư để trích xuất đặc trưng cấp cao.
        attention (nn.Sequential): Cơ chế chú ý không gian giúp làm nổi bật vùng quan trọng.
        gap (nn.AdaptiveAvgPool2d): Lớp Pooling trung bình toàn cục để giảm kích thước không gian.
        classifier (nn.Sequential): Các lớp kết nối đầy đủ với dropout để phân loại.

    Methods:
        forward(x): Lan truyền tiến qua mạng.
        _initialize_weights(): Khởi tạo trọng số của mô hình.
    """
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(CustomCNN, self).__init__()

        # Input: 224x224x3

        # Conv Block 1 - Trích xuất đặc trưng cơ bản
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Thêm convolution để trích xuất đặc trưng sâu hơn
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Output: 112x112x64

        # Conv Block 2 - Trích xuất đặc trưng kết cấu
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Thêm convolution
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Output: 56x56x128

        # Conv Block 3 - Trích xuất đặc trưng phức tạp
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Thêm convolution để nhận diện chi tiết
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Output: 28x28x256

        # Conv Block 4 - Trích xuất đặc trưng cấp cao
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Output: 14x14x512
        
        # Attention mechanism - Tập trung vào các vùng quan trọng của hình ảnh
        self.attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully Connected layers với dropout để ngăn overfitting
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate/2),  # Giảm dropout ở lớp thứ hai
            nn.Linear(256, num_classes),
        )
        
        # Khởi tạo trọng số
        self._initialize_weights()

    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Spatial attention
        attention = self.attention(x)
        x = x * attention
        
        # Global average pooling
        x = self.gap(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


class CustomCNNPlus(nn.Module):
    """
     Mạng CNN mở rộng với các cơ chế chú ý và pooling hiện đại.

    Mô hình này cải tiến từ CustomCNN với các thành phần mới như attention không gian,
    attention kênh, dilation convolution và squeeze-and-excitation để tăng hiệu suất.
        
    Args:
        num_classes (int, tùy chọn): Số lượng lớp đầu ra. Mặc định là 10.
        dropout_rate (float, tùy chọn): Tỷ lệ dropout để tránh overfitting. Mặc định là 0.2.

    Attributes:
        stem (nn.Sequential): Lớp tích chập ban đầu để trích xuất đặc trưng tổng quát.
        block1-4 (nn.Sequential): Các khối tích chập để trích xuất đặc trưng theo từng cấp độ.
        spatial_attention1 (nn.Sequential): Cơ chế chú ý không gian sau khối 1.
        channel_attention (nn.Sequential): Cơ chế chú ý kênh sau khối 3.
        global_pool (nn.AdaptiveAvgPool2d): Pooling trung bình toàn cục để giảm kích thước đặc trưng.
        se (nn.Sequential): Cơ chế squeeze-and-excitation để điều chỉnh trọng số đặc trưng.
        classifier (nn.Sequential): Lớp phân loại với dropout để tránh overfitting.

    Methods:
        forward(x): Lan truyền tiến qua mạng.
        _initialize_weights(): Khởi tạo trọng số của mô hình.
    """
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(CustomCNNPlus, self).__init__()

        # Input: 224x224x3

        # Stem: Initial convolution với kernel lớn để bắt đặc trưng kết cấu
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Output: 112x112x32

        # Block 1: Focus on color and basic texture
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Output: 56x56x64
        
        # Spatial attention sau block 1 - tập trung vào đặc trưng không gian quan trọng
        self.spatial_attention1 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )  

        # Block 2: Extract more complex features
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Thêm một lớp tích chập để tăng khả năng biểu diễn
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # Output: 28x28x128

        # Block 3: Focus on food-specific textures and patterns
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Dilation để tăng vùng tiếp nhận mà không giảm kích thước
            nn.Conv2d(256, 256, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),  # 1x1 conv để tổng hợp thông tin
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Output: 14x14x256
        
        # Channel attention sau block 3 - tập trung vào đặc trưng kênh quan trọng
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 256, kernel_size=1),
            nn.Sigmoid()
        )

        # Block 4: High-level feature extraction
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # Output: 7x7x512

        # Global Average Pooling thay vì Fully Connected layers để giảm số tham số
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Output: 1x1x512

        # Thêm squeeze-and-excitation block trước khi phân loại 
        self.se = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()
        )

        # Phân loại với lớp "bottleneck" trước khi ra kết quả cuối
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),  # Bottleneck layer 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(128, num_classes),
        )

        # Khởi tạo trọng số theo phương pháp Kaiming
        self._initialize_weights()

    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Block 1 với spatial attention
        x = self.block1(x)
        spa_att = self.spatial_attention1(x)
        x = x * spa_att
        
        # Block 2
        x = self.block2(x)
        
        # Block 3 với channel attention
        x = self.block3(x)
        ch_att = self.channel_attention(x)
        x = x * ch_att
        
        # Block 4
        x = self.block4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x_flat = x.view(x.size(0), -1)
        
        # Squeeze-and-excitation
        se_weights = self.se(x_flat)
        x_weighted = x_flat * se_weights
        
        # Classification
        output = self.classifier(x_weighted)
        
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

class CustomCNNPro(nn.Module):
    """
    Mạng phân loại tối ưu với các cơ chế tiên tiến như feature pyramid, attention,
    inception-style blocks và dilated convolution để trích xuất đặc trưng hiệu quả.

    Model Architecture:
    - **Stem Conv**: Khối tích chập ban đầu giúp giảm kích thước ảnh.
    - **Pyramid Features**: Cấu trúc multi-scale feature extraction với nhiều kích thước kernel.
    - **Downsample Blocks**: Các khối giảm kích thước với convolution stride 2.
    - **Inception-style Block**: Tổng hợp đặc trưng bằng nhiều nhánh tích chập khác nhau.
    - **Attention Mechanism**: Tập trung vào vùng quan trọng trong ảnh.
    - **Dilated Convolution**: Tăng receptive field mà không mất thông tin không gian.
    - **Feature Pyramid Block**: Tăng cường khả năng nhận diện đặc trưng trừu tượng.
    - **Global Pooling**: Kết hợp trung bình và cực đại để tổng hợp thông tin.
    - **Fully Connected Layers**: Lớp kết nối đầy đủ với dropout và batch normalization.

    Args:
        num_classes (int, tùy chọn): Số lượng lớp đầu ra. Mặc định là 10.
        dropout_rate (float, tùy chọn): Tỷ lệ dropout để tránh overfitting. Mặc định là 0.3.

    Attributes:
        stem_conv (nn.Sequential): Khối tích chập ban đầu để giảm kích thước ảnh đầu vào.
        pyramid_features (nn.ModuleList): Các nhánh tích chập đa tỷ lệ để trích xuất đặc trưng ở nhiều kích thước.
        down1 (nn.Sequential): Khối giảm kích thước đầu tiên.
        inception1 (nn.ModuleList): Khối inception tổng hợp đặc trưng.
        down2 (nn.Sequential): Khối giảm kích thước thứ hai.
        attention (nn.Sequential): Cơ chế chú ý giúp làm nổi bật vùng quan trọng.
        down3 (nn.Sequential): Khối giảm kích thước thứ ba với tích chập giãn nở.
        feature_pyramid (nn.Sequential): Khối feature pyramid giúp tổng hợp đặc trưng cấp cao.
        avg_pool (nn.AdaptiveAvgPool2d): Lớp Pooling trung bình toàn cục.
        max_pool (nn.AdaptiveMaxPool2d): Lớp Pooling cực đại toàn cục.
        fc (nn.Sequential): Các lớp kết nối đầy đủ để phân loại.

    Methods:
        forward(x): Lan truyền tiến qua mạng để dự đoán đầu ra.
        _initialize_weights(): Khởi tạo trọng số của mô hình.
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(CustomCNNPro, self).__init__()
        
        # Input: 224x224x3
        
        # Efficient stem với pyramid pooling để bắt đặc trưng ở nhiều kích thước
        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Output: 112x112x32
        
        # Multi-scale feature extraction để nhận diện các đặc trưng ở nhiều kích thước
        self.pyramid_features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 3x3 conv
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=5, padding=2),  # 5x5 conv
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=7, padding=3),  # 7x7 conv
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=1),  # 1x1 conv
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
        ])
        # Output: 112x112x64 (16*4 channels)
        
        # Downsample block 1
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Output: 56x56x128
        
        # Inception-style block 1
        self.inception1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 32, kernel_size=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 48, kernel_size=1),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 32, kernel_size=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 48, kernel_size=5, padding=2),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(128, 32, kernel_size=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        ])
        # Output: 56x56x176 (32+64+48+32)
        
        # Downsample block 2
        self.down2 = nn.Sequential(
            nn.Conv2d(176, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Output: 28x28x256
        
        # Attention block - tập trung vào các vùng quan trọng
        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Downsample block 3 với dilated convolution
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        # Output: 14x14x384
        
        # Feature pyramid block - bắt đặc trưng ở nhiều mức trừu tượng
        self.feature_pyramid = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Output: 7x7x512
        
        # Global pooling với cả average và max pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 1x1x512
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 1x1x512
        
        # Fully connected layers với dropout và batch normalization
        self.fc = nn.Sequential(
            nn.Linear(512*2, 512),  # Kết hợp avg_pool và max_pool
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, num_classes)
        )
        
        # Khởi tạo trọng số
        self._initialize_weights()
    
    def forward(self, x):
        # Stem
        x = self.stem_conv(x)
        
        # Pyramid feature extraction (multi-scale)
        pyramid_out = []
        for pyramid in self.pyramid_features:
            pyramid_out.append(pyramid(x))
        x = torch.cat(pyramid_out, dim=1)  # 16*4=64 channels
        
        # Down1
        x = self.down1(x)  # 128 channels
        
        # Inception block
        inception_out = []
        for path in self.inception1:
            inception_out.append(path(x))
        x = torch.cat(inception_out, dim=1)  # 176 channels
        
        # Down2
        x = self.down2(x)  # 256 channels
        
        # Attention mechanism
        att = self.attention(x)
        x = x * att
        
        # Down3 with dilated conv
        x = self.down3(x)  # 384 channels
        
        # Feature pyramid
        x = self.feature_pyramid(x)  # 512 channels
        
        # Dual pooling
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)
        
        # Flatten
        avg_x = avg_x.view(avg_x.size(0), -1)  # 512
        max_x = max_x.view(max_x.size(0), -1)  # 512
        
        # Concatenate different pooling results
        x = torch.cat([avg_x, max_x], dim=1)  # 1024
        
        # FC layers
        x = self.fc(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)