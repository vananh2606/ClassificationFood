import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(CustomCNN, self).__init__()

        # Input: 224x224x3
        
        # Conv Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),    # 3->64 channels
            nn.BatchNorm2d(64),                            # Normalize
            nn.ReLU(inplace=True),                         # Activation
            nn.MaxPool2d(kernel_size=2, stride=2)          # Reduce size /2
        )

        # Output: 112x112x64
        
        # Conv Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64->128 channels
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)          # Reduce size /2
        )

        # Output: 56x56x128
        
        # Conv Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 128->256 channels
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)          # Reduce size /2
        )

        # Output: 28x28x256
        
        # Conv Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # 256->512 channels
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)          # Reduce size /2
        )

        # Output: 14x14x512
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1) # Reduce spatial dimensions to 1x1x512
        
        # Fully Connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),        # Prevent overfitting
            nn.Linear(512, num_classes)        # Final classification
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
    
class CustomStem(nn.Module):
    def __init__(self, num_classes,  dropout_rate=0.2):
        super(CustomStem, self).__init__()
        
        # Stem: Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        # Output: 112x112x32
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Output: 56x56x64
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Output: 28x28x128
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Output: 14x14x256
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Output: 7x7x512
        
        # Global Average Pooling and Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Output: 1x1x512
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
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