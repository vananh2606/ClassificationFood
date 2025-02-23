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
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # 64->64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)          # Reduce size /2
        )

        # Output: 112x112x64
        
        # Conv Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64->128 channels
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # 128->128 channels
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
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 256->256 channels
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
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 512->512 channels
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)          # Reduce size /2
        )

        # Output: 14x14x512
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1) # Reduce spatial dimensions to 1x1x512
        
        # Fully Connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),              # Expand features
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),        # Prevent overfitting
            nn.Linear(1024, 512),              # Reduce features
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)        # Final classification
        )
        
        # Initialize weights
        self._initialize_weights()
        
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
        x = self.fc(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)