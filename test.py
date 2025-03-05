import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchinfo import summary


# Định nghĩa mô hình CNN với Dropout
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=1)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)  # Dropout với xác suất tắt 50%
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 1 * 1, 1)  # Fully connected layer

    def forward(self, x, num_epoch):
        self.num_epoch = num_epoch
        x = self.conv(x)
        print(f"Epoch {self.num_epoch}: After Conv = {x}")
        x = self.bn(x)
        print(f"Epoch {self.num_epoch}: After BatchNorm = {x}")
        x = self.relu(x)
        print(f"Epoch {self.num_epoch}: After ReLU = {x}")
        x = self.dropout(x)  # Thêm Dropout sau ReLU
        print(f"Epoch {self.num_epoch}: After Dropout = {x}")
        x = self.pool(x)
        print(f"Epoch {self.num_epoch}: After MaxPool = {x}")
        x = x.view(x.size(0), -1)  # Flatten
        print(f"Epoch {self.num_epoch}: After Flatten = {x}")
        x = self.fc(x)
        print(f"Epoch {self.num_epoch}: After Fully Connected = {x}")
        return x


# Khởi tạo mô hình
model = CNN()

summary(model, input_size=(1, 3, 3, 3))

# Tạo dữ liệu đầu vào (3x3x3)
x = torch.tensor(
    [
        [
            [[1.0, 2, 3], [4, 5, 6], [7, 8, 9]],  # Channel 1
            [[9.0, 8, 7], [6, 5, 4], [3, 2, 1]],  # Channel 2
            [[1.0, 1, 1], [2, 2, 2], [3, 3, 3]],  # Channel 3
        ]
    ],
    requires_grad=True,
)

# Khởi tạo kernel với giá trị cụ thể Shape (3, 3, 2, 2)
with torch.no_grad():
    model.conv.weight = nn.Parameter(
        torch.tensor(
            [
                [
                    [[1.0, -1], [2, 0]],
                    [[0.5, -0.5], [1, -1]],
                    [[-1, 1], [0.5, -0.5]],
                ],  # Kernel 1
                [
                    [[2.0, 1], [-2, 0]],
                    [[1.5, -1], [-0.5, 0.5]],
                    [[0, 0.5], [-1, 1]],
                ],  # Kernel 2
                [
                    [[1.0, -1], [2, 0]],
                    [[-1, 1], [0.5, -0.5]],
                    [[2, -2], [1, -1]],
                ],  # Kernel 3
            ]
        )
    )

# Giả sử nhãn thật
y_true = torch.tensor([[10.0]])  # Dự đoán giá trị số thực (1 giá trị)

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 200
loss_history = []

for epoch in range(num_epochs):
    model.train()  # Bật Dropout trong Training mode
    optimizer.zero_grad()  # Xóa gradient trước đó

    # Forward pass
    output = model(x, epoch)
    loss = criterion(output, y_true)
    loss_history.append(loss.item())

    # Backward pass
    loss.backward()
    optimizer.step()

    # In loss mỗi 5 epoch
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# Vẽ đồ thị loss
plt.plot(loss_history, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss giảm dần qua từng epoch với CNN")
plt.legend()
plt.show()
