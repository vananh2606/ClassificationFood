import os
import torch
import cv2 as cv
import numpy as np
import shutil
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, utils
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from preprocessing_data import (
    CustomDataset,
    class_names,
    get_transform,
    visualie_dataloader,
)
from custom_model import CustomCNN, CustomCNNPlus
from train_model import train_model
from evaluate_model import evaluate_model


def main():
    # Cấu hình
    TRAIN_PATH = "dataset/train"
    VAL_PATH = "dataset/val"
    TEST_PATH = "dataset/test"
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001

    BEST_MODEL_PATH = "models/KMS/best_model_google.pth"
    FINAL_MODEL_PATH = "models/KMS/final_model_google.pth"

    # Xác định device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Khởi tạo TensorBoard writer
    shutil.rmtree("runs/", ignore_errors=True)  # Xóa thư mục runs nếu tồn tại
    writer = SummaryWriter("runs/Mymodel")  # Ghi log vào thư mục cố định

    # Transform Data
    train_transform, val_test_transform = get_transform((224, 224))

    # Tạo dataset
    train_dataset = CustomDataset(
        TRAIN_PATH, transform=train_transform, max_samples=140
    )
    val_dataset = CustomDataset(VAL_PATH, transform=val_test_transform, max_samples=40)
    test_dataset = CustomDataset(
        TEST_PATH, transform=val_test_transform, max_samples=20
    )

    # Tạo dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train Dataset Size: {len(train_dataloader.dataset)}")
    print(f"Val Dataset Size: {len(val_dataloader.dataset)}")
    print(f"Test Dataset Size: {len(test_dataloader.dataset)}")
    print(f"Train Dataloader Size: {len(train_dataloader)}")
    print(f"Val Dataloader Size: {len(val_dataloader)}")
    print(f"Test Dataloader Size: {len(test_dataloader)}")

    # Số lượng classes
    num_classes = len(class_names)
    print(f"Number Classes: {num_classes}")
    for idx, class_name in enumerate(class_names):
        print(f"  {idx}: {class_name}")

    # Show images in dataloader
    images, labels = next(iter(train_dataloader))
    visualie_dataloader(BATCH_SIZE, images, labels)

    # # Load mô hình ResNet18 pre-trained
    # model = models.resnet18(pretrained=True)
    # # Load mô hình GoogleNet pre-trained
    model = models.googlenet(pretrained=True)
    # # Load mô hình EfficientNet pre-trained
    # model = models.efficientnet_b0(pretrained=True)

    # Freeze các layers
    for param in model.parameters():
        param.requires_grad = False

    # Thay đổi fully connected layer cuối cùng
    if hasattr(model, "fc"):
        num_features = model.fc.in_features  # ResNet18, GoogleNet
        model.fc = nn.Linear(num_features, num_classes)  # ResNet18, GoogleNet
    if hasattr(model, "classifier"):
        num_features = model.classifier[1].in_features  # ENetB0
        model.classifier[1] = nn.Linear(num_features, num_classes)  # ENetB0

    # Sử dụng model
    # model = CustomCNN(num_classes)
    # model = CustomCNNPlus(num_classes)

    # Chuyển mô hình sang device
    model = model.to(device)

    # Tạo một input giả (batch size 32, 3 kênh, kích thước 224x224)
    grid = utils.make_grid(images)  # Ghép ảnh thành lưới
    writer.add_image("Sample Images", grid)

    # Ghi mô hình vào TensorBoard
    writer.add_graph(model, images)

    # Xem mô hình
    summary(model, input_size=(1, 3, 224, 224))

    # Định nghĩa loss function và optimizer
    criterion = nn.CrossEntropyLoss()
    if hasattr(model, "fc"):
        optimizer = optim.Adam(
            model.fc.parameters(), lr=LEARNING_RATE
        )  # ResNet18, GoogleNet
    if hasattr(model, "classifier"):
        optimizer = optim.Adam(
            model.classifier[1].parameters(), lr=LEARNING_RATE
        )  # ENetB0

    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Custom

    # Learning Rate Scheduler (giảm LR khi loss không giảm)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1, verbose=True
    )

    # Train model
    model, history = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        num_epochs=NUM_EPOCHS,
        writer=writer,
    )

    # Lưu model cuối cùng
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "history": history,
        },
        FINAL_MODEL_PATH,
    )

    # Đóng writer
    writer.close()

    # Load model tốt nhất và đánh giá
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Đánh giá model
    y_true, y_pred = evaluate_model(model, test_dataloader, device, class_names)


if __name__ == "__main__":
    main()
