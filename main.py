import os
import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader

from dataset import FoodDataset
from model import CustomCNN
from train_val import train_and_val
from evaluator import evaluate_model


def main():
    # Cấu hình
    TRAIN_PATH = "10_food_classes_all_data/train"
    TEST_PATH = "10_food_classes_all_data/test"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001

    # Map giữa index và tên class
    labels_map = {
        0: "chicken_curry",
        1: "chicken_wings",
        2: "fried_rice",
        3: "grilled_salmon",
        4: "hamburger",
        5: "ice_cream",
        6: "pizza",
        7: "ramen",
        8: "steak",
        9: "sushi",
    }

    # Xác định device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Định nghĩa các transforms
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.ToPILImage(),  # Thêm do input là numpy array từ OpenCV
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToPILImage(),  # Thêm do input là numpy array từ OpenCV
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Tạo dataset
    train_dataset = FoodDataset(TRAIN_PATH, transform=data_transforms["train"])
    test_dataset = FoodDataset(TEST_PATH, transform=data_transforms["test"])

    # Tạo dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}

    # Số lượng classes
    num_classes = len(labels_map)

    print(f"Classes mapping:")
    for idx, class_name in labels_map.items():
        print(f"  {idx}: {class_name}")
    print(f"\nNumber of training images: {dataset_sizes['train']}")
    print(f"Number of test images: {dataset_sizes['test']}")

    # Show images
    images, labels = next(iter(train_dataloader))
    plt.figure(figsize=(int(BATCH_SIZE / 2), int(BATCH_SIZE / 2)))
    for i in range(BATCH_SIZE):
        plt.subplot(4, int(BATCH_SIZE / 4), i + 1)
        plt.imshow(np.transpose(images[i], (1, 2, 0)))  # From (C, H, W) to (H, W, C)
        for idx, class_name in labels_map.items():
            if labels[i] == idx:
                plt.title(class_name)
        plt.axis("off")
    plt.show()

    # # Load mô hình ResNet18 pre-trained
    # model = models.resnet18(pretrained=True)
    # # Load mô hình GoogleNet pre-trained
    # model = models.googlenet(pretrained=True)
    # # Load mô hình EfficientNet pre-trained
    # model = models.efficientnet_b0(pretrained=True)

    # # Freeze các layers
    # for param in model.parameters():
    #     param.requires_grad = False

    ## Thay đổi fully connected layer cuối cùng
    # if hasattr(model, 'fc'):
    #     num_features = model.fc.in_features # ResNet18, GoogleNet
    #     model.fc = nn.Linear(num_features, num_classes) # ResNet18, GoogleNet
    # if hasattr(model, 'classifier'):
    #     num_features = model.classifier[1].in_features # ENetB0
    #     model.classifier[1] = nn.Linear(num_features, num_classes) # ENetB0

    # Sử dụng model customCNN
    model = CustomCNN(num_classes)

    # Chuyển mô hình sang device
    model = model.to(device)

    # Định nghĩa loss function và optimizer
    criterion = nn.CrossEntropyLoss()
    # if hasattr(model, 'fc'):
    #     optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE) # ResNet18, GoogleNet
    # if hasattr(model, 'classifier'):
    #     optimizer = optim.Adam(model.classifier[1].parameters(), lr=LEARNING_RATE) # ENetB0

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning Rate Scheduler (giảm LR khi loss không giảm)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1, verbose=True
    )

    # Train model
    model, history = train_and_val(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        num_epochs=NUM_EPOCHS,
        dataset_sizes=dataset_sizes,
        scheduler=scheduler,
    )

    # Lưu model cuối cùng
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        "models/ENet/final_model_enet.pth",
    )

    # Load model tốt nhất và đánh giá
    checkpoint = torch.load("models/ENet/best_model_enet.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Đánh giá model
    evaluate_model(model, test_dataloader, device)


if __name__ == "__main__":
    main()
