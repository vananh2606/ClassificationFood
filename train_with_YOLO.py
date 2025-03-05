import os
import shutil
import torch
from ultralytics import YOLO

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

# Class names
class_names = list(labels_map.values())


def main():
    # Cấu hình
    TRAIN_PATH = "10_food_classes_all_data/train"
    VAL_PATH = "10_food_classes_all_data/val"
    TEST_PATH = "10_food_classes_all_data/test"
    YOLO_PATH = "10_food_classes_all_data"
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001

    # Tải mô hình YOLOv8 cho phân loại
    model = YOLO("models/YOLO/yolov8n-cls.pt")  # Nanoversion

    # Huấn luyện mô hình YOLO
    results = model.train(
        data=YOLO_PATH,
        epochs=NUM_EPOCHS,
        imgsz=640,
        batch=BATCH_SIZE,
        device=0 if torch.cuda.is_available() else "cpu",
        project="runs/YOLOv8model",
        name="yolov8_classification",
        patience=5,
        save=True,
        lr0=LEARNING_RATE,
    )

    # Đánh giá mô hình
    model = YOLO("runs/YOLOv8model/yolov8_classification/weights/best.pt")
    results = model.val(data=YOLO_PATH)


if __name__ == "__main__":
    main()
