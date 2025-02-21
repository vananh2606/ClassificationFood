import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
import torch.nn as nn
import os
from tqdm import tqdm


def load_model(model_path, num_classes):
    """
    Load trained model from checkpoint
    """
    # Khởi tạo model
    model = models.resnet18(pretrained=True)

    # Thay đổi fully connected layer cuối cùng
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load trained weights
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Chuyển sang eval mode
    model.eval()

    return model


def preprocess_image(image_path):
    """
    Tiền xử lý ảnh để chuẩn bị cho prediction
    """
    # Định nghĩa transform
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Đọc và xử lý ảnh
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Lưu ảnh gốc để hiển thị
    original_image = image.copy()

    # Transform ảnh
    image = transform(image)
    image = image.unsqueeze(0)  # Thêm batch dimension

    return image, original_image


def predict_image(model, image_tensor, device, labels_map):
    """
    Dự đoán class cho ảnh
    """
    # Chuyển ảnh và model sang device
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    # Dự đoán
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_prob, predicted_idx = torch.max(probabilities, 1)

    # Lấy class name và probability
    predicted_label = labels_map[predicted_idx.item()]
    probability = predicted_prob.item()

    # Lấy top 3 predictions
    top3_prob, top3_idx = torch.topk(probabilities, 3, dim=1)
    top3_predictions = [
        (labels_map[idx.item()], prob.item())
        for idx, prob in zip(top3_idx[0], top3_prob[0])
    ]

    return predicted_label, probability, top3_predictions


def visualize_prediction(
    image, predicted_label, probability, top3_predictions, image_name=None
):
    """
    Hiển thị ảnh và kết quả dự đoán
    """
    plt.figure(figsize=(12, 5))

    # Hiển thị ảnh
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    title = f"Image: {image_name}\n" if image_name else ""
    title += f"Prediction: {predicted_label}\nProbability: {probability:.2%}"
    plt.title(title)
    plt.axis("off")

    # Hiển thị top 3 predictions dạng bar chart
    plt.subplot(1, 2, 2)
    labels = [pred[0] for pred in top3_predictions]
    probs = [pred[1] for pred in top3_predictions]

    bars = plt.bar(range(len(probs)), probs)
    plt.xticks(range(len(probs)), labels, rotation=45)
    plt.ylim(0, 1)
    plt.title("Top 3 Predictions")

    # Thêm giá trị probability lên mỗi bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2%}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()
    plt.close()


def predict_single_image(model, image_path, device, labels_map):
    """
    Dự đoán một ảnh và hiển thị kết quả
    """
    # Load ảnh
    image_tensor, original_image = preprocess_image(image_path)
    # Dự đoán
    predicted_label, probability, top3_predictions = predict_image(
        model, image_tensor, device, labels_map
    )
    # Hiển thị kết quả
    visualize_prediction(
        original_image, predicted_label, probability, top3_predictions, image_path
    )


def predict_folder(model, folder_path, device, labels_map):
    """
    Dự đoán từng ảnh trong folder và hiển thị kết quả lần lượt

    Args:
        model: Model đã train
        folder_path: Đường dẫn đến folder chứa ảnh
        device: Device để chạy model
        labels_map: Dictionary mapping giữa index và tên class
    """
    # Lấy danh sách các file ảnh
    image_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print("Không tìm thấy ảnh trong folder!")
        return

    print(f"\nTìm thấy {len(image_files)} ảnh trong folder.")
    print("Nhấn Enter để xem ảnh tiếp theo, nhấn 'q' để thoát.")

    # Dictionary để lưu thống kê predictions
    predictions_stats = {label: 0 for label in labels_map.values()}
    confidence_sum = 0

    # Dự đoán và hiển thị từng ảnh
    for idx, image_file in enumerate(image_files, 1):
        print(f"\nĐang xử lý ảnh {idx}/{len(image_files)}: {image_file}")

        image_path = os.path.join(folder_path, image_file)

        # Xử lý và dự đoán ảnh
        image_tensor, original_image = preprocess_image(image_path)
        predicted_label, probability, top3_predictions = predict_image(
            model, image_tensor, device, labels_map
        )

        # Cập nhật thống kê
        predictions_stats[predicted_label] += 1
        confidence_sum += probability

        # Hiển thị kết quả
        visualize_prediction(
            original_image, predicted_label, probability, top3_predictions, image_file
        )

        # Chờ user input
        if idx < len(image_files):
            user_input = input("Enter để tiếp tục, 'q' để thoát: ")
            if user_input.lower() == "q":
                print("\nĐã dừng prediction.")
                break

    # In thống kê cuối cùng
    print("\nThống kê predictions:")
    print("-" * 30)
    for label, count in predictions_stats.items():
        if count > 0:
            percentage = count / len(image_files) * 100
            print(f"{label}: {count} ảnh ({percentage:.1f}%)")

    avg_confidence = confidence_sum / len(image_files)
    print(f"\nĐộ tin cậy trung bình: {avg_confidence:.1%}")


def main():
    # Các thông số
    MODEL_PATH = "models/GGNet/best_model_restnet18.pth"  # hoặc 'final_model.pth'
    IMAGE_PATH = "pizza_steak/test/steak/4889.jpg"  # Đường dẫn đến ảnh cần dự đoán
    FOLDER_PATH = "10_food_classes_all_data/test/pizza"  # Đường dẫn đến folder chứa ảnh

    # Labels mapping
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

    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(MODEL_PATH, num_classes=len(labels_map))

    # Dự đoán 1 ảnh
    predict_single_image(model, IMAGE_PATH, device, labels_map)

    # Dự đoán cả folder
    # predict_folder(model, FOLDER_PATH, device, labels_map)


if __name__ == "__main__":
    main()
