import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CONFUSION_MATRIX_PATH = "models/KMS/confusion_matrix_efficient.png"
EVALUATION_REPORT_PATH = "models/KMS/model_evaluation_efficient.txt"

def evaluate_model(model, test_loader, device, class_names):
    """Đánh giá mô hình trên tập dữ liệu kiểm tra.

    Hàm này đánh giá mô hình đã cho trên tập dữ liệu kiểm tra bằng cách tính toán
    độ chính xác, báo cáo phân loại và ma trận nhầm lẫn. Nó cũng vẽ ma trận
    nhầm lẫn để trực quan hóa hiệu suất của mô hình.

    Args:
        model (nn.Module): Mô hình PyTorch cần được đánh giá.
        test_loader (DataLoader): DataLoader cho tập dữ liệu kiểm tra.
        device (torch.device): Thiết bị để đánh giá mô hình (ví dụ: 'cuda' hoặc 'cpu').
        class_names (list): Danh sách tên các lớp.

    Returns:
        tuple: Nhãn thực tế (y_true) và nhãn dự đoán (y_pred).
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"Accuracy Score: {acc:.2f}")

    class_report = classification_report(
        all_labels, all_preds, target_names=class_names
    )
    print("\nClassification Report:")
    print(class_report)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    with open(EVALUATION_REPORT_PATH, "w") as f:
        f.write(f"Accuracy Score: {acc:.2f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report + "\n\n")
        f.write("Confusion Matrix:\n")
        np.savetxt(f, conf_matrix, fmt="%d")

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names)

    return all_labels, all_preds


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Vẽ ma trận nhầm lẫn.

    Hàm này tạo và vẽ ma trận nhầm lẫn từ các nhãn thực tế và nhãn dự đoán.
    Nó sử dụng seaborn để tạo heatmap trực quan hóa ma trận.

    Args:
        y_true (array-like): Các nhãn thực tế.
        y_pred (array-like): Các nhãn dự đoán.
        class_names (list): Danh sách tên các lớp.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.show()
