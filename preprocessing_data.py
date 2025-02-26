import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

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


class FoodDataset(Dataset):
    """
    Tạo một dataset tùy chỉnh cho dữ liệu hình ảnh thức ăn.

    Dataset này kế thừa từ lớp `torch.utils.data.Dataset` và được sử dụng để tải và tiền xử lý dữ liệu hình ảnh
    cho các tác vụ học máy. Nó đọc hình ảnh từ một thư mục gốc, áp dụng các phép biến đổi (nếu được cung cấp)
    và trả về hình ảnh và nhãn tương ứng của nó.

    Args:
        root_dir (str): Đường dẫn đến thư mục gốc chứa các thư mục con của từng lớp.
        transform (callable, optional): Các phép biến đổi được áp dụng cho hình ảnh. Mặc định là None.

    Attributes:
        root_dir (str): Đường dẫn đến thư mục gốc.
        transform (callable, optional): Các phép biến đổi.
        images (list): Danh sách các đường dẫn đến hình ảnh.
        labels (list): Danh sách các nhãn tương ứng với hình ảnh.

    Methods:
        __len__(): Trả về số lượng hình ảnh trong dataset.
        __getitem__(index): Trả về hình ảnh và nhãn tại chỉ mục `index`.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label in os.listdir(root_dir):
            folder = os.path.join(root_dir, str(label))
            for image in os.listdir(folder):
                self.images.append(os.path.join(folder, image))
                for key, value in labels_map.items():
                    if value == label:
                        self.labels.append(key)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # print(
        #     f"Index requested: {idx}, Total images: {len(self.images)}, Total labels: {len(self.labels)}"
        # )
        image = cv.imread(self.images[idx])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transform(image_size=(224, 224)):
    """
    Tạo ra các phép biến đổi (transform) cho dữ liệu hình ảnh.

    Hàm này tạo ra hai bộ phép biến đổi: một cho tập huấn luyện (train)
    và một cho tập validation và kiểm tra (val_test). Các phép biến đổi này
    bao gồm thay đổi kích thước, lật ngẫu nhiên theo chiều ngang, xoay ngẫu nhiên,
    thay đổi màu sắc (chỉ cho tập huấn luyện) và chuẩn hóa.

    Args:
        image_size (tuple, optional): Kích thước hình ảnh mong muốn. Mặc định là (224, 224).

    Returns:
        tuple: Hai bộ phép biến đổi: `train_transform` và `val_test_transform`.
    """
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.ToPILImage(),  # Thêm do input là numpy array từ OpenCV
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "val_test": transforms.Compose(
            [
                transforms.ToPILImage(),  # Thêm do input là numpy array từ OpenCV
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }

    return data_transforms["train"], data_transforms["val_test"]


def visualie_dataloader(size, images, labels):
    """
    Hiển thị hình ảnh từ dataloader.

    Args:
        size (int): Số lượng hình ảnh muốn hiển thị.
        images (torch.Tensor): Tensor chứa các hình ảnh.
        labels (torch.Tensor): Tensor chứa nhãn của các hình ảnh.
    """
    plt.figure(figsize=(size // 2, size // 2))
    for i in range(size):
        plt.subplot(4, size // 4, i + 1)
        plt.imshow(np.transpose(np.clip(images[i], 0, 1), (1, 2, 0)))  # (C, H, W) -> (H, W, C)
        plt.title(labels_map.get(labels[i].item(), "Unknown"))
        plt.axis("off")

    plt.show()


def main():
    PATH_FOLDER = "10_food_classes_all_data/test"
    food_dataset = FoodDataset(PATH_FOLDER)

    # Visualize image
    image, label = food_dataset[0]
    plt.imshow(image)
    plt.title(labels_map[label])
    plt.show()


if __name__ == "__main__":
    main()
