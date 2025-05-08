import glob
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Labels
class_names = [
    "chicken_curry",
    "chicken_wings",
    "fried_rice",
    "grilled_salmon",
    "hamburger",
    "ice_cream",
    "pizza",
    "ramen",
    "steak",
    "sushi",
]


class CustomDataset(Dataset):
    """
    Tạo một dataset tùy chỉnh cho dữ liệu hình ảnh thức ăn.

    Dataset này kế thừa từ lớp `torch.utils.data.Dataset` và được sử dụng để tải và tiền xử lý dữ liệu hình ảnh
    cho các tác vụ học máy. Nó đọc hình ảnh từ một thư mục gốc, áp dụng các phép biến đổi (nếu được cung cấp)
    và trả về hình ảnh và nhãn tương ứng của nó.

    Args:
        root_dir (str): Đường dẫn đến thư mục gốc chứa các thư mục con của từng lớp.
        transform (callable, optional): Các phép biến đổi được áp dụng cho hình ảnh. Mặc định là None.
        max_samples (int, optional): Số lượng mẫu tối đa để đọc từ thư mục gốc. Mặc định là None (không giới hạn).
        shuffle (bool, optional): Nếu True, sẽ trộn dữ liệu ngẫu nhiên. Mặc định là False.

    Attributes:
        root_dir (str): Đường dẫn đến thư mục gốc.
        transform (callable, optional): Các phép biến đổi.
        images (list): Danh sách các đường dẫn đến hình ảnh.
        labels (list): Danh sách các nhãn tương ứng với hình ảnh.

    Methods:
        __len__(): Trả về số lượng hình ảnh trong dataset.
        __getitem__(index): Trả về hình ảnh và nhãn tại chỉ mục `index`.
    """

    def __init__(self, root_dir, transform=None, max_samples=None, shuffle=False):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        self.class_names = class_names
        for idx, label in enumerate(self.class_names):
            folder_path = os.path.join(root_dir, label)
            print(f"Đang xử lý thư mục: {folder_path}")
            if not os.path.isdir(folder_path):  # Bỏ qua nếu không phải thư mục
                continue

            image_paths = glob.glob(os.path.join(folder_path, "*.*"))  # Lấy tất cả ảnh
            print(f"Số ảnh trong thư mục {label}: {len(image_paths)}")
            if shuffle:
                np.random.shuffle(image_paths)  # Shuffle trước khi chọn ảnh
            
            if max_samples is not None:
                image_paths = image_paths[:max_samples]

            for image_path in image_paths:
                self.images.append(image_path)
                self.labels.append(idx)

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
    images = images.numpy()

    plt.figure(figsize=(size // 2, size // 2))
    for i in range(size):
        plt.subplot(4, size // 4, i + 1)
        plt.imshow(
            np.transpose(np.clip(images[i], 0, 1), (1, 2, 0))
        )  # (C, H, W) -> (H, W, C)
        plt.title(class_names[labels[i]])
        plt.axis("off")

    plt.show()


def test_image():
    PATH_FOLDER = "10_food_classes_all_data\\test"
    food_dataset = CustomDataset(PATH_FOLDER)

    # Visualize image
    image, label = food_dataset[0]
    plt.imshow(image)
    plt.title(class_names[label])
    plt.show()

def test_batch_image():
    PATH_FOLDER = "10_food_classes_all_data\\test"
    _, test_transform = get_transform()
    food_dataset = CustomDataset(PATH_FOLDER, transform=test_transform)
    test_dataloader = DataLoader(
        food_dataset, batch_size=32, shuffle=True
    )
    # Visualize image
    images, labels = next(iter(test_dataloader))
    visualie_dataloader(size=32, images=images, labels=labels)
    print(images.shape)
    print(labels.shape)


if __name__ == "__main__":
    test_batch_image()
