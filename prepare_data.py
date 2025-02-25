import os
import shutil
import random


def merge_dataset(dataset_dir, train_dir, test_dir):
    """
    Gộp train và test vào dataset

    Args:
        dataset_dir (str): Đường dẫn đến thư mục dataset
        train_dir (str): Đường dẫn đến thư mục train
        test_dir (str): Đường dẫn đến thư mục test
    """
    # Tạo thư mục dataset nếu chưa có
    os.makedirs(dataset_dir, exist_ok=True)

    # Gộp train và test vào dataset
    for folder in os.listdir(train_dir):
        train_folder = os.path.join(train_dir, folder)
        test_folder = os.path.join(test_dir, folder)
        target_folder = os.path.join(dataset_dir, folder)

        os.makedirs(target_folder, exist_ok=True)

        # Di chuyển ảnh từ train
        for file in os.listdir(train_folder):
            shutil.move(
                os.path.join(train_folder, file), os.path.join(target_folder, file)
            )

        # Di chuyển ảnh từ test
        for file in os.listdir(test_folder):
            shutil.move(
                os.path.join(test_folder, file), os.path.join(target_folder, file)
            )

    print("Gộp train và test vào dataset hoàn tất!")


def split_dataset(
    dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
):
    assert train_ratio + val_ratio + test_ratio == 1, "Tổng tỷ lệ phải bằng 1!"
    """
    Chia tập dữ liệu thành train, val, test

    Args:
        dataset_dir (str): Đường dẫn đến thư mục dataset
        output_dir (str): Đường dẫn đến thư mục output
    """

    # Tạo thư mục output
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    for folder in [train_dir, val_dir, test_dir]:
        os.makedirs(folder, exist_ok=True)

    # Duyệt từng class trong dataset
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Tạo thư mục cho từng class trong train, val, test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Lấy danh sách file và xáo trộn ngẫu nhiên
        files = os.listdir(class_path)
        random.shuffle(files)

        # Chia dữ liệu theo tỷ lệ
        num_train = int(len(files) * train_ratio)
        num_val = int(len(files) * val_ratio)

        train_files = files[:num_train]
        val_files = files[num_train : num_train + num_val]
        test_files = files[num_train + num_val :]

        # Di chuyển file vào thư mục tương ứng
        for f in train_files:
            shutil.move(
                os.path.join(class_path, f), os.path.join(train_dir, class_name, f)
            )
        for f in val_files:
            shutil.move(
                os.path.join(class_path, f), os.path.join(val_dir, class_name, f)
            )
        for f in test_files:
            shutil.move(
                os.path.join(class_path, f), os.path.join(test_dir, class_name, f)
            )

    print("Chia tập dữ liệu hoàn tất!")


merge_dataset(
    "10_food_classes_all_data/dataset",
    "10_food_classes_all_data/train",
    "10_food_classes_all_data/test",
)

# Gọi hàm với tỷ lệ tùy chỉnh
split_dataset(
    "10_food_classes_all_data/dataset",
    "10_food_classes_all_data",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
)
