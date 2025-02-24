import os
from torch.utils.data import Dataset
import cv2 as cv
import matplotlib.pyplot as plt

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


class FoodDataset(Dataset):
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
