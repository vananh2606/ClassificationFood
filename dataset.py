import os
from torch.utils.data import Dataset
import cv2 as cv

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
        image = cv.imread(self.images[idx])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
