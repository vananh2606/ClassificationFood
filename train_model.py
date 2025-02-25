import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class EarlyStopping:
    """
    Early stopping để dừng huấn luyện khi hiệu suất của mô hình trên tập validation không cải thiện nữa.

    Args:
        patience (int): Số lượng epochs để chờ đợi trước khi dừng huấn luyện nếu không có cải thiện. Mặc định là 5.
        min_delta (float): Giá trị cải thiện tối thiểu để coi là có cải thiện. Mặc định là 0.
        verbose (bool): Cho phép in thông báo nếu True. Mặc định là True.

    Attributes:
        patience (int): Số lượng epochs để chờ đợi.
        min_delta (float): Giá trị cải thiện tối thiểu.
        verbose (bool): Cho phép in thông báo.
        counter (int): Đếm số epochs mà không có cải thiện.
        best_loss (float): Giá trị loss tốt nhất trên tập validation.
        early_stop (bool): Cờ báo hiệu early stopping.
        best_state (dict): Trạng thái tốt nhất của mô hình (bao gồm trọng số, optimizer, v.v.).

    Methods:
        __call__(val_loss, model, optimizer, epoch, history): Kiểm tra xem có nên early stopping hay không dựa trên val_loss hiện tại.
        save_checkpoint(model, optimizer, epoch, loss, history): Lưu checkpoint của mô hình nếu có cải thiện.
        get_best_state(): Trả về trạng thái tốt nhất của mô hình.
    """

    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model, optimizer, epoch, history):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer, epoch, val_loss, history)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"\nEarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("\nEarly stopping triggered")
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer, epoch, val_loss, history)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, model, optimizer, epoch, loss, history):
        """Lưu model checkpoint tốt nhất"""
        self.best_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "history": history,
        }

    def get_best_state(self):
        """Trả về trạng thái tốt nhất của model"""
        return self.best_state


def train_model(
    model,
    criterion,
    optimizer,
    train_dataloader,
    val_dataloader,
    device,
    num_epochs=10,
    scheduler=None,
):
    """Huấn luyện mô hình PyTorch.

    Hàm này huấn luyện mô hình PyTorch đã cho bằng cách sử dụng các tham số đầu vào được cung cấp.
    Nó thực hiện huấn luyện trên nhiều epoch, theo dõi loss và accuracy của training và validation,
    và lưu model có hiệu suất tốt nhất.

    Args:
        model (nn.Module): Mô hình PyTorch cần được huấn luyện.
        criterion (callable): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        train_dataloader (DataLoader): DataLoader cho tập huấn luyện.
        val_dataloader (DataLoader): DataLoader cho tập validation.
        device (torch.device): Thiết bị để huấn luyện mô hình (ví dụ: 'cuda' hoặc 'cpu').
        num_epochs (int, optional): Số epoch để huấn luyện. Mặc định là 10.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Mặc định là None.

    Returns:
        tuple: Mô hình đã huấn luyện và history training.
    """
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4, verbose=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}\n" + "-" * 10)

        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_dataloader, device
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        val_loss, val_acc = val_one_epoch(model, criterion, val_dataloader, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if early_stopping(val_loss, model, optimizer, epoch, history):
            print("\nRestoring best model weights...")
            best_state = early_stopping.get_best_state()
            model.load_state_dict(best_state["model_state_dict"])
            print(f"Best model was from epoch {best_state['epoch']+1}")
            plot_training_history(history)
            return model, history

        if val_acc > best_acc and val_loss < best_loss:
            best_acc = val_acc
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler else None
                    ),
                    "loss": val_loss,
                    "acc": val_acc,
                },
                "models/CustomModel/best_model_cm+.pth",
            )

        if scheduler is not None:
            scheduler.step(val_loss)

    print(f"Best val Acc: {best_acc:.4f}")
    print(f"Best val Loss: {best_loss:.4f}")
    plot_training_history(history)
    return model, history


def train_one_epoch(model, criterion, optimizer, train_dataloader, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    pbar = tqdm(train_dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        running_corrects += torch.sum(predicted == labels.data)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_train_loss = running_loss / len(train_dataloader.dataset)
    epoch_train_acc = running_corrects.double() / len(train_dataloader.dataset)
    return epoch_train_loss, epoch_train_acc.item()


def val_one_epoch(model, criterion, val_dataloader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    pbar = tqdm(val_dataloader, desc="Validation")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            running_corrects += torch.sum(predicted == labels.data)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_val_loss = running_loss / len(val_dataloader.dataset)
        epoch_val_acc = running_corrects.double() / len(val_dataloader.dataset)
        return epoch_val_loss, epoch_val_acc.item()


def plot_training_history(history):
    """Vẽ đồ thị lịch sử huấn luyện bao gồm loss và accuracy.

    Hàm này nhận vào một dictionary `history` chứa lịch sử huấn luyện
    và vẽ đồ thị loss và accuracy của training và validation theo epoch.

    Args:
        history (dict): Một dictionary chứa lịch sử huấn luyện, với các key:
            - "train_loss": Danh sách các giá trị loss của training theo epoch.
            - "train_acc": Danh sách các giá trị accuracy của training theo epoch.
            - "val_loss": Danh sách các giá trị loss của validation theo epoch.
            - "val_acc": Danh sách các giá trị accuracy của validation theo epoch.

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history["train_loss"], label="train loss")
    ax1.plot(history["val_loss"], label="val loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history["train_acc"], label="train acc")
    ax2.plot(history["val_acc"], label="val acc")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("models/CustomModel/training_history_cm+.png")
    plt.show()
