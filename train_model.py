import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

BEST_MODEL_PATH = "models/CustomModel/Pro/best_model_pro.pth"
TRAINING_HISTORY_PATH = "models/CustomModel/Pro/training_history_pro.png"


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
        __call__(val_loss, val_acc, model, optimizer, epoch, history): Kiểm tra xem có nên early stopping hay không dựa trên val_loss hiện tại.
        save_checkpoint(model, optimizer, epoch, val_loss, val_acc, history): Lưu checkpoint của mô hình nếu có cải thiện.
        get_best_state(): Trả về trạng thái tốt nhất của mô hình.
    """

    def __init__(self, patience=5, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, val_acc, model, optimizer, scheduler, epoch, history):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, val_acc, history
            )
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(
                    f"\nBộ đếm Early Stopping: {self.counter} trên tổng số {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("\nEarly Stopping được kích hoạt")
        else:
            self.best_loss = val_loss
            self.save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, val_acc, history
            )
            self.counter = 0

        return self.early_stop

    def save_checkpoint(
        self, model, optimizer, scheduler, epoch, val_loss, val_acc, history
    ):
        """Lưu model checkpoint tốt nhất"""
        self.best_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "history": history,
        }

    def get_best_state(self):
        """Trả về trạng thái tốt nhất của model"""
        return self.best_state


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloader,
    device,
    num_epochs=10,
    writer=None,
):
    """Huấn luyện mô hình PyTorch.

    Hàm này huấn luyện mô hình PyTorch đã cho bằng cách sử dụng các tham số đầu vào được cung cấp.
    Nó thực hiện huấn luyện trên nhiều epoch, theo dõi loss và accuracy của training và validation,
    và lưu model có hiệu suất tốt nhất.

    Args:
        model (nn.Module): Mô hình PyTorch cần được huấn luyện.
        criterion (callable): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        train_dataloader (DataLoader): DataLoader cho tập huấn luyện.
        val_dataloader (DataLoader): DataLoader cho tập validation.
        device (torch.device): Thiết bị để huấn luyện mô hình (ví dụ: 'cuda' hoặc 'cpu').
        num_epochs (int, optional): Số epoch để huấn luyện. Mặc định là 10.

    Returns:
        tuple: Mô hình đã huấn luyện và history training.
    """
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4, verbose=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    since = time.time()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}\n" + "-" * 10)

        # Phase training
        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_dataloader, device
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        # Phase validation
        val_loss, val_acc = val_one_epoch(model, criterion, val_dataloader, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if writer is not None:
            # Ghi histogram cơ sở mô hình với TensorBoard
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch)

            # Ghi loss và accuracy vào TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

        # Cập nhật learning rate scheduler nếu có
        if scheduler is not None:
            scheduler.step(val_loss)

        if early_stopping(
            val_loss, val_acc, model, optimizer, scheduler, epoch, history
        ):
            print("\nĐang khôi phục trọng số mô hình tốt nhất...")
            best_state = early_stopping.get_best_state()
            model.load_state_dict(best_state["model_state_dict"])
            print(f"Mô hình tốt nhất từ epoch {best_state['epoch']+1}")
            plot_training_history(history)
            return model, history

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "history": history,
                },
                BEST_MODEL_PATH,
            )

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")
    plot_training_history(history)

    return model, history


def train_one_epoch(model, criterion, optimizer, train_dataloader, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    pbar = tqdm(train_dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Xóa gradient trước đó
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)  # Tính toán output
        loss = criterion(outputs, labels)  # Tính toán loss

        # Backward pass
        loss.backward()  # Tính toán gradient
        optimizer.step()  # Cập nhật trọng số

        # Tính toán loss trung bình trên batch
        running_loss += loss.item() * images.size(0)  # image.size(0) = batch_size

        _, predicted = torch.max(outputs, dim=1)
        # Tính toán số lượng dự đoán đúng trên batch
        running_corrects += torch.sum(predicted == labels.data)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    train_dataset_size = len(train_dataloader.dataset)
    # Tính toán loss trung bình trên epoch
    epoch_train_loss = running_loss / train_dataset_size
    # Tính toán accuracy trung bình trên epoch
    epoch_train_acc = running_corrects.double() / train_dataset_size
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

            # Tính toán loss trung bình trên batch
            running_loss += loss.item() * images.size(0)  # image.size(0) = batch_size

            _, predicted = torch.max(outputs, dim=1)
            # Tính toán số lượng dự đoán đúng trên batch
            running_corrects += torch.sum(predicted == labels.data)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_dataset_size = len(val_dataloader.dataset)
        # Tính toán loss trung bình trên epoch
        epoch_val_loss = running_loss / val_dataset_size
        # Tính toán accuracy trung bình trên epoch
        epoch_val_acc = running_corrects.double() / val_dataset_size
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
    plt.savefig(TRAINING_HISTORY_PATH)
    plt.show()
