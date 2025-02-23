import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from early_stopping import EarlyStopping

def train_model(
    model,
    criterion,
    optimizer,
    train_dataloader,
    test_dataloader,
    device,
    num_epochs=10,
    dataset_sizes=None,
    scheduler=None,
):
    """
    Train a PyTorch model for food classification.

    Args:
        model: PyTorch model
        criterion: Loss function
        optimizer: Optimizer
        train_dataloader: Training data loader
        test_dataloader: Testing data loader
        device: Device to train on (cuda/cpu)
        num_epochs: Number of epochs to train
        dataset_sizes: Dictionary containing sizes of train and test datasets

    Returns:
        model: Trained model
        history: Training history
    """
    # Khởi tạo EarlyStopping
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4, verbose=True)
    
    # Khởi tạo dictionary để lưu history
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Best model tracking
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Mỗi epoch có phase train và test
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = test_dataloader

            running_loss = 0.0
            running_corrects = 0

            # Sử dụng tqdm để hiển thị progress bar
            pbar = tqdm(dataloader, desc=f"{phase} Phase")

            # Iterate over data
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Save history
            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                history["test_loss"].append(epoch_loss)
                history["test_acc"].append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Early Stopping và save best model
            if phase == "test":
                # Check Early Stopping
                if early_stopping(epoch_loss, model, optimizer, epoch, history):
                    print("\nRestoring best model weights...")
                    best_state = early_stopping.get_best_state()
                    model.load_state_dict(best_state['model_state_dict'])
                    print(f"Best model was from epoch {best_state['epoch']+1}")
                    # Plot training history
                    plot_training_history(history)
                    return model, history

                # Lưu best model dựa trên accuracy
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": epoch_loss,
                            "acc": epoch_acc,
                        },
                        "models/ENet/best_model_enet.pth",
                    )

                # Gọi ReduceLROnPlateau nếu cần để giảm Learning Rate nếu test loss không giảm
                if scheduler is not None:
                    scheduler.step(epoch_loss)

        print()

    print(f"Best test Acc: {best_acc:4f}")

    # Plot training history
    plot_training_history(history)

    return model, history


def plot_training_history(history):
    """
    Plot training and validation loss/accuracy curves

    Args:
        history: Dictionary containing training history
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["test_loss"], label="test")
    ax1.set_title("Model Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Plot accuracy
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["test_acc"], label="test")
    ax2.set_title("Model Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("models/Enet/training_history_enet.png")
    plt.show()
