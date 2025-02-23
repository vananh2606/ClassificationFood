import numpy as np
import torch

class EarlyStopping:
    """
    Early stopping để dừng training khi model không cải thiện
    
    Args:
        patience (int): Số epochs chờ trước khi dừng training
        min_delta (float): Giá trị tối thiểu để coi là có cải thiện
        verbose (bool): In thông báo nếu True
    """
    def __init__(self, patience=5, min_delta=0, verbose=True):
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
                print(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('\nEarly stopping triggered')
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer, epoch, val_loss, history)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, model, optimizer, epoch, loss, history):
        """Lưu model checkpoint tốt nhất"""
        self.best_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'history': history
        }
    
    def get_best_state(self):
        """Trả về trạng thái tốt nhất của model"""
        return self.best_state