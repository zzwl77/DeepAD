import torch
import numpy as np
import matplotlib.pyplot as plt

class Warmup_ExpDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_lr, peak_lr, final_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        super(Warmup_ExpDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = (self.peak_lr - self.warmup_lr) * self.last_epoch / self.warmup_epochs + self.warmup_lr
        else:
            #Exponential decay
            decay_rate = (self.final_lr / self.peak_lr) ** (1 / (self.total_epochs - self.warmup_epochs))
            lr = self.peak_lr * (decay_rate ** (self.last_epoch - self.warmup_epochs))
            
        return [lr for _ in self.optimizer.param_groups]