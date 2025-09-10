# utils/schedule.py
import math

class WarmupCosine:
    def __init__(self, optimizer, base_lr, warmup_steps, total_steps):
        self.opt = optimizer
        self.base_lr = base_lr
        self.warmup = warmup_steps
        self.total = total_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup:
            lr = self.base_lr * self.step_num / max(1, self.warmup)
        else:
            t = (self.step_num - self.warmup) / max(1, self.total - self.warmup)
            lr = 0.5 * self.base_lr * (1 + math.cos(math.pi * t))
        for pg in self.opt.param_groups:
            pg['lr'] = lr
        return lr
