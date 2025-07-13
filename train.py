import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from config import Config
import time
import os
import pandas as pd
from pynvml import *

class EarlyStopping:
    """早停类，当验证损失停止下降时停止训练"""
    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoint.pt'):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def get_gpu_memory():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used/1024**3  # 返回GB单位

class ModelTrainer:
    def __init__(self, model, model_name, steps_per_epoch, device=Config.DEVICE):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LR,
            weight_decay=1e-5  # 权重衰减防止过拟合
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LR * 10,  # 最大学习率
            steps_per_epoch=steps_per_epoch,  # 每个epoch的步数
            epochs=Config.EPOCHS,  # 总epoch数
            anneal_strategy='cos'  # 余弦退火策略
        )

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        self.accumulation_steps = 4  # 每4步更新一次权重

        # 初始化训练记录
        self.history = {
            'train_loss': [],
            'val_mse': [],
            'val_mae': [],
            'epoch_time': [],
            'lr': []
        }
        self.best_mse = float('inf')
        self.best_mae = float('inf')
        self.best_loss = float('inf')
        self.best_model = None
        self.save_dir = model_name
        if device == 'cuda':
            self.start_mem = get_gpu_memory()
        else:
            self.start_mem = 0
        self.max_mem = 0
        os.makedirs(self.save_dir, exist_ok=True)

        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=15,
            verbose=True,
            path=os.path.join(self.save_dir, 'best_model.pth')
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        if self.device == 'cuda':
            self.start_mem = get_gpu_memory()

        step = 0
        batch_idx = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            with torch.cuda.amp.autocast():

                preds = self.model(batch_X)
                loss = self.mse_loss(preds, batch_y)

                mae_loss = self.mae_loss(preds, batch_y)
                combined_loss = loss + 0.1 * mae_loss

            # 梯度缩放和反向传播
            scaled_loss = combined_loss / self.accumulation_steps
            scaled_loss.backward()

            # 梯度累积
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # 更新参数
                self.optimizer.step()
                self.optimizer.zero_grad()

                # 更新学习率
                self.scheduler.step()

            total_loss += combined_loss.item()
            batch_idx += 1
            step += 1

        return total_loss / len(train_loader)

    def evaluate(self, test_loader, return_predictions=False):
        self.model.eval()
        total_mse, total_mae = 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                preds = self.model(batch_X)

                total_mse += self.mse_loss(preds, batch_y).item()
                total_mae += self.mae_loss(preds, batch_y).item()

                # 保存预测结果用于后续分析
                if return_predictions:
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(batch_y.cpu().numpy())

        mse = total_mse / len(test_loader)
        mae = total_mae / len(test_loader)
        if return_predictions:
            return mse, mae, np.vstack(all_preds), np.vstack(all_targets)
        return mse, mae

    def train(self, train_loader, test_loader, epochs=Config.EPOCHS):
        epoch_bar = tqdm(
            range(epochs),
            desc="Training",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            ncols=100
        )
        start_time = time.time()

        for epoch in epoch_bar:
            epoch_start = time.time()

            # 训练
            train_loss = self.train_epoch(train_loader)

            # 验证
            val_mse, val_mae = self.evaluate(test_loader)

            # 记录指标
            self.history['train_loss'].append(train_loss)
            self.history['val_mse'].append(val_mse)
            self.history['val_mae'].append(val_mae)
            self.history['epoch_time'].append(time.time() - epoch_start)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # 检查早停
            self.early_stopping(train_loss, self.model)
            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # 保存最佳模型（基于验证MSE）
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.best_mse = val_mse
                self.best_mae = val_mae
                self.best_model = self.model.state_dict()
                torch.save(self.best_model,
                           os.path.join(self.save_dir, "best_model.pth"))

            end_mem = get_gpu_memory()

            # 每10个epoch保存checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)

            # 更新进度条信息
            epoch_bar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val MSE': f'{val_mse:.4f}',
                'Val MAE': f'{val_mae:.4f}'
            })
            if end_mem - self.start_mem > self.max_mem:
                self.max_mem = end_mem - self.start_mem

        # 保存最终结果
        self._save_training_log()
        spend_time = time.time() - start_time
        print(f"\nTraining completed in {spend_time:.2f}s")
        self.model.load_state_dict(self.best_model)

        return self.history

    def predict(self, test_loader):
        """生成预测结果"""
        _, _, preds, targets = self.evaluate(test_loader, return_predictions=True)
        return preds, targets

    def _save_checkpoint(self, epoch):
        """保存训练状态"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'history': self.history,
            'best_mse': self.best_mse,
            'best_mae': self.best_mae
        }
        torch.save(checkpoint,
                   os.path.join(self.save_dir, f'checkpoint_epoch{epoch}.pth'))

    def _save_training_log(self):
        """保存训练日志"""
        pd.DataFrame(self.history).to_csv(
            os.path.join(self.save_dir, 'training_log.csv'),
            index=False
        )

def run_experiment(model_name, model_class, model_args, X_train, y_train, X_test, y_test):
    """运行完整实验流程"""
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    # 优化数据加载
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=min(4, os.cpu_count())  # 多进程加载
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    # 初始化模型和训练器
    model = model_class(**model_args)
    steps_per_epoch = len(train_loader)
    trainer = ModelTrainer(model, model_name, steps_per_epoch)
    # 训练和评估
    history = trainer.train(train_loader, test_loader)

    final_mse, final_mae = trainer.evaluate(test_loader)

    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    memory_used = trainer.max_mem

    return final_mse, final_mae, history, trainer, total_params, trainable_params, memory_used