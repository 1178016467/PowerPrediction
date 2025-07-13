import datetime
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

from config import Config
from train import run_experiment
from tqdm import tqdm
import logging

import torch

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_prediction_comparison(model_name, targets, preds, scaler, dates=None, prediction_type='short', save_path=None):
    """绘制预测结果对比图"""
    plt.figure(figsize=(15, 8))

    # 确保输入是numpy数组
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    print(f"真实值形状: {targets.shape}, 预测值形状: {preds.shape}")

    # 反标准化目标值
    temp = np.zeros((len(targets[0]), Config.INPUT_SIZE))
    print(temp.shape)
    temp[:, 0] = targets[0]
    targets = scaler.inverse_transform(temp)[:, 0]

    # 反标准化预测值
    temp[:, 0] = preds[0]
    preds = scaler.inverse_transform(temp)[:, 0]

    print(f"真实值形状: {targets.shape}, 预测值形状: {preds.shape}")

    # 如果没有提供日期，创建序号
    if dates is None:
        all_dates = range(len(targets))
        xlabel = 'Time Steps'
    else:
        all_dates = dates[0]

        xlabel = 'Dates'
        # 检查长度是否匹配
        if len(all_dates) != targets.shape[0]:
            # 如果长度不匹配，只取第一个样本的日期序列
            print(f"警告: 日期序列长度({len(all_dates)})与数据长度({targets.shape[0]})不匹配，使用第一个样本的日期")
            all_dates = dates[0]
            print(f"日期序列长度: {len(all_dates)}")
            # 截断数据以匹配日期长度
            targets = targets[:len(all_dates)]
            preds = preds[:len(all_dates)]

    print(f"真实值形状: {targets.shape}, 预测值形状: {preds.shape}")

    # 绘制对比曲线
    plt.plot(all_dates, targets, label='ground_truth', color='blue', linewidth=2, alpha=0.8)
    plt.plot(all_dates, preds, label='prediction', color='red', linewidth=2, alpha=0.8)

    # 设置图表标题和标签
    title_text = 'short-term' if prediction_type == 'short' else 'long-term'
    plt.title(f'{model_name} - {title_text}Generation Power Prediction', fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Global_active_power (W)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 旋转x轴标签（如果是日期）
    if dates is not None:
        plt.xticks(rotation=45)

    # 计算误差统计
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))

    # 添加误差信息
    error_text = f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nType: {title_text}prediction'
    plt.text(0.02, 0.98, error_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")

    plt.show()

    return mse, mae


def visualize_predictions(model_name, trainer, test_loader, scaler, dates, prediction_type='short', save_path=None):
    """绘制完整测试集预测结果对比图"""
    plt.figure(figsize=(18, 10))

    # 获取预测结果
    preds, targets = trainer.predict(test_loader)
    print(f"原始预测结果形状: {preds.shape}, 真实值形状: {targets.shape}")
    print(f"样本日期形状: {dates.shape}")

    # 检查预测结果维度
    if preds.ndim == 1:
        # 如果预测结果是一维，假设每个样本只有一个预测值
        preds = preds.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    # 获取每个样本的预测长度
    pred_length = preds.shape[1] if preds.ndim > 1 else 1

    print(f"预测长度: {pred_length}")

    # 创建日期序列数组
    all_dates = []
    all_preds = []
    all_targets = []

    # 处理每个样本
    for i in range(len(preds)):
        # 获取当前样本的日期序列
        sample_date_seq = dates[i]

        # 获取当前样本的实际预测长度
        actual_length = min(len(sample_date_seq), pred_length)

        # 获取当前样本的预测值和真实值
        sample_preds = preds[i][:actual_length]
        sample_targets = targets[i][:actual_length]
        # 添加到总列表
        all_dates.extend(sample_date_seq[:actual_length])
        all_preds.extend(sample_preds)
        all_targets.extend(sample_targets)

    # 转换为数组
    all_dates = np.array(all_dates)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    print(f"总预测点数: {len(all_preds)}, 总真实点数: {len(all_targets)}")

    # 反标准化预测值和真实值
    # 创建一个与原始特征维度相同的临时数组
    temp_pred = np.zeros((len(all_preds), Config.INPUT_SIZE))
    temp_pred[:, 0] = all_preds
    preds_rescaled = scaler.inverse_transform(temp_pred)[:, 0]

    temp_true = np.zeros((len(all_targets), Config.INPUT_SIZE))
    temp_true[:, 0] = all_targets
    targets_rescaled = scaler.inverse_transform(temp_true)[:, 0]

    print(f"原始预测结果形状: {preds.shape}, 真实值形状: {targets.shape}")
    print(f"反标准化预测结果形状: {preds_rescaled.shape}, 反标准化真实值形状: {targets_rescaled.shape}")
    # 创建日期-值DataFrame用于排序
    df = pd.DataFrame({
        'date': all_dates,
        'pred': preds_rescaled,
        'true': targets_rescaled
    })

    # 确保日期是datetime类型
    if not isinstance(df['date'].iloc[0], datetime.date):
        df['date'] = pd.to_datetime(df['date'])

    # 按日期排序
    df.sort_values('date', inplace=True)

    # 创建日期索引
    dates = df['date'].values
    preds_rescaled = df['pred'].values
    targets_rescaled = df['true'].values

    # 设置日期格式化
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签

    # 绘制真实值曲线
    plt.plot(dates, targets_rescaled, label='ground_truth', color='blue', linewidth=2, alpha=0.8)

    # 绘制预测值散点图
    plt.scatter(dates, preds_rescaled, label='prediction', color='red', s=3, alpha=0.3)

    # 添加平均预测值线
    daily_avg = df.groupby('date')['pred'].mean()
    plt.plot(daily_avg.index, daily_avg.values, label='daily_avg',
             color='green', linewidth=2, linestyle='--')

    # 设置图表标题和标签
    title_text = 'short-term' if prediction_type == 'short' else 'long-term'
    plt.title(f'{model_name} - {title_text}Generation Power Prediction', fontsize=18, fontweight='bold')
    plt.xlabel('dates', fontsize=14)
    plt.ylabel('Global_active_power (W)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 设置Y轴限制为正值
    plt.ylim(bottom=0)

    # 计算误差统计
    mse = np.mean((preds_rescaled - targets_rescaled) ** 2)
    mae = np.mean(np.abs(preds_rescaled - targets_rescaled))

    # 添加误差信息
    error_text = f'MSE: {mse:.4f}\nMAE: {mae:.4f}\npred_length: {len(preds_rescaled)}'
    plt.text(0.02, 0.95, error_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 添加日期范围信息
    min_date = df['date'].min().strftime('%Y-%m-%d')
    max_date = df['date'].max().strftime('%Y-%m-%d')
    date_range = f"{min_date} to {max_date}"
    plt.text(0.02, 0.85, f'date range: {date_range}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"完整测试集对比图已保存到: {save_path}")

    plt.show()

    return mse, mae

def evaluate_multiple_runs(model_name, model_class, model_args, x_train, y_train, x_test, y_test, scaler, num_runs=5):
    """多次运行实验并统计结果"""
    mse_results, mae_results = [], []
    all_histories = []
    best_trainer = None
    save_dir = "results" + "/" + model_name
    log_file = 'results/results.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # 配置日志系统
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # 文件日志
            logging.StreamHandler()  # 控制台日志
        ]
    )
    logging.info("Model Name: {}".format(model_name))
    test_dataset_short = torch.utils.data.TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset_short,
        batch_size=Config.BATCH_SIZE
    )
    for run in tqdm(range(num_runs), desc="Experiment Runs"):
        start_time = time.time()
        mse, mae, history, trainer, total_params, trainable_params, memory_used = run_experiment(save_dir + "/" + str(run+1), model_class, model_args, x_train, y_train, x_test, y_test)
        # 获取预测结果
        preds, targets = trainer.predict(test_loader)
        all_preds = []
        all_targets = []

        # 获取每个样本的预测长度
        pred_length = preds.shape[1] if preds.ndim > 1 else 1
        # 处理每个样本
        for i in range(len(preds)):

            # 获取当前样本的实际预测长度
            actual_length = pred_length

            # 获取当前样本的预测值和真实值
            sample_preds = preds[i][:actual_length]
            sample_targets = targets[i][:actual_length]

            # 添加到总列表
            all_preds.extend(sample_preds)
            all_targets.extend(sample_targets)

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # 反标准化预测值和真实值
        # 创建一个与原始特征维度相同的临时数组
        temp_pred = np.zeros((len(all_preds), Config.INPUT_SIZE))
        temp_pred[:, 0] = all_preds
        preds_rescaled = scaler.inverse_transform(temp_pred)[:, 0]

        temp_true = np.zeros((len(all_targets), Config.INPUT_SIZE))
        temp_true[:, 0] = all_targets
        targets_rescaled = scaler.inverse_transform(temp_true)[:, 0]

        mse = np.mean((preds_rescaled - targets_rescaled) ** 2)
        mae = np.mean(np.abs(preds_rescaled - targets_rescaled))

        if run == 0 or mse < best_mse:
            best_mse = mse
            best_trainer = trainer
        mse_results.append(mse)
        mae_results.append(mae)
        all_histories.append(history)
        spend_time = time.time() - start_time
        print(f"Run {run + 1}: MSE={mse:.4f}, MAE={mae:.4f}, Time={spend_time:.2f}s")
        logging.info(
            "Run %02d | MSE=%.4f | MAE=%.4f | Time=%.2fs | Total_Params=%02dM | Trainable_Params=%02dM | Memory_Used=%.2fGB",
            run + 1,
            mse,
            mae,
            spend_time,
            total_params,
            trainable_params,
            memory_used
        )

    logging.info(
        "MSE_MEAN=%.4f | MSE_STD=%.4f | MAE_MEAN=%.4f | MAE_STD=%.4f",
        np.mean(mse_results),
        np.std(mse_results),
        np.mean(mae_results),
        np.std(mae_results)
    )
    return best_trainer, {
        'mse_mean': np.mean(mse_results),
        'mse_std': np.std(mse_results),
        'mae_mean': np.mean(mae_results),
        'mae_std': np.std(mae_results),
        'raw_results': list(zip(mse_results, mae_results)),
        'histories': all_histories
    }


def print_results(results, model_name, forecast_type):
    """打印格式化结果"""
    print(f"\n{model_name} - {forecast_type} Forecast Results:")
    print(f"MSE: Mean={results['mse_mean']:.4f}, Std={results['mse_std']:.4f}")
    print(f"MAE: Mean={results['mae_mean']:.4f}, Std={results['mae_std']:.4f}")
    print("Detailed Results:")
    for i, (mse, mae) in enumerate(results['raw_results']):
        print(f"  Run {i + 1}: MSE={mse:.4f}, MAE={mae:.4f}")


def plot_history(histories, model_name, forecast_type):
    """绘制训练历史"""
    plt.figure(figsize=(12, 6))

    # 平均训练损失
    avg_train_loss = np.mean([h['train_loss'] for h in histories], axis=0)
    plt.subplot(1, 2, 1)
    plt.plot(avg_train_loss, label='Train Loss')
    plt.title(f'{model_name} - {forecast_type}\nTraining Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    # 平均验证指标
    avg_val_mse = np.mean([h['val_mse'] for h in histories], axis=0)
    avg_val_mae = np.mean([h['val_mae'] for h in histories], axis=0)

    plt.subplot(1, 2, 2)
    plt.plot(avg_val_mse, label='Validation MSE')
    plt.plot(avg_val_mae, label='Validation MAE')
    plt.title(f'{model_name} - {forecast_type}\nValidation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"results/{model_name}_{forecast_type}_history.png")
    plt.close()
