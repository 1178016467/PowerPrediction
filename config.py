import torch


class Config:
    # 数据参数
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    TARGET_COL = "Global_active_power"

    # 序列参数 - 使用滑动窗口
    HISTORY_LEN = 90  # 输入序列长度（天）
    SHORT_PRED_LEN = 90  # 短期预测长度（天）
    LONG_PRED_LEN = 365  # 长期预测长度（天）

    # 模型参数
    INPUT_SIZE = 13  # 特征数量（包括计算的剩余能耗）
    HIDDEN_SIZE = 200
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 4

    # 训练参数
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 150
    NUM_RUNS = 1
    DROPOUT = 0.2

    # 设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
