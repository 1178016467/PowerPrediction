import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import Config
import warnings

warnings.filterwarnings('ignore')

# 定义列名
COLUMN_NAMES = [
    'DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
    'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
]


def load_and_preprocess():
    """加载并预处理原始数据，处理分钟级数据和缺失值"""
    # 加载训练数据
    train_df = pd.read_csv(Config.TRAIN_PATH, names=COLUMN_NAMES, header=0, parse_dates=['DateTime'])

    # 加载测试数据
    test_df = pd.read_csv(Config.TEST_PATH, names=COLUMN_NAMES, header=None, parse_dates=['DateTime'])

    # 合并以便统一处理
    full_df = pd.concat([train_df, test_df])

    # 处理缺失值（'?'替换为NaN）
    numeric_cols = COLUMN_NAMES[1:]

    for col in numeric_cols:
        # 先将可能的字符串'?'转换为NaN
        full_df[col] = full_df[col].replace('?', np.nan)
        # 然后转换为数值类型
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    # 添加日期列
    full_df['Date'] = full_df['DateTime'].dt.date

    # 按天聚合数据
    daily_agg = full_df.groupby('Date').agg({
        'Global_active_power': 'sum', # 目标值
        'Global_reactive_power': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    })

    # 计算剩余能耗
    daily_agg['Total_energy'] = daily_agg['Global_active_power'] / 60  # kWh
    daily_agg['Sub_metering_total'] = daily_agg[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].sum(axis=1)
    daily_agg['Sub_metering_remainder'] = (daily_agg['Total_energy'] * 1000) - daily_agg['Sub_metering_total']

    # 处理负值和缺失值
    daily_agg['Sub_metering_remainder'] = daily_agg['Sub_metering_remainder'].clip(lower=0)
    daily_agg.fillna(method='ffill', inplace=True)
    daily_agg.fillna(0, inplace=True)

    # 删除临时列
    daily_agg.drop(['Total_energy', 'Sub_metering_total'], axis=1, inplace=True)

    # 排序并重置索引
    daily_agg = daily_agg.sort_index().reset_index()

    # 划分回训练和测试集
    train_dates = train_df['DateTime'].dt.date.unique()
    test_dates = test_df['DateTime'].dt.date.unique()

    train_data = daily_agg[daily_agg['Date'].isin(train_dates)]
    test_data = daily_agg[daily_agg['Date'].isin(test_dates)]

    return train_data, test_data


def create_sliding_windows(data, dates, target_col, input_length, output_length):
    """使用滑动窗口方法创建时间序列样本"""
    X, y, date_seq = [], [], []
    n = len(data)

    # 确保有足够的数据点
    if n < input_length + output_length:
        return np.array([]), np.array([]), np.array([])

    # 使用滑动窗口生成样本
    for i in range(n - input_length - output_length + 1):
        # 输入序列: [i, i+input_length)
        X.append(data[i:i + input_length])

        # 输出序列: [i+input_length, i+input_length+output_length)
        y.append(data[i + input_length:i + input_length + output_length, target_col])

        # 预测日期序列
        date_seq.append(dates[i + input_length:i + input_length + output_length])
    return np.array(X), np.array(y), np.array(date_seq)


def prepare_datasets():
    """准备训练和测试数据集，使用滑动窗口方法"""
    # 加载和处理数据
    train_data, test_data = load_and_preprocess()

    # 获取目标列索引
    target_idx = train_data.columns.get_loc(Config.TARGET_COL) - 1  # 减去日期列

    # 提取日期
    train_dates = train_data['Date'].values
    test_dates = test_data['Date'].values

    # 提取特征值（跳过日期列）
    train_values = train_data.iloc[:, 1:].values
    test_values = test_data.iloc[:, 1:].values

    # 标准化数据
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)

    # 创建短期预测数据集
    x_train_short, y_train_short, train_date_short = create_sliding_windows(
        train_scaled, train_dates, target_idx, Config.HISTORY_LEN, Config.SHORT_PRED_LEN
    )
    x_test_short, y_test_short, test_date_short  = create_sliding_windows(
        test_scaled, test_dates, target_idx, Config.HISTORY_LEN, Config.SHORT_PRED_LEN
    )

    # 创建长期预测数据集
    x_train_long, y_train_long, train_date_long = create_sliding_windows(
        train_scaled, train_dates, target_idx, Config.HISTORY_LEN, Config.LONG_PRED_LEN
    )
    x_test_long, y_test_long, test_date_long  = create_sliding_windows(
        test_scaled, test_dates, target_idx, Config.HISTORY_LEN, Config.LONG_PRED_LEN
    )

    # 检查数据有效性
    if x_train_short.size == 0 or x_train_long.size == 0:
        raise ValueError("Data Error: Not enough data for training or testing.")

    return (
        x_train_short, y_train_short, x_test_short, y_test_short, test_date_short,
        x_train_long, y_train_long, x_test_long, y_test_long, test_date_long,
        scaler
    )