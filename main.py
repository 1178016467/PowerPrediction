import numpy as np
from data_processing import prepare_datasets
from models import Seq2SeqLSTM, PowerTransformer, MambaTimeSeriesModel
from utils import evaluate_multiple_runs, print_results, plot_history, plot_prediction_comparison, visualize_predictions
from config import Config
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    # 准备数据 (使用滑动窗口)
    (x_train_short, y_train_short, x_test_short, y_test_short, test_date_short,
     x_train_long, y_train_long, x_test_long, y_test_long, test_date_long,
     scaler) = prepare_datasets()

    # 模型参数
    model_args = {
        'input_size': Config.INPUT_SIZE,
        'd_model': Config.D_MODEL,
        'num_layers': Config.NUM_LAYERS,
    }

    # 短期预测实验
    print("\n" + "=" * 80)
    print("SHORT-TERM FORECASTING (90 days)")
    print("=" * 80)

    # 用于绘制对比图
    test_dataset_short = torch.utils.data.TensorDataset(
        torch.tensor(x_test_short, dtype=torch.float32),
        torch.tensor(y_test_short, dtype=torch.float32)
    )
    test_loader_short = torch.utils.data.DataLoader(
        test_dataset_short,
        batch_size=Config.BATCH_SIZE
    )

    # LSTM模型
    lstm_args_short = {'input_size': Config.INPUT_SIZE, 'hidden_size': Config.HIDDEN_SIZE, 'output_length': Config.SHORT_PRED_LEN}
    print("\n>>> Training LSTM for short-term forecasting")
    lstm_model_short, lstm_short_results = evaluate_multiple_runs(
        "LSTM_short",
        Seq2SeqLSTM, lstm_args_short,
        x_train_short, y_train_short,
        x_test_short, y_test_short,
        scaler,
        num_runs=Config.NUM_RUNS
    )
    print_results(lstm_short_results, "LSTM", "Short-term")
    plot_history(lstm_short_results['histories'], "LSTM", "Short-term")

    # 获取预测结果
    preds, targets = lstm_model_short.predict(test_loader_short)

    # 绘制预测对比图
    mse, mae =plot_prediction_comparison(
        "LSTM_short",
        targets,
        preds,
        scaler,
        test_date_short,
        prediction_type='short',
        save_path="results/LSTM_short/individual_comparison.png"
    )
    all_mse, all_mae = visualize_predictions(
        "LSTM_short",
        lstm_model_short,
        test_loader_short,
        scaler,
        test_date_short,
        prediction_type='short',
        save_path="results/LSTM_short/overall_comparison.png"
    )

    # Transformer模型
    transformer_args_short = {**model_args, 'nhead': Config.NHEAD, 'output_length': Config.SHORT_PRED_LEN}
    print("\n>>> Training Transformer for short-term forecasting")
    transformer_model_short, transformer_short_results = evaluate_multiple_runs(
        "Transformer_short",
        PowerTransformer, transformer_args_short,
        x_train_short, y_train_short,
        x_test_short, y_test_short,
        scaler,
        num_runs=Config.NUM_RUNS
    )
    print_results(transformer_short_results, "Transformer", "Short-term")
    plot_history(transformer_short_results['histories'], "Transformer", "Short-term")

    # 获取预测结果
    preds, targets = transformer_model_short.predict(test_loader_short)

    # 绘制预测对比图
    mse, mae = plot_prediction_comparison(
        "Transformer_short",
        targets,
        preds,
        scaler,
        test_date_short,
        prediction_type='short',
        save_path="results/Transformer_short/individual_comparison.png"
    )
    all_mse, all_mae = visualize_predictions(
        "Transformer_short",
        transformer_model_short,
        test_loader_short,
        scaler,
        test_date_short,
        prediction_type='short',
        save_path="results/Transformer_short/overall_comparison.png"
    )

    # Mamba模型
    mamba_args_short = {**model_args, 'output_length': Config.SHORT_PRED_LEN}
    print("\n>>> Training MambaTimeSeriesModel for short-term forecasting")
    mamba_model_short, mamba_short_results  = evaluate_multiple_runs(
        "Mamba_short",
        MambaTimeSeriesModel, mamba_args_short,
        x_train_short, y_train_short,
        x_test_short, y_test_short,
        scaler,
        num_runs=Config.NUM_RUNS
    )
    print_results(mamba_short_results , "MambaTimeSeriesModel", "Short-term")
    plot_history(mamba_short_results ['histories'], "MambaTimeSeriesModel", "Short-term")

    # 获取预测结果
    preds, targets = mamba_model_short.predict(test_loader_short)

    # 绘制预测对比图
    mse, mae = plot_prediction_comparison(
        "Mamba_short",
        targets,
        preds,
        scaler,
        test_date_short,
        prediction_type='short',
        save_path="results/Mamba_short/individual_comparison.png"
    )
    all_mse, all_mae = visualize_predictions(
        "Mamba_short",
        mamba_model_short,
        test_loader_short,
        scaler,
        test_date_short,
        prediction_type='short',
        save_path="results/Mamba_short/overall_comparison.png"
    )

    # 长期预测实验
    print("\n" + "=" * 80)
    print("LONG-TERM FORECASTING (365 days)")
    print("=" * 80)

    # 用于绘制对比图
    test_dataset_long = torch.utils.data.TensorDataset(
        torch.tensor(x_test_long, dtype=torch.float32),
        torch.tensor(y_test_long, dtype=torch.float32)
    )
    test_loader_long = torch.utils.data.DataLoader(
        test_dataset_long,
        batch_size=Config.BATCH_SIZE
    )

    # LSTM模型
    lstm_args_long = {'input_size': Config.INPUT_SIZE, 'hidden_size': Config.HIDDEN_SIZE, 'output_length': Config.LONG_PRED_LEN}
    print("\n>>> Training LSTM for long-term forecasting")
    lstm_model_long,lstm_long_results = evaluate_multiple_runs(
        "LSTM_long",
        Seq2SeqLSTM, lstm_args_long,
        x_train_long, y_train_long,
        x_test_long, y_test_long,
        scaler,
        num_runs=Config.NUM_RUNS
    )
    print_results(lstm_long_results, "LSTM", "Long-term")
    plot_history(lstm_long_results['histories'], "LSTM", "Long-term")

    # 获取预测结果
    preds, targets = lstm_model_long.predict(test_loader_long)

    # 绘制预测对比图
    mse, mae = plot_prediction_comparison(
        "LSTM_long",
        targets,
        preds,
        scaler,
        test_date_long,
        prediction_type='long',
        save_path="results/LSTM_long/individual_comparison.png"
    )
    all_mse, all_mae = visualize_predictions(
        "LSTM_long",
        lstm_model_long,
        test_loader_long,
        scaler,
        test_date_long,
        prediction_type='long',
        save_path="results/LSTM_long/overall_comparison.png"
    )

    # Transformer模型
    transformer_args_long = {**model_args, 'nhead': Config.NHEAD, 'output_length': Config.LONG_PRED_LEN}
    print("\n>>> Training Transformer for long-term forecasting")
    transformer_model_long,transformer_long_results = evaluate_multiple_runs(
        "Transformer_long",
        PowerTransformer, transformer_args_long,
        x_train_long, y_train_long,
        x_test_long, y_test_long,
        scaler,
        num_runs=Config.NUM_RUNS
    )
    print_results(transformer_long_results, "Transformer", "Long-term")
    plot_history(transformer_long_results['histories'], "Transformer", "Long-term")

    # 获取预测结果
    preds, targets = transformer_model_long.predict(test_loader_long)

    # 绘制预测对比图
    mse, mae = plot_prediction_comparison(
        "Transformer_long",
        targets,
        preds,
        scaler,
        test_date_long,
        prediction_type='long',
        save_path="results/Transformer_long/individual_comparison.png"
    )
    all_mse, all_mae = visualize_predictions(
        "Transformer_long",
        transformer_model_long,
        test_loader_long,
        scaler,
        test_date_long,
        prediction_type='long',
        save_path="results/Transformer_long/overall_comparison.png"
    )

    # Mamba模型
    mamba_args_long = {**model_args, 'output_length': Config.LONG_PRED_LEN}
    print("\n>>> Training MambaTimeSeriesModel for long-term forecasting")
    mamba_model_long,mamba_long_results  = evaluate_multiple_runs(
        "Mamba_long",
        MambaTimeSeriesModel, mamba_args_long,
        x_train_long, y_train_long,
        x_test_long, y_test_long,
        scaler,
        num_runs=Config.NUM_RUNS
    )
    print_results(mamba_long_results, "MambaTimeSeriesModel", "Long-term")
    plot_history(mamba_long_results['histories'], "MambaTimeSeriesModel", "Long-term")


    # 获取预测结果
    preds, targets = mamba_model_long.predict(test_loader_long)

    # 绘制预测对比图
    mse, mae = plot_prediction_comparison(
        "Mamba_long",
        targets,
        preds,
        scaler,
        test_date_long,
        prediction_type='long',
        save_path="results/Mamba_long/individual_comparison.png"
    )
    all_mse, all_mae = visualize_predictions(
        "Mamba_long",
        mamba_model_long,
        test_loader_long,
        scaler,
        test_date_long,
        prediction_type='long',
        save_path="results/Mamba_long/overall_comparison.png"
    )

    # 保存最终结果
    final_results = {
        'short_term': {
            'LSTM': lstm_short_results,
            'Transformer': transformer_short_results,
            'Mamba': mamba_short_results
        },
        'long_term': {
            'LSTM': lstm_long_results,
            'Transformer': transformer_long_results,
            'Mamba': mamba_long_results
        }
    }
    np.save("results/final_results.npy", final_results)
    print("\nAll experiments completed. Results saved to results/final_results.npy")


if __name__ == "__main__":
    main()