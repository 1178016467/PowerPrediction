import torch
import torch.nn as nn
import numpy as np
from config import Config
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        seq_len = x.size(1)
        return x + self.pe[:seq_len].permute(1, 0, 2)


class AdaptivePositionalEncoding(nn.Module):
    """自适应位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.pe = PositionalEncoding(d_model, max_len).pe

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.alpha * self.pe[:seq_len].permute(1, 0, 2).to(x.device)


class Seq2SeqLSTM(nn.Module):
    """LSTM序列到序列模型"""

    def __init__(self, input_size, hidden_size, output_length):
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_length = output_length

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,  # 输入格式为 (batch, seq, features)
            dropout=Config.DROPOUT
        )

        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, output_length)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM 处理
        # lstm_out 形状: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # 只取最后一个时间步的输出
        # last_out 形状: (batch_size, hidden_size)
        last_out = lstm_out[:, -1, :]

        # 全连接层处理
        x = self.relu(self.fc1(last_out))
        output = self.fc2(x)

        return output


class PowerTransformer(nn.Module):
    """Transformer预测模型"""

    def __init__(self, input_size, d_model, nhead, num_layers, output_length):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.output_length = output_length

        # 输入嵌入层
        self.embedding = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            dropout=Config.DROPOUT, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(d_model * 2, output_length)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # 输入嵌入
        x = self.embedding(x)  # (batch, seq_len, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # 取最后一个时间步的特征
        x = x[:, -1, :]  # (batch, d_model)

        # 输出层
        return self.output_layer(x)  # (batch, output_length)


class MambaBlock(nn.Module):
    """Mamba 块的核心组件"""

    def __init__(self, input_dim, hidden_dim, state_dim, dt_rank):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.dt_rank = dt_rank

        # 输入投影层
        self.in_proj = nn.Linear(input_dim, hidden_dim, bias=False)

        # 卷积层
        self.conv1d = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
            bias=False
        )

        # 状态空间参数
        self.A = nn.Parameter(torch.randn(hidden_dim, state_dim))
        self.D = nn.Parameter(torch.ones(hidden_dim))

        # 时间步长参数 - 修复实现
        self.dt_proj = nn.Sequential(
            nn.Linear(hidden_dim, dt_rank, bias=True),
            nn.ReLU(),
            nn.Linear(dt_rank, hidden_dim, bias=True)
        )

        # 输出投影层
        self.out_proj = nn.Linear(hidden_dim, input_dim, bias=False)

        # 层归一化
        self.norm = nn.LayerNorm(input_dim)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        # 确保稳定性
        nn.init.normal_(self.A, mean=0.0, std=0.01)
        nn.init.orthogonal_(self.in_proj.weight)
        nn.init.orthogonal_(self.out_proj.weight)

        # 卷积层初始化
        nn.init.kaiming_normal_(self.conv1d.weight, nonlinearity='relu')

        # dt_proj 初始化
        nn.init.xavier_uniform_(self.dt_proj[0].weight)
        nn.init.zeros_(self.dt_proj[0].bias)
        nn.init.xavier_uniform_(self.dt_proj[2].weight)
        nn.init.zeros_(self.dt_proj[2].bias)

    def forward(self, x):
        # 归一化
        residual = x
        x = self.norm(x)

        # 输入投影 - 形状: (batch, seq_len, hidden_dim)
        x = self.in_proj(x)

        # 1D 卷积处理
        # 正确形状变换: (batch, hidden_dim, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = F.silu(x)
        # 恢复形状: (batch, seq_len, hidden_dim)
        x = x.permute(0, 2, 1)

        # 计算时间步长参数 - 修复实现
        # 输入形状: (batch, seq_len, hidden_dim)
        # 输出形状: (batch, seq_len, hidden_dim)
        dt = self.dt_proj(x)
        dt = F.softplus(dt)

        # 离散化状态空间模型，确保数值稳定性
        A = -torch.exp(self.A)  # (hidden_dim, state_dim)

        # 状态空间模型计算
        batch_size, seq_len, _ = x.shape
        state = torch.zeros(batch_size, self.hidden_dim, self.state_dim, device=x.device)
        outputs = []

        for i in range(seq_len):
            # 获取当前时间步的特征
            x_t = x[:, i, :]  # (batch, hidden_dim)
            dt_t = dt[:, i, :]  # (batch, hidden_dim)

            # 离散化参数 - 正确广播
            # 形状: (batch, hidden_dim, state_dim)
            dA = A * dt_t.unsqueeze(-1)
            discrete_A = torch.exp(dA)

            # 安全处理小值
            safe_A = torch.where(torch.abs(A) < 1e-5, torch.ones_like(A), A)
            discrete_B = (discrete_A - 1) / safe_A * dt_t.unsqueeze(-1)

            # 更新状态
            state = discrete_A * state + discrete_B * x_t.unsqueeze(-1)

            # 计算输出，使用 einsum 确保正确广播
            y = torch.einsum('bhc,hc->bh', state, self.A) + self.D * x_t
            outputs.append(y)

        # 堆叠输出: (batch, seq_len, hidden_dim)
        x = torch.stack(outputs, dim=1)

        # 输出投影
        x = self.out_proj(x)

        # 残差连接
        return residual + x


class MambaTimeSeriesModel(nn.Module):
    """基于 Mamba 的时间序列预测模型"""
    def __init__(self, input_size, output_length,
                 d_model=256, state_dim=16, dt_rank=8,
                 num_layers=4, expansion=2):

        super().__init__()
        self.input_size = input_size
        self.output_length = output_length
        hidden_dim = d_model

        # 输入嵌入层
        self.embedding = nn.Linear(input_size, hidden_dim)

        # Mamba 块堆叠
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim * expansion,
                state_dim=state_dim,
                dt_rank=dt_rank
            )
            for _ in range(num_layers)
        ])

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(hidden_dim, output_length)
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # 自适应归一化
        self.norm = nn.LayerNorm(hidden_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 嵌入层初始化
        nn.init.xavier_uniform_(self.embedding.weight)
        # 输出层初始化
        nn.init.kaiming_uniform_(self.output_layer[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.output_layer[0].bias)
        nn.init.xavier_uniform_(self.output_layer[3].weight)
        nn.init.zeros_(self.output_layer[3].bias)

    def forward(self, x):
        # 输入嵌入
        x = self.embedding(x)  # (batch, seq_len, hidden_dim)

        # 位置编码
        x = self.pos_encoder(x)

        # 通过 Mamba 块
        for block in self.mamba_blocks:
            x = block(x)

        # 归一化
        x = self.norm(x)

        # 取最后一个时间步的特征
        x = x[:, -1, :]  # (batch, hidden_dim)

        # 输出层
        return self.output_layer(x)  # (batch, output_length)