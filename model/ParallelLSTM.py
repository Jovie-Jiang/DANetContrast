import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMBranch(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 num_layers, 
                 convnext_dim):
        super(LSTMBranch, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, convnext_dim)

    def forward(self, x):
        # 获取输入的尺寸信息
        batch_size, channels, depth, height, width = x.size()
        # 将每个输入展平成二维张量 (batch_size, depth, height*width*channels)
        input_h = x.view(batch_size, depth, -1)
        self.lstm.flatten_parameters()
        # 分别通过2个 LSTM 进行处理
        lstm_out_h, _ = self.lstm(input_h)
        # 取每个 LSTM 的最后一个时间步的输出
        lstm_out_h = lstm_out_h[:, -1, :]
        # 调整到与 convnext_out 一致的输出维度
        out = self.fc(lstm_out_h)
        return out

class CNN3DModule(nn.Module):
    def __init__(self):
        super(CNN3DModule, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),

            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
        )
        # Add an Adaptive Avg Pool to ensure the output is always (batch_size, 128)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        # x: (batch_size, channels, depth, height, width)
        x = self.conv_layers(x)
        # Apply adaptive pooling to ensure fixed output size
        x = self.adaptive_pool(x)
        return x.flatten(1)  # Flatten to (batch_size, 128)


class PLCNModel(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes):
        super(PLCNModel, self).__init__()

        # Three LSTMs for temporal feature extraction
        self.lstm = LSTMBranch(input_dim = 64 * 64 * 2 , 
                                      hidden_dim = hidden_size,
                                      num_layers = num_layers,
                                      convnext_dim =128
                                      )
        self.cnn3d = CNN3DModule()
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x1, x2, x3, x4):
        raw_h = torch.cat((x1, x3), dim=1)  # 合并 input_h 和 dop_h
        raw_v = torch.cat((x2, x4), dim=1)  # 合并 input_v 和 dop_v
        x3d = torch.cat((raw_h, raw_v), dim=1)  # (batch_size, 2 * channel, sequence_length, height, width)
        out1 = self.lstm(raw_h)
        out2 = self.lstm(raw_v)
        out3d = self.cnn3d(x3d)
        # combined = torch.cat((out1, out2, out3d), dim=1)
        # combined_flattened = combined.view(combined.size(0), -1)
        # return self.fc(combined_flattened)
        combined = out1 + out2 + out3d

        # Pass through the fully connected layer
        return self.fc(combined)


if __name__ == '__main__':
    # 创建一个 PLCN 模型实例
    model = PLCNModel(hidden_size=128, num_layers=2, num_classes=9)
    # 创建随机输入张量，假设每种输入尺寸如下：
    # LSTM 输入：时间序列的三个光谱图 (sequence_length, input_size)
    x1 = torch.randn(8, 1, 60, 64, 64)  # STFT 输入
    x2 = torch.randn(8, 1, 60, 64, 64)  # SPWVD 输入
    x3 = torch.randn(8, 1, 60, 64, 64)  # STFT 输入
    x4 = torch.randn(8, 1, 60, 64, 64)  # SPWVD 输入
    # 前向传播
    output = model(x1, x2, x3, x4)
    # 打印模型输出和形状
    print(output)
    # print("输出形状:", output.shape)
