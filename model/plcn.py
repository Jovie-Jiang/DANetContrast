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

    def forward(self, raw_h):
        batch_size, channels, depth, height, width = raw_h.size()
        input_h = raw_h.view(batch_size, depth, -1)
        self.lstm.flatten_parameters()
        lstm_out_h, _ = self.lstm(input_h)
        lstm_out_h = lstm_out_h[:, -1, :]
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
    def __init__(self,num_classes):
        super(PLCNModel, self).__init__()

        self.lstm = LSTMBranch(input_dim = 50*50*2 , 
                               hidden_dim = 128,
                               num_layers = 1,
                               convnext_dim =128
                            )
        self.cnn3d = CNN3DModule()
        self.fc = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x1, x2, x3, x4):
        raw_h = torch.cat((x1, x3), dim=1)  
        raw_v = torch.cat((x2, x4), dim=1)  
        x3d = torch.cat((raw_h, raw_v), dim=1)   
        out1 = self.lstm(raw_h)
        out2 = self.lstm(raw_v)
        out3d = self.cnn3d(x3d)
        combined = torch.cat((out1, out2, out3d), dim=1)
        return self.fc(combined)
    

if __name__ == '__main__':
    # 假设有一个输入视频数据，形状为 (batch_size, 1, depth, height, width)
    input_tensor1 = torch.randn(2, 1, 10, 50, 50).cuda()  # 例如，一个16帧的灰度视频片段
    input_tensor2 = torch.randn(2, 1, 10, 50, 50).cuda()  # 例如，一个16帧的灰度视频片段
    input_tensor3 = torch.randn(2, 1, 10, 50, 50).cuda()  # 例如，一个16帧的灰度视频片段
    input_tensor4 = torch.randn(2, 1, 10, 50, 50).cuda()  # 例如，一个16帧的灰度视频片段
    model = PLCNModel(num_classes=9).cuda()
    output = model(input_tensor1, input_tensor2,input_tensor3,input_tensor4)
    print(output)  # 输出形状应为 (batch_size, num_classes)