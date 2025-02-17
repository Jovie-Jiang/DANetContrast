# 库函数调用
import torch
import torch.nn as nn

# VGG3D模块
class CustomVGG3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=2,num_classes=4):
        super(CustomVGG3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

        )

        self.classifier = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),  # 修改为 num_classes
            nn.Softmax(dim=1)  # Softmax 用于多分类
        )

    def forward(self, x1,x2,x3,x4):
        # x1 = torch.cat((x1, x2, x3, x4), dim=1)  # 合并 input_h 和 dop_h
        # x2 = torch.cat((x2, x4), dim=1)  # 合并 input_v 和 dop_v

        x1 = torch.cat((x1, x3), dim=1)  # 合并 input_h 和 dop_h
        x2 = torch.cat((x2, x4), dim=1)  # 合并 input_v 和 dop_v

        x1 = self.features(x1)
        x1 = x1.view(x1.size(0), -1)
        # x1 = self.classifier(x1)

        x2 = self.features(x2)
        x2 = x2.view(x2.size(0), -1)
        # x2 = self.classifier(x2)

        # x3 = self.features(x3)
        # x3 = x3.view(x3.size(0), -1)
        # # x3 = self.classifier(x3)

        # x4 = self.features(x4)
        # x4 = x4.view(x4.size(0), -1)
        # # x4 = self.classifier(x4)

        # x = x1 + x2 +x3 + x4
        # x = self.classifier(x)

        # predicted_class = torch.argmax(x, dim=1)
        # return predicted_class
        x = x1 + x2
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    x1 = torch.ones([1, 1, 40, 256, 32])
    x2 = torch.ones([1, 1, 40, 256, 32])
    x3 = torch.ones([1, 1, 40, 256, 32])
    x4 = torch.ones([1, 1, 40, 256, 32])

    model = CustomVGG3D(in_channels=2, out_channels=1,num_classes=9)
    f = model(x1,x2,x3,x4)
    print(f)
