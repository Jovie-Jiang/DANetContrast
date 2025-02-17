import torch
import torch.nn as nn
import torch.nn.functional as F

'''-------------一、BasicBlock模块-----------------------------'''
# 用于ResNet18和ResNet34基本残差结构块
class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

'''-------------二、Bottleneck模块-----------------------------'''
# 用于ResNet50及以上的残差结构块
class Bottleneck3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels // 4)
        self.conv3 = nn.Conv3d(out_channels // 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

'''----------ResNet18----------'''
class ResNet_18_3D(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNet_18_3D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(block, 64, 2, stride=1)
        self.layer2 = self.make_layer(block, 128, 2, stride=2)
        self.layer3 = self.make_layer(block, 256, 2, stride=2)
        self.layer4 = self.make_layer(block, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
 
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)
 
    def forward(self, x1, x2, x3, x4):
        raw_h = torch.cat((x1, x3), dim=1)  # 合并 input_h 和 dop_h
        raw_v = torch.cat((x2, x4), dim=1)  # 合并 input_v 和 dop_v

        out1 = self.conv1(raw_h)
        out1 = self.layer1(out1)
        out1 = self.layer2(out1)
        out1 = self.layer3(out1)
        out1 = self.layer4(out1)
        out1 = F.adaptive_avg_pool3d(out1, (1, 1, 1))  # Global average pooling
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc(out1)

        out2 = self.conv1(raw_v)
        out2 = self.layer1(out2)
        out2 = self.layer2(out2)
        out2 = self.layer3(out2)
        out2 = self.layer4(out2)
        out2 = F.adaptive_avg_pool3d(out2, (1, 1, 1))  # Global average pooling
        out2 = out2.view(out2.size(0), -1)
        out2 = self.fc(out2)
        return out1 + out2
        # out = self.conv1(x)
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = F.adaptive_avg_pool3d(out, (1, 1, 1))  # Global average pooling
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        # return out

'''----------ResNet34----------'''
class ResNet_34_3D(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNet_34_3D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(block, 64, 3, stride=1)
        self.layer2 = self.make_layer(block, 128, 4, stride=2)
        self.layer3 = self.make_layer(block, 256, 6, stride=2)
        self.layer4 = self.make_layer(block, 512, 3, stride=2)
        self.fc = nn.Linear(512, num_classes)
 
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)
 
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))  # Global average pooling
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

'''---------ResNet50--------'''
class ResNet_50_3D(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNet_50_3D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(block, 256, 3, stride=1)
        self.layer2 = self.make_layer(block, 512, 4, stride=2)
        self.layer3 = self.make_layer(block, 1024, 6, stride=2)
        self.layer4 = self.make_layer(block, 2048, 3, stride=2)
        self.fc = nn.Linear(2048, num_classes)  # 修正fc层输入维度
 
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)
 
    def forward(self, x1,x2,x3,x4):
        raw_h = torch.cat((x1, x3), dim=1)  # 合并 input_h 和 dop_h
        raw_v = torch.cat((x2, x4), dim=1)  # 合并 input_v 和 dop_v

        out1 = self.conv1(raw_h)
        out1 = self.layer1(out1)
        out1 = self.layer2(out1)
        out1 = self.layer3(out1)
        out1 = self.layer4(out1)
        out1 = F.adaptive_avg_pool3d(out1, (1, 1, 1))  # Global average pooling
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc(out1)

        out2 = self.conv1(raw_v)
        out2 = self.layer1(out2)
        out2 = self.layer2(out2)
        out2 = self.layer3(out2)
        out2 = self.layer4(out2)
        out2 = F.adaptive_avg_pool3d(out2, (1, 1, 1))  # Global average pooling
        out2 = out2.view(out2.size(0), -1)
        out2 = self.fc(out2)
        return out1+out2

        # out = self.conv1(x)
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = F.adaptive_avg_pool3d(out, (1, 1, 1))  # Global average pooling
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        # predicted_class = torch.argmax(out, dim=1)  # Find the class with the highest probability
        # return predicted_class
        # return out


if __name__ == '__main__':
    # 创建一个 ResNet_50 实例
    model = ResNet_18_3D(Bottleneck3D, num_classes=9)
    
    # 创建一个随机输入张量，假设输入尺寸为 3x32x32
    x1 = torch.randn(1, 1, 60, 64, 64)
    x2 = torch.randn(1, 1, 60, 64, 64)
    x3 = torch.randn(1, 1, 60, 64, 64)
    x4 = torch.randn(1, 1, 60, 64, 64)
    
    # 通过模型进行前向传播
    output = model(x1,x2,x3,x4)
    print(output)
    
    # 打印输出的形状
    # print("Output shape:", output.shape)