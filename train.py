import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from load_dataset import loaddatasplit
from util import get_params_groups, create_lr_scheduler
from model import vgg, vivit, swin_transformer, timesformer_off, resnet, efficientnet,ParallelLSTM,plcn
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast, GradScaler


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float)  
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=1e-3, type=float)  # 权重衰减参数，以防止模型过度拟合训练数据， AdamW 优化器使用

    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--num_classes', type=int, default=9)

    parser.add_argument('--train_path', default='/root/share/jjw_data/realdata_all/train', help='load train datasets')
    parser.add_argument('--valid_path', default='/root/share/jjw_data/realdata_all/train', help='load train datasets')
   
    parser.add_argument('--weights', default="weights/PLCNModel_action9.pth", help='initial weights path')
    parser.add_argument('--time_patch', default=60, type=int)

    parser.add_argument('--box_size', default=8, type=int)
    parser.add_argument('--boundary_size', default=10, type=int)
    parser.add_argument('--output_dir', default='log', help='path where to save, empty for no saving')

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training/testing')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    return parser


if __name__ == '__main__':
    # 创建参数解析器
    # 解析命令行参数
    parser = get_argparser()
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用自定义的load_dataset.Load_data函数加载训练集和验证集
    train_set = loaddatasplit.Load_data(args, mode="train")
    validation_set = loaddatasplit.Load_data(args, mode="valid")

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                                              num_workers=args.num_workers, pin_memory=True,persistent_workers=True)
    validationloader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False,
                                                    num_workers=args.num_workers, pin_memory=True,persistent_workers=True)
    print("训练集数据加载完毕", len(train_set))
    print("验证集数据加载完毕", len(validation_set))

    # model = ConvNeXt_3d.convnext_tiny(num_classes=args.num_classes)
    # model = ConvLstmSENET.convnext_3d_tiny(num_classes=args.num_classes)
    # model = combinemodel.arcra(num_classes=args.num_classes)
    # model = swin_transformer.SwinTransformer3D()# 64*64
    # model = vivit.ViViTBackbone(
    #         t=60,h=64, w=64,  # 输入视频的时间维度（长度）、高度和宽度
    #         patch_t=6, patch_h=4, patch_w=4,  #每个时间维度、高度和宽度上的图像块大小
    #         num_classes=9,  # 分类的类别数
    #         dim=512,        # 模型中特征的维度
    #         depth=6,        # 模型中的层数
    #         heads=4,       # 自注意力机制中的头数
    #         mlp_dim=8,      # MLP（多层感知机）中的隐藏层维度
    #         model=3         # 模型的类型
    #     )   # 64*64
    # model = resnet.ResNet_18_3D(resnet.Bottleneck3D, num_classes=9)

    # model = vgg.CustomVGG3D(in_channels=2, out_channels=1,num_classes=9)
    # model = efficientnet.EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 9}, in_channels=2)

    model = plcn.PLCNModel(num_classes=9)
    
    # model = timesformer_off.TimeSformer(
    #         dim = 512,
    #         image_size = 224,
    #         patch_size = 16,
    #         num_frames = 10,
    #         num_classes = 9,
    #         depth = 12,
    #         heads = 8,
    #         dim_head =  64,
    #         attn_dropout = 0.1,
    #         ff_dropout = 0.1
        # )
    
    # 动态获取模型的名称
    model_name = type(model).__name__
    save_path = f"/root/projects/JJW_Model/contrast/weights/{model_name}_action.pth"

    # 参数预加载,如果指定了预训练权重文件，则加载该文件中的权重到模型中
    if args.weights != "":
        # 使用 assert 语句检查指定的权重文件是否存在
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        # 从整个字典中提取出与模型相关的部分，将文件中的内容存储在 state_dict 变量中。
        state_dict = torch.load(args.weights, map_location=device)["model_state_dict"]
        model.load_state_dict(state_dict)  # 将加载的状态字典应用到当前创建的模型中

    model = model.to(device)

    # 配置优化器和学习率调度器  调用get_params_groups函数，获取指定权重衰减（L2正则化）的参数
    pg = get_params_groups(model, weight_decay=args.weight_decay)
    # 创建 AdamW 优化器，在优化器的更新规则上添加了权重衰减
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(trainloader), args.epochs, warmup=True, warmup_epochs=1)
    criterion = nn.CrossEntropyLoss() 

    scaler = torch.cuda.amp.GradScaler()  # 混合精度训练的梯度缩放器

    # 构建日志文件路径
    log_file = f'/root/projects/JJW_Model/contrast/logs/{model_name}_action9.log'
    previous_acc = 0.84080
    previous_loss = 10000

    with open( log_file, 'a') as f:
        for epoch in range(args.epochs):
            f.write(f"Epoch {epoch + 1}\n")
            model.train()
            running_loss = 0.0
            all_sample = 0.0
            confidence_sample = 0.0
            correct_sample = 0.0
            for i, (raw_h, raw_v, dop_h, dop_v, action, action_confidence) in enumerate(trainloader):
                raw_h = raw_h.to(device, non_blocking=True)
                raw_v = raw_v.to(device, non_blocking=True)
                dop_h = dop_h.to(device, non_blocking=True)
                dop_v = dop_v.to(device, non_blocking=True)
                action = action.to(device, non_blocking=True)
                action = action.to(device, non_blocking=True).long()
                action_confidence = action_confidence.to(device, non_blocking=True)

                optimizer.zero_grad()

                # 自动混合精度上下文
                with torch.cuda.amp.autocast():
                    output = model(raw_h, raw_v, dop_h, dop_v)  # 前向传播
                    loss = criterion(output, action)  # 计算损失

                # 使用 scaler 进行反向传播
                scaler.scale(loss).backward()  # 梯度计算
                scaler.step(optimizer)         # 梯度更新
                scaler.update()                # 更新缩放比例

                running_loss += loss.item()
                prediction = torch.argmax(output, 1)
                all_sample = all_sample + len(action)
                # calculation of accuracy

                for pred, true_label, val in zip(prediction, action, action_confidence):
                    if bool(val) and true_label in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                         confidence_sample += 1
                         if pred == true_label:
                            correct_sample += 1

                if (i+1) % 100 == 0:
                    print('[%d, %5d] loss: %.3f, accuracy: %.5f' % (
                        epoch + 1, i + 1, running_loss / all_sample, correct_sample / confidence_sample))

            model.eval()
            validation_loss = 0
            val_all_sample = 0.0
            val_confidence_sample = 0.0
            val_correct_sample = 0.0
            y_true_val = []  
            y_pred_val = []  

            with torch.no_grad():
                with autocast():
                    for i, (raw_h, raw_v, dop_h, dop_v, action, action_confidence) in enumerate(validationloader):
                        raw_h = raw_h.to(device, non_blocking=True)
                        raw_v = raw_v.to(device, non_blocking=True)
                        dop_h = dop_h.to(device, non_blocking=True)
                        dop_v = dop_v.to(device, non_blocking=True)
                        action = action.to(device, non_blocking=True).long()
                        action_confidence = action_confidence.to(device, non_blocking=True)

                        output = model(raw_h, raw_v, dop_h, dop_v)
                        loss = criterion(output, action)
                        validation_loss += loss.item()

                        prediction = torch.argmax(output, 1)
                        val_all_sample = val_all_sample + len(action)
                        
                        for pred, true_label, val in zip(prediction, action, action_confidence):
                            if bool(val) and true_label in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                                val_confidence_sample += 1
                                if pred == true_label:
                                    val_correct_sample += 1
                                y_true_val.append(true_label.item())
                                y_pred_val.append(pred.item())

            cm_val = confusion_matrix(y_true_val, y_pred_val)
            print(f"Validation Confusion Matrix at Epoch {epoch + 1}:\n{cm_val}")
            f.write(f"[Epoch {epoch + 1}, validation] Confusion Matrix:\n{cm_val}\n")

            val_acc = val_correct_sample / val_confidence_sample
            val_loss = validation_loss / val_all_sample

            # 如果验证准确度提高或者准确度相同但验证损失更小，则保存模型
            if val_acc > previous_acc or (val_acc == previous_acc and val_loss < previous_loss):
                torch.save({'model_state_dict': model.state_dict()}, save_path)
                previous_acc = val_acc
                previous_loss = val_loss
                print('saved')

            print('all validation: %.5f, confidence validation: %.5f, correct validation: %.5f' % (
                val_all_sample, val_confidence_sample, val_correct_sample))
            print('val loss: %.5f, accuracy: %.5f' % (val_loss, val_acc))
            f.write(
                    f'[Epoch {epoch + 1}, tain] loss: {running_loss / all_sample:.3f}, train_accuracy: {correct_sample / confidence_sample:.5f}\n')
            f.write(
                    f'[Epoch {epoch + 1}, Vail] loss: {val_loss:.5f}, accuracy: {val_acc:.5f}\n')
            
            f.write(f'[Epoch {epoch + 1}] max_vail_acc: {previous_acc:.5f}\n')
            
            f.flush()  # 再次刷新缓冲区
