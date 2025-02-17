import logging
from skimage import measure
from torch.utils import data
import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import json
import math
from collections import Counter
from scipy.spatial.distance import cdist
torch.cuda.empty_cache()


def get_argparser():
    parser = argparse.ArgumentParser('Set transformer location', add_help=False)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')  # outputs/checkpoint.pth

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')

    parser.add_argument('--train_path', default='/root/share/jjw_data/realdata_all/train', help='load train datasets')
    parser.add_argument('--valid_path', default='/root/share/jjw_data/realdata_all/vail', help='load train datasets')

    parser.add_argument('--end_epoch', default=200, type=int)
    parser.add_argument('--is_train', default=True, type=bool)
    parser.add_argument('--size', default=64, type=int)
    parser.add_argument('--use_hxj', default=True, type=bool)

    parser.add_argument('--time_patch', default=60, type=int)

    parser.add_argument('--box_size', default=8, type=int)
    parser.add_argument('--boundary_size', default=10, type=int)
    parser.add_argument('--best_perf', default=0, type=float)
    parser.add_argument('--print_freq', default=10, type=int)
    return parser


def acquire_rootdir(root, num_type=None):  # 获取根目录路径
    if num_type is None:
        num_type = [1, 2, 3]
    dir_list = []
    for t1 in num_type:
        if t1 == 4 or t1 == 5:
            continue
        else:
            if os.path.exists(root + '/' + str(t1)):
                first_root = root + '/' + str(t1) + '/'
                second_type = os.listdir(root + '/' + str(t1))
                for t2 in second_type:
                    if os.listdir(first_root + t2):
                        dir_list.append(os.path.join(first_root, t2))
                    else:
                        continue
    return dir_list


def so_cfar_2d(img, num_train, num_guard, rate_fa, boundary_left, boundary_right):
    """
    实现 SO-CFAR 算法的二维矩阵版本

    参数:
    - matrix: 输入图像（二维numpy数组）
    - num_train: 每个参考窗口中的训练单元数（假设为正方形的区域）
    - num_guard: 每个参考窗口中的保护单元数（假设为正方形的区域）
    - rate_fa: 期望的虚警概率（假警率）

    返回:
    - detection: 与输入矩阵相同大小的布尔矩阵，其中 True 表示检测到目标
    """
    matrix = img.copy()
    rows, cols = matrix.shape
    detection = np.zeros((rows, cols), dtype=bool)

    # 计算门限因子
    alpha = num_train * (rate_fa ** (-1 / num_train) - 1)

    # 提取处理范围边界
    col_start = boundary_left
    col_end = boundary_right

    # 确保列边界在有效范围内
    col_start = max(col_start, num_guard + num_train)
    col_end = min(col_end, cols - (num_guard + num_train))

    # 滑动窗口检测
    for i in range(num_guard + num_train, rows - (num_guard + num_train)):
        for j in range(col_start, col_end):
            # 提取左上参考窗口和右下参考窗口
            leading_window = matrix[i - num_guard - num_train:i - num_guard, j - num_guard - num_train:j - num_guard]
            lagging_window = matrix[i + num_guard + 1:i + num_guard + 1 + num_train,
                             j + num_guard + 1:j + num_guard + 1 + num_train]
            

            # 计算左、右参考窗口的平均噪声功率
            noise_estimation_leading = np.mean(leading_window)
            noise_estimation_lagging = np.mean(lagging_window)

            # 使用两个参考窗口的最小平均值作为门限
            noise_estimation = min(noise_estimation_leading, noise_estimation_lagging)
            threshold = alpha * noise_estimation

            # 将测试单元与门限比较
            if matrix[i, j] > threshold:
                detection[i, j] = True

    # # 创建一个与输入矩阵相同大小的结果矩阵，保留检测目标和边界之外的原始数据
    # result_with_original_values = np.zeros_like(matrix)
    #
    # # 保留检测到目标的原始数值
    # result_with_original_values[detection] = matrix[detection]
    # return result_with_original_values
    return detection


def retain_largest_detections(img, detection, box_size, num_to_retain=1, distance_threshold=5):
    """
    在检测结果中仅保留最大的n个目标，并将靠近的目标合并为一个
    参数:
    - img: 输入图像(二维numpy数组)
    - detection: 检测结果（与输入图像相同大小的数值矩阵）
    - box_size: 要保留的正方形区域的大小
    - num_to_retain: 要保留的最大目标区域的数量
    - dilation_radius: 膨胀操作的半径（像素）
    返回:
    - start_row, end_row, start_col, end_col: 保留目标的边界坐标
    """

    # 复制原图
    matrix = img.copy()
    result_with_targets = np.zeros_like(matrix)

    # 使用 skimage 库的 measure 模块找到检测区域的轮廓
    labeled_matrix, num_features = measure.label(detection, return_num=True)

    # 如果没有检测到目标，直接返回空结果
    if num_features == 0:
        return 0, 0, 0, 0

    # 找到所有目标区域 提取每个目标的质心坐标
    region_props = measure.regionprops(labeled_matrix)
    # 提取每个目标的质心坐标
    centroids = np.array([region.centroid for region in region_props])

    # 计算质心之间的距离矩阵和面积
    distance_matrix = cdist(centroids, centroids)
    areas = np.array([region.area for region in region_props])

    # 初始化合并后的区域标签
    merged_labels = np.full(len(centroids), -1)  # -1 表示尚未归类
    current_label = 0

    # 对每个目标进行分组，距离小于阈值的归为一类
    for i in range(len(centroids)):
        if merged_labels[i] == -1:  # 目标尚未分组
            merged_labels[i] = current_label
            for j in range(i + 1, len(centroids)):
                if distance_matrix[i, j] < distance_threshold:
                    merged_labels[j] = current_label
            current_label += 1

    # 计算每个分组的总面积
    total_areas = []
    grouped_regions = []

    for label in range(current_label):
        group_indices = np.where(merged_labels == label)[0]
        group_area = areas[group_indices].sum()
        total_areas.append(group_area)

        # 计算分组区域的边界框
        group_coords = np.vstack([region_props[i].coords for i in group_indices])
        min_row, min_col = np.min(group_coords, axis=0)
        max_row, max_col = np.max(group_coords, axis=0)
        grouped_regions.append((min_row, min_col, max_row, max_col))

    # 找到面积最大的分组
    largest_group_index = np.argmax(total_areas)
    largest_region = grouped_regions[largest_group_index]
    min_row, min_col, max_row, max_col = largest_region

    # 计算面积最大的区域的中心
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2

    # 计算正方形的边界
    half_side = box_size // 2
    start_row = max(0, center_row - half_side)
    end_row = min(matrix.shape[0], center_row + half_side)
    start_col = max(0, center_col - half_side)
    end_col = min(matrix.shape[1], center_col + half_side)

    return start_row, end_row, start_col, end_col


def remove_raw(data, bg):
    data = data - bg  # 减背景 30*110
    data[np.where(data < 0)] = 0
    data -= np.average(data, axis=0)  # 减平均值
    data[np.where(data < 0)] = 0
    # new_data = cv2.resize(np.flip(new_data, axis=0), (384, 192))[np.newaxis, :, :]
    new_data = np.flip(data, axis=0)  # 沿行翻转数据
    new_data = np.array(new_data, dtype=np.float32)
    return new_data


def remove_dop(data):
    data -= np.average(data, axis=0)
    data[np.where(data < 0)] = 0
    # new_data = cv2.resize(np.flip(new_data, axis=0), (384, 192))[np.newaxis, :, :]
    new_data = np.flip(data, axis=0)  # 沿行翻转数据
    new_data = np.array(new_data, dtype=np.float32)
    return new_data


def find_center(new_pose):
    valid_coordinates = new_pose[~np.all(new_pose == [99999, 99999, 99999], axis=1)]
    if len(valid_coordinates) == 0:
        center = 0
    else:
        x = round(np.mean(valid_coordinates[:, 0]) / 1000)
        z = round(np.mean(valid_coordinates[:, 2]) / 1000)
        distance = math.sqrt(x * x + z * z)
        center = round(distance / 0.15)
        center = min(center, 109)
        # a = 14 / 110
        # convert_x = []
        # for i in range(len(valid_coordinates)):
        #     x = math.sqrt(valid_coordinates[i][0] ** 2 + valid_coordinates[i][2] ** 2) / 1000 / a
        #     convert_x.append([x])
        # x = np.mean(valid_coordinates[:, 0])
        # y = np.mean(valid_coordinates[:, 1])
        # z = np.mean(valid_coordinates[:, 2])
        # center_x = round(math.sqrt(x**2 + z**2)/1000/a)
        # center_x = round(np.mean(convert_x))
    return center


def process_heatmap(box_raw, img_raw, boundary_left, box_size):
    m, n = box_raw.shape[0], box_raw.shape[1]
    index = int(box_raw.argmax())
    center_x = int(index / n)  # Row
    center_z = index % n  # Column
    x = center_x
    z = center_z + boundary_left

    x_left = max(0, z - box_size)
    x_right = min(z + box_size, img_raw.shape[1])
    z_top = max(0, x - box_size)
    z_bottom = min(img_raw.shape[0], x + box_size)
    return x_left, x_right, z_top, z_bottom


def pad_image_to_110(img):
    top_padding = 40
    bottom_padding = 40
    padded_img = np.pad(img, ((top_padding, bottom_padding), (0, 0)), mode='constant', constant_values=0)
    return padded_img


def reshape50(img, z_top, z_bottom, x_left, x_right):
    # 计算目标区域的大小
    target_height = z_bottom - z_top
    target_width = x_right - x_left
    
    # 创建一个 50x50 的空白图像
    new_img = np.zeros((40, 40), dtype=img.dtype)  # 如果图像是彩色的，可能需要使用 img.shape[2] 作为通道数
    
    # 计算中央位置
    center_x = (40 - target_width) // 2
    center_y = (40 - target_height) // 2
    
    # 确保目标区域可以放置在新图像内
    new_img[center_y:center_y+target_height, center_x:center_x+target_width] = img[z_top:z_bottom, x_left:x_right]
    
    return new_img


def convert(w, h, angle):
    angle_list = np.arange(-angle, angle + 2 * angle / w, 2 * angle / w)
    if w % 2 == 0:
        angle_list = np.delete(angle_list, w // 2)
    else:
        angle_list = np.delete(angle_list, (w // 2, w // 2 + 1))
        angle_list = np.insert(angle_list, w // 2, 0)
    angle_list = np.reshape(angle_list, (w, 1))
    range_index = np.reshape(np.arange(0, h), (1, h))
    index_x = range_index * np.sin(np.deg2rad(angle_list))
    index_x = np.rint(index_x).astype(int)
    index_y = range_index * np.cos(np.deg2rad(angle_list))
    index_y = np.rint(index_y).astype(int)
    return index_x, index_y


class Convert():
    def __init__(self, max_range, cut_off):
        self.max_range = max_range
        self.cut_off = cut_off
        self.frq_len = self.max_range - self.cut_off
        self.x = np.arange(0, self.frq_len)
        self.y = (200 - self.x)
        self.out_im = np.zeros((self.frq_len * 2, self.frq_len))
        self.X, self.Y = convert(1500, self.frq_len, 90)

    def __call__(self, heatmap):
        self.out_im[(self.X + self.frq_len, self.Y)] = cv2.resize(heatmap, (self.frq_len, 1500))
        dikaer = self.out_im[self.frq_len * 3 // 5: - self.frq_len * 3 // 5, :]
        return dikaer


def npcv(img, v, cv_w, cv_h, rot_k, T=False):
    data = img.copy()
    data[np.where(data < 0)] = 0
    data[np.where(data > v)] = v
    data = np.rint((data / v) * 255)
    data = np.array(data, dtype='uint8')
    if T:
        data = data.T
    data = cv2.resize(np.rot90(data, rot_k), (cv_w, cv_h))
    data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    data = cv2.applyColorMap(data, cv2.COLORMAP_JET)

    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
    plt.show()


def show(img1,img2):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Matrix')
    plt.imshow(img1, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Detection with Borders')
    plt.imshow(img2, cmap='gray')
    plt.show()


class Load_data():
    def __init__(self, args, mode):
        self.args = args
        if mode == 'train':
            self.root = args.train_path
        else:
            self.root = args.train_path

        self.root_dir = acquire_rootdir(self.root)
        self.box_size = args.box_size
        self.boundary_size = args.boundary_size

        # 多普勒热图路径
        self.dop_h_path, self.dop_v_path = [], []
        # 水平、垂直热图、标签路径
        self.h_path, self.label_path, self.index = [], [], []
        self.h_crop_path, self.label_crop_path, = [], [],
        num_file = 0

        self.convert = Convert(156, 46)
        self.split_num2 = 3600
        self.split_num1 = 400
        
        # 参数设置
        self.num_train = 5 # 训练单元数（假设为正方形的边长）
        self.num_guard = 2  # 保护单元数（假设为正方形的边长）
        self.rate_fa = 0.0001  # 虚警概率

        file_num_name = "/root/projects/JJW_Model/contrast/load_dataset/action9mini.json"
        with open(file_num_name, 'r') as json_file:
            all_num_file = json.load(json_file)

        for i, dir in enumerate(self.root_dir):
            if dir in all_num_file:
                num_root_dir_split = self.root_dir[i].split('/')
                file_num_people = int(num_root_dir_split[-2])

                h_directory = dir + "/pre_rf"  # 行为动作数据文件
                label_directory = dir + "/new_label"  # 定位标签文件目录路径

                # 获取 post_rf 和 location 子目录中的文件列表，并按文件名的数值大小进行排序
                hv_dir = os.listdir(h_directory)
                hv_dir.sort(key=lambda x: int(x[:-4]))  # 按字符串的最后4位排序
                label_dir = os.listdir(label_directory)
                label_dir.sort(key=lambda x: int(x[:-4]))
                
                if mode == 'train':
                    hv_dir = hv_dir[self.split_num1:self.split_num2]
                    label_dir = label_dir[self.split_num1:self.split_num2]
                    if file_num_people in (1, 2, 3):
                        for num in range(file_num_people):
                            for k, h in enumerate(hv_dir):
                                self.h_path.append(h_directory + "/" + h)  # pose_rf文件中的所有数据
                                if k <= len(hv_dir) - self.args.time_patch:
                                    self.h_crop_path.append(h_directory + "/" + h)  # 待裁剪文件的所有数据
                            for k, label in enumerate(label_dir):
                                self.label_path.append(label_directory + "/" + label)  # location文件中的所有数据
                                if k <= len(label_dir) - self.args.time_patch:
                                    self.label_crop_path.append(label_directory + "/" + label)
                                    self.index.append(num)
                else:
                    hv_dir1 = hv_dir[:self.split_num1]
                    label_dir1 = label_dir[:self.split_num1]
                    hv_dir2 = hv_dir[self.split_num2:]
                    label_dir2 = label_dir[self.split_num2:]
                    if file_num_people in (1, 2, 3):
                        for num in range(file_num_people):
                            for k, h in enumerate(hv_dir1):
                                self.h_path.append(h_directory + "/" + h)  # pose_rf文件中的所有数据
                                if k <= len(hv_dir1) - self.args.time_patch:
                                    self.h_crop_path.append(h_directory + "/" + h)  # 待裁剪文件的所有数据
                            for k, label in enumerate(label_dir1):
                                self.label_path.append(label_directory + "/" + label)  # location文件中的所有数据
                                if k <= len(label_dir1) - self.args.time_patch:
                                    self.label_crop_path.append(label_directory + "/" + label)
                                    self.index.append(num)

                            for k, h in enumerate(hv_dir2):
                                self.h_path.append(h_directory + "/" + h)  # pose_rf文件中的所有数据
                                if k <= len(hv_dir2) - self.args.time_patch:
                                    self.h_crop_path.append(h_directory + "/" + h)  # 待裁剪文件的所有数据
                            for k, label in enumerate(label_dir2):
                                self.label_path.append(label_directory + "/" + label)  # location文件中的所有数据
                                if k <= len(label_dir2) - self.args.time_patch:
                                    self.label_crop_path.append(label_directory + "/" + label)
                                    self.index.append(num)
                                        
    # 每个循环索引表示 4 帧数据,将返回 h_crop_path 列表长度除以 4
    def __len__(self):
        return len(self.h_crop_path)//2

    def __getitem__(self, idx):
        idx *= 2            
        filename = self.h_crop_path[idx]
        parts = filename.split('/')
        desired_path = '/'.join(parts[:-2])
        background_dir = os.path.join(desired_path, 'average_bg.npy')
        background = np.load(background_dir, allow_pickle=True)  # 加载背景数据

        if len(background) == 2:
            bg_h = background[0]
            bg_v = background[1]
        else:
            bg_h = background[4]
            bg_v = background[5]

        filename_idx = self.h_path.index(filename)  # 找到当前filename在h_path列表中的索引

        # pose_rf文件中的热图数据处理
        # self.joints = []
        raw_h, raw_v, dop_h, dop_v = [], [], [], []
        action, action_list = [], []
        raw_h_list, raw_v_list, dop_h_list, dop_v_list = [], [], [], []
        most_action = 0

        for i in range(self.args.time_patch):  # 一次性加载时序数据
            data = np.load(self.h_path[filename_idx + i], allow_pickle=True)  # 加载水平数据
            if len(data) == 4:
               data0=data[0]
               data1=data[1]
               data2=data[2]
               data3=data[3]
            else:
               data0=data[4].real
               data1=data[5].real
               data2=data[6].real
               data3=data[7].real

            mask_h = np.zeros((30, 110))  # 掩码
            mask_v = np.zeros((30, 110))

            r_h = remove_raw(data0, bg_h).copy()  
            img_raw_h = r_h

            r_v = remove_raw(data1, bg_v).copy()  
            img_raw_v = r_v

            d_h = remove_dop(data2).copy()
            img_dop_h = d_h

            d_v = remove_dop(data3).copy()
            img_dop_v = d_v

            action_mapping = {9:2}
            label_dir = self.label_path[filename_idx + i]
            label = np.load(label_dir, allow_pickle=True).tolist()  # (1,5)
            
            # 获取每一帧action
            people = label['people'][self.index[idx]]
            if 'new_action' in people:
                new_action = people['new_action']
                mapped_action = action_mapping.get(new_action, new_action)
                action.append(mapped_action)
                # action.append(new_action)
            else:
                continue
           
            # 获取对应的人的pose，计算center
            pose = label['people'][self.index[idx]]['pose']
            center = find_center(pose)

            # 根据距离确定目标范围
            boundary_left = int(np.round(max(0, center - self.boundary_size)))
            boundary_right = int(np.round(min(img_raw_h.shape[1], center + self.boundary_size)))

            if boundary_left <= boundary_right:
                box_raw_h = img_raw_h[:, boundary_left:boundary_right]  # (30, 20)
                box_raw_v = img_raw_v[:, boundary_left:boundary_right]
            else:
                # 处理空序列的情况，例如输出错误信息或采取其他操作
                # 不进行中心点校准
                box_raw_h = img_raw_h[:, center]
                box_raw_v = img_raw_v[:, center]


            # 运行SO-CFAR算法,保留目标
            img_raw_h_det = so_cfar_2d(img_raw_h, self.num_train, self.num_guard, self.rate_fa, boundary_left, boundary_right)
            z_top_h, z_bottom_h, x_left_h, x_right_h = retain_largest_detections(img_raw_h, img_raw_h_det, self.box_size, 1, 5)
            # 运行SO-CFAR算法,保留目标
            img_raw_v_det = so_cfar_2d(img_raw_v, self.num_train, self.num_guard, self.rate_fa, boundary_left, boundary_right)
            z_top_v, z_bottom_v, x_left_v, x_right_v = retain_largest_detections(img_raw_v, img_raw_v_det, self.box_size, 1, 5)

            if z_top_h == 0 and z_bottom_h == 0 and x_left_h == 0 and x_right_h == 0:
                x_left_h, x_right_h, z_top_h, z_bottom_h = process_heatmap(box_raw_h, img_raw_h, boundary_left, self.box_size)
            if z_top_v == 0 and z_bottom_v == 0 and x_left_v == 0 and x_right_v == 0:
                x_left_v, x_right_v, z_top_v, z_bottom_v = process_heatmap(box_raw_v, img_raw_v, boundary_left, self.box_size)

             
            mask_h[z_top_h:z_bottom_h, x_left_h:x_right_h] = 1
            mask_v[z_top_v:z_bottom_v, x_left_v:x_right_v] = 1

            img_raw_h_end = img_raw_h.copy() * mask_h
            img_dop_h_end = img_dop_h.copy() * mask_h
            img_raw_v_end = img_raw_v.copy() * mask_v
            img_dop_v_end = img_dop_v.copy() * mask_v


            # img_raw_h_end = cv2.resize(img_raw_h_end, (50, 50))
            # img_raw_v_end = cv2.resize(img_raw_v_end, (50, 50))
            # img_dop_h_end = cv2.resize(img_dop_h_end, (50, 50))
            # img_dop_v_end = cv2.resize(img_dop_v_end, (50, 50))

            # img_raw_h_end = cv2.resize(img_raw_h_end, (256, 32))
            # img_raw_v_end = cv2.resize(img_raw_v_end, (256, 32))
            # img_dop_h_end = cv2.resize(img_dop_h_end, (256, 32))
            # img_dop_v_end = cv2.resize(img_dop_v_end, (256, 32))

            
            img_raw_h_end = cv2.resize(img_raw_h_end, (64, 64))
            img_raw_v_end = cv2.resize(img_raw_v_end, (64, 64))
            img_dop_h_end = cv2.resize(img_dop_h_end, (64, 64))
            img_dop_v_end = cv2.resize(img_dop_v_end, (64, 64))

            # img_raw_h_end = cv2.resize(img_raw_h_end, (224, 224))
            # img_raw_v_end = cv2.resize(img_raw_v_end, (224, 224))
            # img_dop_h_end = cv2.resize(img_dop_h_end, (224, 224))
            # img_dop_v_end = cv2.resize(img_dop_v_end, (224, 224))
            

            raw_h_list.append(img_raw_h_end)
            raw_v_list.append(img_raw_v_end)
            dop_h_list.append(img_dop_h_end)
            dop_v_list.append(img_dop_v_end)

        # location文件中的动作标签
        label_confidence = False

        if action:  # 确保 action 列表不为空
            most_action = Counter(action).most_common(1)[0][0]
        if len(set(action)) == 1:
            label_confidence = True

        raw_h_array = np.array(raw_h_list, dtype=np.float32)  # (60, 50, 50)
        raw_v_array = np.array(raw_v_list, dtype=np.float32)  # (60, 50, 50)
        dop_h_array = np.array(dop_h_list, dtype=np.float32)  # (60, 50, 50)
        dop_v_array = np.array(dop_v_list, dtype=np.float32)  # (60, 50, 50)
        # 添加新的维度
        raw_h = torch.from_numpy(raw_h_array[np.newaxis, :, :, :])  # (1, 60, 50, 50)
        raw_v = torch.from_numpy(raw_v_array[np.newaxis, :, :, :])  # (1, 60, 50, 50)
        dop_h = torch.from_numpy(dop_h_array[np.newaxis, :, :, :])  # (1, 60, 50, 50)
        dop_v = torch.from_numpy(dop_v_array[np.newaxis, :, :, :])  # (1, 60, 50, 50)

        return raw_h, raw_v, dop_h, dop_v, most_action, label_confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_argparser()])
    args = parser.parse_args()

    dataset_train = Load_data(args, mode='train')
    # dataset_val = Load_data(args, mode='test')
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)
    # vailloader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
    #                                          num_workers=args.num_workers, pin_memory=True)

    print("训练集数据加载完毕", len(dataset_train))
    # print("验证集数据加载完毕", len(dataset_val))

    for idx in range(len(dataset_train)):
        sample = dataset_train[idx]  # 调用 __getitem__ 方法获取单个样本
        break
# def setup_logging(log_file):
#     logging.basicConfig(filename=log_file, level=logging.ERROR,
#                         format='%(asctime)s - %(levelname)s - %(message)s')


# if __name__ == '__main__':

#     # 初始化日志系统
#     log_file = '/root/projects/JJW/ConvNeXt/logs/data_loading_errors.log'
#     setup_logging(log_file)

#     parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_argparser()])
#     args = parser.parse_args()

#     dataset_train = Load_data(args, mode='train')
#     dataset_val = Load_data(args, mode='test')
#     trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
#                                               num_workers=args.num_workers, pin_memory=True)
#     vailloader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
#                                              num_workers=args.num_workers, pin_memory=True)

#     print("训练集数据加载完毕", len(dataset_train))
#     print("验证集数据加载完毕", len(dataset_val))

#     for idx in range(len(dataset_train)):
#         try:
#             sample = dataset_train[idx]  # 调用 __getitem__ 方法获取单个样本
#         except Exception as e:
#             logging.error(f"Error loading training data at index {idx}: {e}")
#             continue

#     # 测试验证集数据加载

#     for idx in range(len(dataset_val)):
#         try:
#             sample = dataset_val[idx]
#         except Exception as e:
#             logging.error(f"Error loading validation data at index {idx}: {e}")
#             continue

#     print("训练和验证数据测试完成")