import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class ConvolutionHead(nn.Module):

    def __init__(self, num_filters=8, features_per_filter=4):
        super(ConvolutionHead, self).__init__()
        self.num_filters = num_filters
        self.features_per_filter = features_per_filter
        self._w_value_list = None

        # 定义卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(36, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, self.num_filters, kernel_size=3, stride=1, padding=1)
        ])

        self.filter_out_width = 25
        self.filter_out_height = 10

        # 定义全连接层
        self.dense_layers = nn.ModuleList([
            nn.Linear(self.filter_out_width * self.filter_out_height, self.features_per_filter)
            for _ in range(self.num_filters)
        ])

    def forward(self, x):
        # TODO
        image_height = int(x.shape[3])
        image_width = int(x.shape[4])
        image_channels = int(x.shape[2])

        # # 图像标准化
        head = x.contiguous().view(-1, image_channels, image_height, image_width)
        bt_sq, c, h, w = head.shape

        # 将 tensor 转换为 float 类型
        input_tensor = head.float()

        # 展平每个通道的像素值，形状变为 [batch_size*time, channels*height*width]
        flat_tensor = input_tensor.view(input_tensor.size(0), -1)

        # 计算每个图像的均值和标准差
        mean = flat_tensor.mean(dim=1, keepdim=True)
        std = flat_tensor.std(dim=1, keepdim=True)

        # 避免除以零，添加一个小的数值
        std = torch.clamp(std, min=1e-5)

        # 标准化
        standardized_tensor = (flat_tensor - mean) / std

        # 恢复原始形状
        head = standardized_tensor.view(bt_sq, c, h, w)

        # head = F.instance_norm(head.float())  # 类似于 TensorFlow 的 per_image_standardization

        # 卷积层
        self.conv_outputs = [head]
        for conv_layer in self.conv_layers:
            head = F.relu(conv_layer(head))
            # head = conv_layer(head)
            self.conv_outputs.append(head)

        # 分割卷积输出
        filter_output = torch.split(head, split_size_or_sections=1, dim=1)

        # print("Each filter output is of shape " + str(filter_output[0].shape))
        self.filter_out_width = filter_output[0].shape[2]
        self.filter_out_height = filter_output[0].shape[3]
        filter_out_flattened = int(self.filter_out_width * self.filter_out_height)

        # print("Filter out width: " + str(self.filter_out_width))
        # print("Filter out height: " + str(self.filter_out_height))
        # print("Filter out flattened: " + str(filter_out_flattened))

        # 全连接层
        feature_layer_list = []
        self._w_list = []
        for i in range(self.num_filters):
            flatten = filter_output[i].view(-1, filter_out_flattened)
            feats = self.dense_layers[i](flatten)
            self._w_list.append(self.dense_layers[i].weight)
            feature_layer_list.append(feats)

        # 拼接特征
        self.feature_layer = torch.cat(feature_layer_list, dim=1)
        # print("Feature layer shape: " + str(self.feature_layer.shape))
        total_features = int(self.feature_layer.shape[1])

        # 重塑特征层
        # feature_layer = self.feature_layer.view(x.shape[0], x.shape[1], total_features)  # # [bt, sq, 32]
        feature_layer = self.feature_layer
        return feature_layer

    def visual_backprop(self, x_value):
        if self._w_value_list is None:
            self._w_value_list = [w.detach().numpy() for w in self._w_list]

        # 获取卷积层输出和特征层输出
        A = [conv_output.detach().numpy() for conv_output in self.conv_outputs]
        feats = self.feature_layer.detach().numpy()

        aux_list = []
        means = []
        for i in range(len(A)):  # 对每个特征图
            a = A[i][0]  # 特征图 [c, h, w]
            per_channel_max = a.max(axis=1).max(axis=1)
            a /= (per_channel_max.reshape(1, -1, 1) + 0.0001)
            means.append(np.mean(a, axis=0))

        feat_act = []
        for i in range(len(self._w_value_list)):
            w_l = np.split(self._w_value_list[i], self.features_per_filter, axis=1)

            for w_i in range(len(w_l)):
                w = np.reshape(w_l[w_i], (self.filter_out_height, self.filter_out_width))
                feat_map = np.abs(w) * feats[0, i * self.features_per_filter + w_i]
                feat_act.append(feat_map)

        feat_act = np.stack(feat_act, axis=-1)
        for i in range(A[-1][0].shape[0]):
            aux_list.append(("feat_{:d}".format(i), A[-1][0][i]))

        feat_act = np.mean(feat_act, axis=-1)

        for i in range(len(means) - 2, -1, -1):
            smaller = means[i + 1]
            aux_list.append(("layer_{:d}".format(i), smaller))

            scaled_up = cv2.resize(smaller, (means[i].shape[1], means[i].shape[0]))
            means[i] = np.multiply(means[i], scaled_up)

        mask = means[0]
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        mask = np.clip(mask, 0, 1)

        return mask, aux_list