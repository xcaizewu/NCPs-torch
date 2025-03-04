import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from convolution_head import ConvolutionHead  # 假设 ConvolutionHead 已经实现
from augmentation_utils import reduce_mean_mse_with_exp_weighting  # 假设该函数已经实现
import wormflow3 as wf  # 假设 wormflow3 已经实现

class End2EndWormPilot(nn.Module):

    def __init__(self, wm_size, conv_grad_scaling, learning_rate, curve_factor, ode_solver_unfolds=None):
        super(End2EndWormPilot, self).__init__()
        self.learning_rate = learning_rate
        self.image_width = 200
        self.image_height = 78
        self.conv_grad_scaling = conv_grad_scaling
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 卷积头
        self.conv = ConvolutionHead(num_filters=8, features_per_filter=4)
        self.conv.to(device)

        # 选择 Wormnet 架构
        if wm_size == "fully":
            print("Using fully connected NPC network")
            architecture = wf.FullyConnectedWormnetArchitecture(1, num_units=19)
        elif wm_size == "rand":
            print("Using random NPC network")
            architecture = wf.RandomWormnetArchitecture(1, num_units=19, sensory_density=6, inter_density=4, motor_density=6, seed=20190120, input_size=None)
        else:
            print("Using designed NPC network")
            architecture = wf.CommandLayerWormnetArchitectureMK2(1, num_interneurons=12, num_command_neurons=6, sensory_density=6, inter_density=4, recurrency=6, motor_density=6, seed=20190120, input_size=None)

        # 初始化 WormnetCell
        self.wm = wf.WormnetCell(architecture)
        self.wm.to(device)
        if ode_solver_unfolds is not None:
            self.wm._ode_solver_unfolds = ode_solver_unfolds
        self.wm._output_mapping = wf.MappingType.Linear

        # 损失函数
        self.curve_factor = curve_factor

        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x, rnn_init_state=None):
        # 卷积特征提取
        feature_layer = self.conv(x)  # [bt, sq, 32]

        # 初始化 RNN 状态
        if rnn_init_state is None:
            rnn_init_state = torch.zeros(x.size(0)*x.size(1), self.wm.state_size, device=x.device)

        # RNN 前向传播
        y, final_state = self.wm(feature_layer, rnn_init_state, b=x.size(0), s=x.size(1))

        # 获取 sensory neurons
        sensory_neurons = self.wm._map_inputs(feature_layer, reuse_scope=True)

        return y, final_state, sensory_neurons

    def get_saliency_map(self, x):
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0)
        return self.conv.visual_backprop(x)

    def inference_features(self, x):
        with torch.no_grad():
            feature_layer = self.conv(x)
        return feature_layer

    def evaluate_single_sequence(self, x, y, init_state=None):
        if init_state is None:
            init_state = torch.zeros(1, self.wm.state_size, device=x.device)
        seq_len = x.size(0)
        y = y.view(1, seq_len, 1)
        x = x.view(1, seq_len, x.size(1), x.size(2), x.size(3))

        with torch.no_grad():
            y_pred, final_state, _ = self.forward(x, init_state)
            loss = torch.mean((y - y_pred) ** 2)
            mae = torch.mean(torch.abs(y - y_pred))

        return loss.item(), mae.item(), final_state

    def replay_internal_state(self, x, init_state):
        init_state = init_state.view(1, self.wm.state_size)
        x = x.view(1, 1, x.size(0), x.size(1), x.size(2))

        with torch.no_grad():
            sensory_neurons, y_pred, final_state = self.forward(x, init_state)

        return sensory_neurons.flatten(), y_pred.flatten(), final_state.flatten()

    def train_iter(self, batch_x, batch_y):

        self.optimizer.zero_grad()
        # 前向传播
        y_pred, final_state, _ = self.forward(batch_x)

        # 计算损失
        mse_loss = nn.MSELoss()(y_pred, batch_y)
        mae = nn.L1Loss()(y_pred, batch_y)
        # loss = reduce_mean_mse_with_exp_weighting(y_pred, batch_y, exp_factor=self.curve_factor)
        # mae = torch.mean(torch.abs(y_pred - batch_y))

        # 反向传播
        mse_loss.backward()

        # # 梯度裁剪
        # for param in self.parameters():
        #     # if param.grad is not None and "perception" in param.name:
        #     if param.grad is not None:
        #         param.grad *= self.conv_grad_scaling
        # for name, param in self.named_parameters():
        #     if "conv" in name and param.grad is not None:
        #         param.grad *= self.conv_grad_scaling
        torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 10)
        # # 参数约束
        self.wm.get_param_constrain_op()

        # 更新参数
        self.optimizer.step()



        return mse_loss.item(), mae.item()

    def save_checkpoint(self, path, name='model'):
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, f'{name}.pth')
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(os.path.join(path, 'model.pth'))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])