import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import argparse
# from models.cnn_model import CnnModel
from e2e_worm_pilot import End2EndWormPilot
# from models.e2e_lstm import End2EndLSTMPilot
# from models.e2e_rnn import UniversalRNNPilot
from active_data_provider import ActiveDataProvider


def evaluate_on_validation(model, data_provider, max_seq_len):
    """
    在验证集上评估模型性能。
    参数:
        model: 模型
        data_provider: 数据提供器
        max_seq_len: 最大序列长度
    返回:
        平均损失和平均绝对误差
    """
    model.eval()
    losses = []
    abs_errors = []
    rnn_state = None

    with torch.no_grad():
        for x, y, frameskip in data_provider.iterate_as_single_sequence(max_seq_len):
            # 如果有帧跳跃，重置 RNN 状态
            if frameskip:
                rnn_state = None

            # 评估单个序列
            loss, mae, rnn_state = model.evaluate_single_sequence(x, y, rnn_state)
            losses.append(loss)
            abs_errors.append(mae)

    return np.mean(losses), np.mean(abs_errors)


# 解析命令行参数
parser = argparse.ArgumentParser(description='Train and test model with passive test data')
parser.add_argument('--debug', action='store_true', help='Debug Flag')

training_files = [
    "/home/user/8T/science_robotics_code/training_data_active/20190628-094335_blue_prius_devens_rightside.h5",
    "/home/user/8T/science_robotics_code/training_data_active/20190723-133449_blue_prius_devens_rightside.h5",
    "/home/user/8T/science_robotics_code/training_data_active/20190723-161821_blue_prius_devens_rightside.h5",
    "/home/user/8T/science_robotics_code/training_data_active/20190723-154501_blue_prius_devens_rightside.h5",
]
validation_files = [
    "/home/user/8T/science_robotics_code/training_data_active/20190628-150233_blue_prius_devens_rightside.h5",
    "/home/user/8T/science_robotics_code/training_data_active/20190723-134708_blue_prius_devens_rightside.h5",
]

# 超参数
parser.add_argument('--lr', default=0.0008, type=float, help='Learning Rate (default: 0.0005)')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
parser.add_argument('--conv_grad_scaling', default=1.0, type=float, help='Scaling factor of the convolution layer gradients (RNNs only)')
parser.add_argument('--epochs', type=int, default=800, help='Number of training epochs')
parser.add_argument('--curve_factor', default=0.0, type=float, help='Factor in the exponential term of the sample weighting (0 means no weighting at all)')
parser.add_argument('--shadow_gamma', default=0.0, type=float, help='Maximum value of gamma distortion of the shadow augmentation')
parser.add_argument('--darkening_ratio', default=0.66, type=float, help='Ratio of show many shadows are darkened vs lightened')

# 模型参数
parser.add_argument('--new', action='store_true', help='Overwrites existing base_path')
parser.add_argument('--restore', action='store_true', help='Continues training an existing session')
parser.add_argument('--base_path', default='session', help='Base path to store the sessions')
parser.add_argument('--model', default='None', help='Type of model. Options: cnn, e2e_wm and e2e_lstm')

# RNN 参数
parser.add_argument('--seq_len', type=int, default=16, help='Sequence length')

# 模型特定参数
parser.add_argument('--drf', default=0.5, type=float, help='Dropout keep_prob for flattened layer (default: 0.5)')
parser.add_argument('--dr1', default=0.5, type=float, help='Dropout keep_prob for first FC layer  (default: 0.5)')
parser.add_argument('--dr2', default=0.7, type=float, help='Dropout keep_prob for second FC layer (default: 0.7)')
parser.add_argument('--lstm_size', type=int, default=64, help='Number of LSTM cells')
parser.add_argument('--lstm_clip', type=float, default=10, help='Clip LSTM memory values')
parser.add_argument('--lstm_forget_bias', type=float, default=1.0, help='Forget bias of LSTM cell')
parser.add_argument('--wm_size', default='mk2', help='Use denser wormnet architecture')

# 日志记录
parser.add_argument('--log_period', type=int, default=1, help='Log period for evaluating validation performance')

# 解析参数
args = parser.parse_args()
if args.model not in ['cnn', 'e2e_lstm', 'e2e_wm', 'e2e_ctrnn']:
    raise ValueError('Unknown model type: ' + str(args.model))

# 基础路径
base_path = os.path.join("active_sessions", f"{args.model}_{args.base_path}")
training_history_log = os.path.join(base_path, f"train_{args.model}_{args.base_path}.csv")

if os.path.exists(base_path):
    if args.restore:
        raise NotImplementedError("Continuing a session is currently not implemented")
    # elif not args.new:
    #     raise ValueError('Session directory already exists, but neither --restore nor --new command line options were specified!')
else:
    os.makedirs(base_path)

# 创建训练日志文件
with open(training_history_log, 'w') as f:
    f.write("epoch; train_loss; train_mae; test_loss; test_mae\n")

# 加载数据
print("Loading data ...")
train_data_provider = ActiveDataProvider(
    h5_files=training_files,
    shadow_max_gamma=args.shadow_gamma,
    shadow_darkening_ratio=args.darkening_ratio,
    debug_flag=args.debug
)
train_data_provider.summary(set_name="Training set")

valid_data_provider = ActiveDataProvider(
    h5_files=validation_files,
    debug_flag=args.debug,
    mode='val'
)
valid_data_provider.summary(set_name="Validation set")

is_rnn_model = args.model in ["e2e_lstm", "e2e_wm", "e2e_ctrnn"]

# 初始化模型
if args.model == 'cnn':
    model = CnnModel(
        learning_rate=args.lr,
        curve_factor=args.curve_factor,
        drf=args.drf,
        dr1=args.dr1,
        dr2=args.dr2
    )
elif args.model == 'e2e_wm':
    model = End2EndWormPilot(
        wm_size=args.wm_size,
        conv_grad_scaling=args.conv_grad_scaling,
        learning_rate=args.lr,
        curve_factor=args.curve_factor,
    )
elif args.model == 'e2e_lstm':
    model = End2EndLSTMPilot(
        lstm_size=args.lstm_size,
        conv_grad_scaling=args.conv_grad_scaling,
        learning_rate=args.lr,
        curve_factor=args.curve_factor,
        clip_value=args.lstm_clip,
        forget_bias=args.lstm_forget_bias,
    )
elif args.model == "e2e_ctrnn":
    model = UniversalRNNPilot(
        num_units=args.lstm_size,
        conv_grad_scaling=args.conv_grad_scaling,
        learning_rate=args.lr,
        curve_factor=args.curve_factor,
        clip_value=args.lstm_clip,
        rnn_type="ctrnn",
        ctrnn_global_feedback=True,
    )


# 训练循环
print('Entering training loop')
for epoch in range(args.epochs):
    model.train()

    # 在每个 log_period 或最后一个 epoch 评估验证集
    if epoch % args.log_period == 0 or epoch == args.epochs - 1:
        checkpoint_dir = os.path.join(base_path, "checkpoints", f"epoch_{epoch:03d}")
        model.save_checkpoint(checkpoint_dir)

        # 评估验证集
        test_loss, test_mae = evaluate_on_validation(model, valid_data_provider, max_seq_len=args.seq_len)

        # 如果是 wormnet 模型，导出参数
        if args.model == "e2e_wm":
            dump_dir = os.path.join(base_path, "checkpoints", f"epoch_{epoch:03d}", "wm_dump")
            model.wm.export_parameters(dump_dir)
        elif args.model == "e2e_lstm":
            # 导出 LSTM 参数
            lstm_w, lstm_b = model.fused_cell.get_weights()
            _, _, forget_w, _ = np.split(lstm_w, indices_or_sections=4, axis=1)
            _, _, forget_b, _ = np.split(lstm_b, indices_or_sections=4, axis=0)
            np.savetxt(os.path.join(base_path, "checkpoints", f"epoch_{epoch:03d}", "forget_bias.csv"), forget_b)
            np.savetxt(os.path.join(base_path, "checkpoints", f"epoch_{epoch:03d}", "forget_w.csv"), forget_w)

    train_losses = []
    train_abs_errors = []

    if is_rnn_model:
        # 训练 RNN 模型
        for e in range(train_data_provider.count_epoch_size(args.batch_size, args.seq_len)):
            batch_x, batch_y = train_data_provider.create_sequenced_batch(args.batch_size, args.seq_len)
            loss, abs_err = model.train_iter(batch_x, batch_y)
            train_losses.append(loss)
            train_abs_errors.append(abs_err)
    else:
        # 训练 CNN 模型
        for batch_x, batch_y in train_data_provider.iterate_shuffled_train(args.batch_size):
            loss, abs_err = model.train_iter(batch_x, batch_y)
            train_losses.append(loss)
            train_abs_errors.append(abs_err)

    # 记录训练和验证损失
    if epoch % args.log_period == 0 or epoch == args.epochs - 1:
        train_loss = np.mean(train_losses)
        train_mae = np.mean(train_abs_errors)
        with open(training_history_log, 'a') as f:
            f.write(f"{epoch}; {train_loss}; {train_mae}; {test_loss}; {test_mae}\n")
        print(f'Metrics after {epoch} epochs, train loss: {train_loss:.2f}, train mae: {train_mae:.2f}, test loss: {test_loss:.2f}, test mae: {test_mae:.2f}')

# 训练结束，保存最终模型
model.save_checkpoint(os.path.join(base_path, "checkpoints", "final"))
test_loss, test_mae = evaluate_on_validation(model, valid_data_provider, max_seq_len=32)

# 记录最终结果
train_loss = np.mean(train_losses)
train_mae = np.mean(train_abs_errors)
with open(training_history_log, 'a') as f:
    f.write(f"{args.epochs}; {train_loss}; {train_mae}; {test_loss}; {test_mae}\n")
print(f'Metrics after all {args.epochs} epochs, train loss: {train_loss:.2f}, train mae: {train_mae:.2f}, test loss: {test_loss:.2f}, test mae: {test_mae:.2f}')