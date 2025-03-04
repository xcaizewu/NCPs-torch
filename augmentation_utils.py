import numpy as np
import cv2
import torch
import torch.nn.functional as F


def reduce_mean_mse_with_exp_weighting(y_hat, y_target, exp_factor=0):
    """
    计算加权均方误差损失。
    参数:
        y_hat: 预测值，shape 为 [batch_size, ...]
        y_target: 目标值，shape 为 [batch_size, ...]
        exp_factor: 加权因子
    返回:
        加权均方误差损失
    """
    assert exp_factor >= 0.0
    weights = torch.exp(torch.abs(y_target) * exp_factor)
    error = y_hat - y_target
    return torch.sum(weights * torch.square(error)) / torch.sum(weights)


def adjust_gamma(images, gamma=1.0):
    """
    调整图像的 gamma 值。
    参数:
        images: 输入图像，shape 为 [H, W, C] 或 [batch_size, H, W, C]
        gamma: gamma 值
    返回:
        调整后的图像
    """
    return torch.pow(images, gamma)


def draw_shadow(img, thickness=10, blur=3, angle=np.pi / 2, offset_x=0, offset_y=0, gamma=1.5):
    """
    在图像上绘制阴影。
    参数:
        img: 输入图像，shape 为 [H, W, C]
        thickness: 阴影厚度
        blur: 阴影模糊程度
        angle: 阴影角度
        offset_x: 阴影水平偏移
        offset_y: 阴影垂直偏移
        gamma: 阴影 gamma 值
    返回:
        带阴影的图像
    """
    mask = np.zeros([img.shape[0], img.shape[1], 1])

    r = 200
    point1 = (int(r * np.cos(angle) + offset_x + img.shape[1] / 2), int(r * np.sin(angle) + offset_y + img.shape[0] / 2))
    point2 = (int(-r * np.cos(angle) + offset_x + img.shape[1] / 2), int(-r * np.sin(angle) + offset_y + img.shape[0] / 2))
    cv2.line(mask, point1, point2, (1.0), thickness)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    mask = mask.reshape([img.shape[0], img.shape[1], 1])

    img2 = adjust_gamma(torch.from_numpy(np.copy(img)), gamma).numpy()
    img_merged = mask * img2 + (1.0 - mask) * img
    return img_merged


if __name__ == "__main__":
    # 读取图像并归一化
    img = cv2.imread('images/crop_00.png').astype(np.float32) / 255.0
    print("img shape: ", str(img.shape))
    cv2.imwrite('images/save.png', img * 255)

    # 添加噪声
    noise_table = [0.1, 0.15, 0.2]
    for i in range(len(noise_table)):
        img_noise = np.clip(img + np.random.normal(loc=0, scale=noise_table[i], size=img.shape), 0, 1)
        cv2.imwrite('images/noise_{:02d}.png'.format(i), img_noise * 255)

    # 绘制阴影
    for i in range(20):
        thickness = np.random.randint(10, 100)
        kernel_sizes = [3, 5, 7]
        blur = kernel_sizes[np.random.randint(0, len(kernel_sizes))]
        angle = np.random.uniform(low=0, high=np.pi)
        offset_x = np.random.randint(-100, 100)
        offset_y = np.random.randint(-30, 30)
        gamma = np.random.uniform(1, 2)
        do_darken = np.random.rand() > 0.33  # 2/3 darker, 1/3 lighter
        if not do_darken:
            print("light")
            gamma = 1.0 / gamma
        else:
            print("dark")
        img_merged = draw_shadow(img, thickness, blur, angle, offset_x, offset_y, gamma)
        cv2.imwrite("images/shadowed_{:02d}.png".format(i), np.clip(255 * img_merged, 0, 255))