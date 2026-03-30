from typing import Callable, Union
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor, concat

ModuleType = Union[str, Callable[..., nn.Module]]
SIGMA_MIN = 0.002
SIGMA_MAX = 80
rho = 7
S_churn = 1
S_min = 0
S_max = float('inf')
S_noise = 1

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim=100,
                 gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.KLDiv = nn.KLDivLoss(reduction='batchmean')

    def __call__(self, denoise_fn, data ,labels,num_classes=None, common_feat=None):
        # 生成正态分布噪声
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        # 根据给定的均值和标准差生成sigma
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # 计算加权系数
        weight = (sigma ** 2 + self.sigma_data ** 2) / (
                sigma * self.sigma_data) ** 2
        n = torch.randn_like(data) * sigma.unsqueeze(1)

        class_errors = []
        loss_class= None
        probs=None
        if num_classes is not None:
            labels_tensor = torch.zeros((labels.shape[0], num_classes), device=data.device)
            labels_tensor[torch.arange(labels.shape[0]), labels.to(torch.int)] = 1
            class_labels = labels_tensor

            D_yn = denoise_fn(data + n, sigma, None, class_labels)
            for y in range(num_classes):
                y = torch.full((data.shape[0],), y).cuda()
                labels_tensor = torch.zeros((y.shape[0], num_classes), device=data.device)
                labels_tensor[torch.arange(y.shape[0]), y.to(torch.int)] = 1
                class_labels2 = labels_tensor
                # 生成对应类别的噪声预测
                D_yn2 = denoise_fn(data + n, sigma,None, class_labels2)

                # 计算加权误差
                error = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2 * ((D_yn2 - data) ** 2).mean(
                    dim=1)
                class_errors.append(error)

            class_errors = torch.stack(class_errors, dim=1)  # [B, C]
            probs = F.softmax(-class_errors, dim=1)
            loss_class = F.cross_entropy(probs, labels)
        else:
            D_yn = denoise_fn(data + n, sigma, common_feat, None)
        # D_yn = denoise_fn(data + n, sigma, None, common_feat)
        loss_re = weight.unsqueeze(1) * ((D_yn - data) ** 2) # 计算加权的重建损失
        reconstruction_errors = (D_yn - data) ** 2   # 计算重建误差
        score = torch.sqrt(torch.sum(reconstruction_errors, 1))# 计算每个样本的重建误差的平方和
        return loss_re,reconstruction_errors, score, D_yn,loss_class,probs


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels # 编码维度 dim_t
        self.max_positions = max_positions # 最大位置数
        self.endpoint = endpoint # 是否使用端点

    def forward(self, x):
        # 首先生成一个频率序列,按比例递增的数值
        freqs = torch.arange(start=0, end=self.num_channels // 2,
                             dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))

        # 将频率值进行缩放
        freqs = (1 / self.max_positions) ** freqs # 计算频率

        x = x.ger(freqs.to(x.dtype)) # 外积生成位置编码
        x = torch.cat([x.cos(), x.sin()], dim=1) # 拼接余弦和正弦编码
        return x


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=256, num_classes=7):
        super().__init__()

        self.dim_t = dim_t # 内部表示维度

        # self.label_embedding2 = nn.Embedding(num_classes, dim_t)
        # self.label_embedding = nn.Linear(in_features=num_classes, out_features=dim_t, bias=False)

        self.proj = nn.Linear(d_in, dim_t)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )
        # 噪声位置编码
        self.map_noise = PositionalEmbedding(num_channels=dim_t)

        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        self.feat_proj = nn.Linear(d_in, dim_t) # 特征投影
        self.head = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(dim_t, dim_t)
        )

    def forward(self, x, noise_labels, common_feat=None, y=None):
        emb = self.map_noise(noise_labels) # 获取噪声标签的嵌入
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # 调整形状并翻转

        emb = self.time_embed(emb) # 时间嵌入

        # if y is not None:
        #     # y=y.long()
        #     y_emb = self.label_embedding(y *np.sqrt(self.label_embedding.in_features)) # (B, dim_t)
        #     # y_emb = self.label_embedding(y * np.sqrt(1024))
        #     # y_emb2 = self.label_embedding2(y)
        #     emb = emb + y_emb  # 也可以选择拼接 cat([emb, y_emb], dim=1)
        #     # emb =  concat([emb, y_emb], dim=1)


        if (common_feat is None):
            x_proj = self.proj(x) +  emb # 如果没有公共特征，仅使用嵌入和投影
        else:
            x_proj = self.proj(x) + emb + self.feat_proj(common_feat) # 否则加上公共特征

        return self.mlp(x_proj) # 通过MLP处理输入
    # def forward(self, x, noise_labels, common_feat=None, y=None):
    #     emb = self.map_noise(noise_labels) # 获取噪声标签的嵌入
    #     emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # 调整形状并翻转
    #
    #     emb = self.time_embed(emb) # 时间嵌入
    #     if (common_feat is None):
    #         x = self.proj(x) + emb # 如果没有公共特征，仅使用嵌入和投影
    #     else:
    #         x = self.proj(x) + emb + self.feat_proj(common_feat) # 否则加上公共特征
    #     return self.mlp(x) # 通过MLP处理输入

class Precond(nn.Module):
    def __init__(self,
                 denoise_fn,
                 hid_dim,
                 sigma_min=0,
                 sigma_max=float('inf'),
                 sigma_data=0.5,
                 ):
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn  # 使用的去噪函数

    def forward(self, x, sigma, common_feat=None,y=None):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (
                sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x  # 输入数据缩放
        F_x = self.denoise_fn_F(x_in.to(dtype), c_noise.flatten(), common_feat,y) # 去噪处理

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)# 输出与去噪结果的结合
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class Model(nn.Module):
    def __init__(self, denoise_fn,hid_dim, P_mean=-1.2, P_std=1.2,
                 sigma_data=0.5, gamma=5, opts=None, pfgmpp=False):
        super().__init__()

        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.P_mean=P_mean
        self.P_std = P_std
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim,   
                               gamma=5, opts=None)

    def forward(self, x, labels=None,num_classes=None,common_feat=None):
        loss_re, reconstruction_errors,score, reconstructed ,loss_class, probs= self.loss_fn(self.denoise_fn_D, x,None,None,common_feat)
        return loss_re.mean(-1).mean(),reconstruction_errors.mean(dim=1), score, reconstructed,loss_class, probs


def classify2(dm_condition, ae, x, edge, labels, num_classes, num_samples=50):
    """
    分类推理函数
    使用扩散模型计算每个类别的损失，并进行预测。
    """
    # 初始化变量
    probs_all = []
    x = ae.encode(x, edge)

    for _ in range(num_samples):
        # 计算扩散损失并得到每个类别的损失
        loss_per_class = []
        for class_id in range(num_classes):
            # 创建类别标签
            class_labels = torch.full((x.shape[0],), class_id).cuda()

            # 使用扩散模型进行前向传播并计算每个类别的损失
            loss = dm_condition.get_one_instance_prediction(x)  # 获取预测损失
            loss_per_class.append(loss)

        # 将损失转化为 logit 形式
        loss_per_class = torch.stack(loss_per_class, dim=1)  # [N, num_classes]
        probs = F.softmax(-loss_per_class, dim=1)  # 计算每个类别的概率
        probs_all.append(probs)

        # 根据最终概率进行分类
        final_probs = torch.mean(torch.stack(probs_all), dim=0)  # 多次采样后求平均
        pred_labels = torch.argmax(final_probs, dim=1)  # 预测类别

        # 计算分类准确率
        correct = (pred_labels == labels).sum().item()
        accuracy = correct / labels.shape[0]
        print("Accuracy: {:.4f}".format(accuracy))

    return final_probs, accuracy


def classify(dm_condition,ae, x,edge,labels, num_classes, num_samples=50):
    """
    分类推理函数
    """
    probs_all = []
    x = ae.encode(x, edge)
    for _ in range(num_samples):

        # 蒙特卡洛采样
        sigma = (torch.randn(x.shape[0]) * dm_condition.P_std + dm_condition.P_mean).exp().to(x.device)
        noise = torch.randn_like(x) * sigma.unsqueeze(1)

        # 计算每个类别的误差
        class_errors = []

        for y in range(num_classes):
            y = torch.full((x.shape[0],), y).cuda()
            labels_tensor = torch.zeros((y.shape[0], num_classes), device=x.device)
            labels_tensor[torch.arange(y.shape[0]), y.to(torch.int)] = 1
            class_labels = labels_tensor
            pred = dm_condition.denoise_fn_D(x + noise, sigma, None,class_labels)
            error = torch.norm(pred - x, dim=1) ** 2
            # loss,reconstruction_errors, score_train, reconstructed, loss_c, probs = dm_condition(x, y, num_classes,
            #                                                                     common_feat=None)
            class_errors.append(error)

        # 计算概率
        class_errors = torch.stack(class_errors, dim=1)  # [B, C]
        probs = F.softmax(-class_errors, dim=1)
        probs_all.append(probs)
        final_probs = torch.mean(torch.stack(probs_all), dim=0)
        pred_labels = torch.argmax(final_probs, dim=1)  # [N]
        correct = (pred_labels == labels).sum().item()
        accuracy = correct / labels.shape[0]
        print(accuracy)

    # # 平均多次采样结果
    # final_probs = torch.mean(torch.stack(probs_all), dim=0)
    # pred_labels = torch.argmax(final_probs, dim=1)  # [N]
    # correct = (pred_labels == labels).sum().item()
    # accuracy = correct / labels.shape[0]
    # print(accuracy)
    return final_probs,accuracy

def sample_step(net, num_steps, i, t_cur, t_next, x_next, common_feat=None):
    x_cur = x_next
    # Increase noise temporarily.    
    gamma = min(S_churn / num_steps, math.sqrt(
        2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() \
            * S_noise * torch.randn_like(x_cur)
    # Euler step.                  

    denoised = net(x_hat, t_hat, common_feat).to(torch.float32)
    ##################################
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur
    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised = net(x_next, t_next, common_feat).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (
                0.5 * d_cur + 0.5 * d_prime)

    return x_next


def sample_dm(net, noise, num_steps, common_feat = None):
    step_indices = torch.arange(num_steps, dtype=torch.float32,
                                device=noise.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    z = noise.to(torch.float32) * t_steps[0]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    with (torch.no_grad()):
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            z = sample_step(net, num_steps, i, t_cur, t_next, z, common_feat)
    return z

# 该函数负责在每一个时间步上执行一次采样步骤。它的目标是通过反向扩散过程从噪声逐步恢复数据。
def sample_step_free(condition_net,uncondition_net,y, num_steps, i, t_cur, t_next, x_next, common_feat = None, lamda=None):
    x_cur = x_next # 设置当前样本为下一步样本，作为采样的起点
    # 计算gamma，用于增加噪声的扰动（S_churn 是扰动系数）
    gamma = min(S_churn / num_steps, math.sqrt(
        2) - 1) if S_min <= t_cur <= S_max else 0
    # 当前时间步的噪声强度，包含扰动部分/
    t_hat = condition_net.round_sigma(t_cur + gamma * t_cur)
    # 使用噪声模型生成扰动的样本
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() \
            * S_noise * torch.randn_like(x_cur)

    # 通过条件网络和无条件网络分别去噪
    denoised_condition = condition_net(x_hat, t_hat, common_feat=common_feat,y=y).to(torch.float32)
    denoised_uncondition = uncondition_net(x_hat, t_hat).to(torch.float32)
    # 计算去噪后的梯度（通过当前样本与去噪后的样本计算得到）
    d_cur_condition = (x_hat - denoised_condition) / t_hat
    d_cur_uncondition = (x_hat - denoised_uncondition) / t_hat

    ###################################
    ## Guidance Process
    ###################################
    # 在引导过程中，结合条件网络和无条件网络的梯度，调节当前梯度的影响
    d_cur = (1 + lamda) * d_cur_uncondition - (lamda) * d_cur_condition
    # d_cur = d_cur_condition
    # 计算下一个时间步的样本
    x_next = x_hat + (t_next - t_hat) * d_cur
    # Apply 2nd order correction.
    # 如果当前时间步不是最后一个步骤，应用二阶修正
    if i < num_steps - 1:
        # denoised_condition = d_cur_condition(x_next, t_next, common_feat=common_feat).to(torch.float32)
        # denoised_uncondition = d_cur_uncondition(x_next, t_next).to(torch.float32)

        denoised_condition = condition_net(x_next, t_next, common_feat=common_feat,y=y).to(torch.float32)
        denoised_uncondition = uncondition_net(x_next, t_next).to(torch.float32)

        d_prime_condition = (x_next - denoised_condition) / t_next
        d_prime_uncondition = (x_next - denoised_uncondition) / t_next
        # 再次进行引导，修正梯度
        d_prime = (1. + lamda) * d_prime_uncondition - lamda * d_prime_condition
        # d_prime =  d_prime_condition
        x_next = x_hat + (t_next - t_hat) * (
                0.5 * d_cur + 0.5 * d_prime)

    return x_next # 返回更新后的样本


def sample_dm_free(condition_net,uncondition_net, noise,y, num_steps, common_feat = None, lamda=None):
    # 创建一个时间步的索引序列
    step_indices = torch.arange(num_steps, dtype=torch.float32,
                                device=noise.device)

    sigma_min = max(SIGMA_MIN, uncondition_net.sigma_min)
    sigma_max = min(SIGMA_MAX, uncondition_net.sigma_max)
    # sigma_min = max(SIGMA_MIN, condition_net.sigma_min)
    # sigma_max = min(SIGMA_MAX, condition_net.sigma_max)
    # 计算出每个时间步对应的噪声强度
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho #
    t_steps = torch.cat(
        [condition_net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    # 使用输入噪声（noise）初始化采样过程，将其乘以初始噪声强度t_steps[0]。
    z = noise.to(torch.float32) * t_steps[0]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    with (torch.no_grad()):
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            z = sample_step_free(condition_net,uncondition_net,y, num_steps, i, t_cur, t_next, z, common_feat, lamda=lamda)
    return z
