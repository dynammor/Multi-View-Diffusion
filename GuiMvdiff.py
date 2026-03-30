import tqdm
import torch
from pygod.metric import *
from pygod.utils import load_data
from torch_geometric.datasets import DGraphFin

import os
import tqdm
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform


# from torch.optim.lr_scheduler import ReduceLROnPlateau
from diffusion_models import MLPDiffusion, Model, sample_dm, sample_dm_free, classify
from datetime import datetime

from pygod.metric.metric import *

from torch_geometric.utils import to_dense_adj
from pygod.nn.decoder import DotProductDecoder
from pygod.nn.functional import double_recon_loss
from pygod.utils import load_data

# from mo import GCN
from torch_geometric.nn import GCN


# 使用 t 作为索引，从 a 中提取对应的元素
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())  # 
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# define beta schedule
betas = linear_beta_schedule(timesteps=500)

# define alphas
alphas = 1. - betas
# 对 alphas 的累积乘积，表示数据保持的累积程度
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# 图自编码器
class Graph_AE(nn.Module):
    # def __init__(self,
    #              in_dim,
    #              hid_dim=64,
    #              num_layers=4,
    #              dropout=0.,
    #              act=torch.nn.functional.relu,
    #              sigmoid_s=False,
    #              backbone=GCN,
    #              **kwargs):
    #     super(Graph_AE, self).__init__()
    #
    #     # split the number of layers for the encoder and decoders
    #     assert num_layers >= 2, \
    #         "Number of layers must be greater than or equal to 2."
    #     encoder_layers = math.floor(num_layers / 2)
    #     decoder_layers = math.ceil(num_layers / 2)
    #
    #     self.shared_encoder = backbone(nfeat=in_dim,
    #                                    nhid=hid_dim,
    #                                    nclass=hid_dim,
    #                                    dropout=dropout,
    #                                    layers=encoder_layers,
    #                                    **kwargs)
    #
    #     self.attr_decoder = backbone(nfeat=hid_dim,
    #                                  nhid=hid_dim,
    #                                   nclass=in_dim,
    #                                  dropout=dropout,
    #                                  layers=decoder_layers,
    #                                  **kwargs)
    #
    #     self.struct_decoder = DotProductDecoder(in_dim=hid_dim,
    #                                             hid_dim=hid_dim,
    #                                             num_layers=decoder_layers - 1,
    #                                             dropout=dropout,
    #                                             act=act,
    #                                             sigmoid_s=sigmoid_s,
    #                                             backbone=GCN,
    #                                             **kwargs)
    #
    #     self.loss_func = double_recon_loss
    #     self.emb = None

    def __init__(self,
                 in_dim,
                 hid_dim=128,
                 num_layers=2,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GCN,
                 num_classes=10,
                 **kwargs):
        super(Graph_AE, self).__init__()

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        self.fc1 = nn.Linear(in_dim, hid_dim)  # 输入层到隐藏层
        self.tanh = nn.Tanh()  # tanh激活函数
        self.fc2 = nn.Linear(hid_dim, hid_dim)  # 隐藏层到输出层
        self.dropout = dropout

        self.bn1 = nn.BatchNorm1d(hid_dim)


        # self.fc1 = nn.Linear(in_dim, hid_dim)
        # self.fc2 = nn.Linear(hid_dim, hid_dim)
        #
        #
        self.fc3 = nn.Linear(hid_dim, hid_dim)
        self.fc4 = nn.Linear(hid_dim, in_dim)

        self.fc_class = nn.Linear(hid_dim, num_classes)

        self.shared_encoder = backbone(in_channels=in_dim,
                                       hidden_channels=hid_dim,
                                       num_layers=encoder_layers,
                                       out_channels=hid_dim,
                                       dropout=dropout,
                                       act=act,
                                       **kwargs)

        self.attr_decoder = backbone(in_channels=hid_dim,
                                     hidden_channels=hid_dim,
                                     num_layers=decoder_layers,
                                      out_channels=in_dim,
                                     dropout=dropout,
                                     act=act,
                                     **kwargs)

        self.struct_decoder = DotProductDecoder(in_dim=hid_dim,
                                                hid_dim=hid_dim,
                                                num_layers=decoder_layers - 1,
                                                dropout=dropout,
                                                act=act,
                                                sigmoid_s=sigmoid_s,
                                                backbone=backbone,
                                                **kwargs)

        self.loss_func = double_recon_loss
        self.emb = None

        # self.fc = nn.Linear(hid_dim, num_classes)
        # self.fc2 = nn.Linear(num_classes, hid_dim)
    # x是属性，s是结构，属性编码器用的GCN
    def forward(self, x, edge_index):
        self.emb = self.encode(x, edge_index)
        # self.emb2 = self.fc(self.emb)
        # self.emb3 = self.fc2(self.emb2)
        x_, s_ = self.decode(self.emb, edge_index)
        self.emb = F.dropout(self.emb, self.dropout, training=self.training)
        # self.emb=self.fc_class(self.emb)
        return x_, s_, self.emb
    
    # GCN
    # def encode(self, x, edge_index):
    #     self.emb = self.shared_encoder(x, edge_index)
    #     return self.emb
    #
    # def decode(self, emb, edge_index):
    #     x_ = self.attr_decoder(emb, edge_index)
    #     s_ = self.struct_decoder(emb, edge_index)
    #     return x_, s_

    # MLP
    def encode(self, x, edge_index):
        self.emb = self.tanh(self.fc1(x))
        self.emb = self.fc2(self.emb)
        return self.emb

    def decode(self, emb, edge_index):
        x_ = self.tanh(self.fc3(emb))
        x_ = self.fc4(x_)
        return x_,edge_index

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


from utils import load_data, get_node_pairs_from_adj, load_graph_dataset
from DataLoader import LoadMatData, generate_permutation
import random

import os


class GuiMvdiff(BaseTransform):
    def __init__(self,
                 name="",
                 hid_dim=128,
                 diff_dim=None,
                 ae_epochs=800,
                 diff_epochs=800,
                 patience=100,
                 view_index=3,
                 lr=0.005,
                 wd=0.,
                 lamda=0.0,
                 sample_steps=50,
                 radius = 1,
                 ae_dropout=0.3,
                 ae_lr=0.05,
                 ae_alpha=0.8,
                 verbose=True):

        self.view_index = view_index
        self.name = name
        self.hid_dim = hid_dim
        self.diff_dim = diff_dim
        self.ae_epochs = ae_epochs
        self.diff_epochs = diff_epochs
        self.patience = patience
        self.lr = lr
        self.wd = wd
        self.sample_steps = sample_steps
        # 是否输出详细的训练日志。
        self.verbose = verbose
        self.lamda = lamda
        
        self.common_feat = None
        self.common_feats = None
        self.dm = None
        self.dms = []

        self.ae = None
        self.aes = []
        self.ae_dropout = ae_dropout
        self.ae_lr = ae_lr
        self.ae_alpha = ae_alpha
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.timesteps = 50
        # 图数据的半径参数，可能用于图的处理。
        self.radius = radius

        self.re_A=[]
        self.re_X=[]

        self.latent_re_A=[]
        self.latent_re_X=[]

        self.args=None


    def __call__(self, adj, features, labels, nfeats, num_view, num_class, adj_noself, args):
        return self.forward(adj, features, labels, nfeats, num_view, num_class, adj_noself, args)

    def forward(self, adjs, features, labels, nfeats, num_view, num_class, adj_noself,args):

        # dataset=load_graph_dataset('cora')

        # features = features.cuda()
        # adj = adj.cuda()
        # num_classes = len(set(np.array(dataset.y)))
        # num_classes=8
        self.hid_dim =num_class
        self.args = args
        # 自动计算隐藏层维度，如果未指定的话
        common_feats=[]
        for view_index in range(num_view):
            self.view_index=view_index
            feature=features[self.view_index]
            feature = torch.from_numpy(feature).cuda()
            adj=adjs[self.view_index].cuda()
            if self.hid_dim is None:
                self.hid_dim = 2 ** int(math.log2(feature.size(1)) - 1)

            # self.diff_dim=num_classes
            if self.diff_dim is None:
                self.diff_dim = 2 * self.hid_dim
            num_node_features=feature.size(1)
            # 初始化自编码器模型
            self.ae = Graph_AE(in_dim = num_node_features,
                               hid_dim=self.hid_dim,
                               dropout=self.ae_dropout,
                               num_classes=num_class).cuda()

            # 设置模型保存路径
            self.save_dir = os.getcwd() + '/models/'+ args.dataset + '/full_batch/'
            self.ae_path = self.save_dir + "/" + str(self.ae_dropout) + "_" + str(self.ae_lr) + "_" + str(self.ae_alpha) + "_" + str(self.hid_dim)
            if not os.path.exists(self.ae_path):
                os.makedirs(self.ae_path)

            ######################## train autoencoder #######################
            self.train_ae(adj, feature,labels)# 训练自编码器
            ae_dict = torch.load(self.ae_path + '/Graph_AE.pt') # 加载自编码器的训练权重
            self.ae.load_state_dict(ae_dict['state_dict']) # 将权重加载到自编码器模型中


            # 进行多次实验
            num_trial = 1
            for _ in tqdm.tqdm(range(num_trial)):
                ##################################
                # 无条件扩散模型（无标签）
                denoise_fn = MLPDiffusion(self.hid_dim, self.diff_dim,num_class).cuda()
                self.dm = Model(denoise_fn=denoise_fn,
                                hid_dim=self.hid_dim).cuda() # 初始化无条件扩散模型
                self.common_feat = self.train_dm(adj, feature) # 训练无条件扩散模型

                # 加载扩散模型
                dm_dict = torch.load(self.ae_path + '/edm.pt')
                self.dm.load_state_dict(dm_dict['state_dict'])
                # 获取共享特征
                self.common_feat = dm_dict['common_feat']
                common_feats.append(self.common_feat)

            ae=self.ae
            dm = self.dm
            self.aes.append(ae)
            self.dms.append(dm)
        self.common_feats = torch.stack(common_feats)
        self.common_feats = torch.mean(self.common_feats, dim=0)

        #################################
        for view_index in range(num_view):
            self.view_index = view_index
            feature = features[self.view_index]
            feature = torch.from_numpy(feature).cuda()
            adj = adjs[self.view_index].cuda()
            for _ in tqdm.tqdm(range(num_trial)):
                # 有条件扩散模型
                print(self.common_feats)

                # 初始化有条件扩散模型
                denoise_condition = MLPDiffusion(self.hid_dim, self.diff_dim,None).cuda()
                self.dm_condition = Model(denoise_fn=denoise_condition,
                                hid_dim=self.hid_dim).cuda()
                # 训练
                self.train_dm_condition(adj, feature,self.aes[self.view_index],num_class)
                # 加载
                dm_free_dict = torch.load(self.ae_path + '/conditional_edm.pt')
                self.dm_condition.load_state_dict(dm_free_dict['state_dict'])

            #################################
            # evaluation
            # 模型评估
            # np.array(x.detach().cpu())

            # classify(self.dm_condition,self.ae,x,edge,y,num_classes)
            self.sample_free(self.dm_condition, self.dms[self.view_index],self.aes[self.view_index],adj, feature)

        return self.re_A, self.re_X, self.latent_re_A, self.latent_re_X

    def train_ae(self, adj, feature,labels):
        if self.verbose:
            print('Training autoencoder ...')
        lr = self.ae_lr
        optimizer = torch.optim.Adam(self.ae.parameters(), lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        adj_dense = adj.to_dense()
        idx_train, idx_test = generate_permutation(labels, self.args)
        for epoch in range(1, self.ae_epochs+1):
            self.ae.train()
            optimizer.zero_grad()

            feature = feature.cuda()
            adj = adj.cuda()
            labels=labels.cuda()
            # s = to_dense_adj(adj)[0].cuda()
            # 前向传播
            x_, s_, embedding = self.ae(feature, adj)
            embedding = F.log_softmax(embedding, dim=1)
            losses_revised = F.nll_loss(embedding[idx_train], labels[idx_train])
            # 计算损失，重建损失

            score = self.ae.loss_func(feature, x_, adj_dense, s_, self.ae_alpha)
            loss = torch.mean(score)
            loss=loss+losses_revised

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if epoch%50 == 0:
                print("Epoch:", '%04d' % (epoch), "loss=", "{:.5f}".format(loss))
            save_path = self.ae_path
            torch.save({
                'state_dict': self.ae.state_dict(),
            }, save_path + '/Graph_AE.pt')


    #  训练无条件扩散模型
    def train_dm(self, adj, feature):
        if self.verbose:
            print('Training diffusion model ...')      
        optimizer = torch.optim.Adam(self.dm.parameters(), lr=self.lr,
                                     weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) 
        self.dm.train()
        # 初始化最优损失
        best_loss = float('inf') 

        best_auc = 0
        patience = 0
        # 初始化共享特征
        common_feat = None

        for epoch in range(self.diff_epochs):
            feature = feature.cuda()
            adj = adj.cuda()
            # 使用自编码器编码输入数据
            inputs = self.ae.encode(feature, adj)

            # 初始化或更新共享特征
            if epoch == 0:
                common_feat = torch.mean(inputs, dim=0)
            else:
                s_v = self.cos(common_feat.unsqueeze(0), reconstructed) # 计算共享特征和重建特征的余弦相似度
                omega = softmax_with_temperature(s_v,t=5).reshape(1,-1) # 使用softmax调整共享特征
                common_feat = torch.mm(omega, reconstructed).detach()
                common_feat  = common_feat.squeeze(0)# 更新共享特征

            # 前向传播，计算损失
            loss, reconstruction_errors,score_train, reconstructed ,loss_class,probs = self.dm(inputs)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dm.parameters(), 1.0)
            optimizer.step()
            if epoch%50 == 0:
                print("Epoch:", '%04d' % (epoch), "loss=", "{:.5f}".format(loss))
            scheduler.step()

            if loss < best_loss:
                best_loss = loss
                patience = 0
                save_dir = self.ae_path
                torch.save({
                'state_dict': self.dm.state_dict(),
                'common_feat': common_feat
                }, save_dir + '/edm.pt')
            else:
                patience += 1
                if patience == self.patience:
                    if self.verbose:
                        print('Early stopping')
                    break

        return common_feat
    
    def train_dm_condition(self, adj, feature,ae,num_class):
        if self.verbose:
            print('Training diffusion model ...')      
        optimizer = torch.optim.Adam(self.dm_condition.parameters(), lr=self.lr,
                                     weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        self.dm_condition.train()
        best_loss = float('inf') 

        patience = 0
        feature = feature.cuda()
        adj = adj.cuda()
        # labels = data.y.cuda()
        for epoch in range(self.diff_epochs):


            inputs =ae.encode(feature, adj)

            sigma = (torch.randn(inputs.shape[0]) * self.dm_condition.P_std + self.dm_condition.P_mean).exp().to(inputs.device)
            noise = torch.randn_like(inputs) * sigma.unsqueeze(1)

            # 计算每个类别的误差
            # class_errors = []
            # probs_all = []
            # for y in range(num_classes):
            #     y = torch.full((inputs.shape[0],), y).cuda()
            #     labels_tensor = torch.zeros((y.shape[0], num_classes), device=inputs.device)
            #     labels_tensor[torch.arange(y.shape[0]), y.to(torch.int)] = 1
            #     class_labels = labels_tensor
            #     pred = self.dm_condition.denoise_fn_D(inputs + noise, sigma, None, class_labels)
            #     error = torch.norm(pred - inputs, dim=1) ** 2
            #     class_errors.append(error)
            #
            # class_errors = torch.stack(class_errors, dim=1)  # [B, C]
            # probs = F.softmax(-class_errors, dim=1)
            # # final_probs = torch.mean(torch.stack(probs), dim=0)
            # # pred_labels = torch.argmax(final_probs, dim=1)  # [N]
            # loss_class = F.cross_entropy(probs, labels)

            # correct = (pred_labels == labels).sum().item()
            # accuracy = correct / labels.shape[0]

            # 前向传播
            loss, reconstruction_errors,score_train, reconstructed,loss_c,probs = self.dm_condition(inputs,None,None, common_feat = self.common_feats)

            loss=loss
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dm_condition.parameters(), 1.0)
            optimizer.step()
            if epoch%50 == 0:
                print("Epoch:", '%04d' % (epoch), "loss=", "{:.5f}".format(loss))
            scheduler.step()

            if loss < best_loss:
                best_loss = loss
                patience = 0
                save_dir = self.ae_path
                torch.save({
                'state_dict': self.dm_condition.state_dict()
                }, save_dir + '/conditional_edm.pt')
            else:
                patience += 1
                if patience == self.patience:
                    if self.verbose:
                        print('Early stopping')
                    break

    #  通过训练好的扩散模型生成样本并计算评估指标
    def sample_free(self, condition_model,uncondition_model,ae , adj, feature):
        ae.eval()
        # 评估模式
        condition_model.eval()
        uncondition_model.eval()
        condition_net = condition_model.denoise_fn_D
        uncondition_net = uncondition_model.denoise_fn_D

        # auc = []
        x = feature
        edge_index = adj
        # labels = data.y.cuda()

        # 自编码器编码
        Z_0 = ae.encode(x, edge_index)

        s = edge_index.to_dense()
        ###############  forward process  ####################
        # 生成随机噪声
        noise = torch.randn_like(Z_0)
        # for i in range(0, self.timesteps):
        for i in [5]:
            print('timesteps',i)
            t = torch.tensor([i] * Z_0.size(0)).long().cuda()
            sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, Z_0.shape)# 提取累积平方根α
            sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, Z_0.shape) # 提取1-累积平方根α
            Z_t = sqrt_alphas_cumprod_t * Z_0 + sqrt_one_minus_alphas_cumprod_t * noise # 计算扩散后的噪声

            # 进行扩散模型采样
            if self.sample_steps > 0:
                reconstructed = sample_dm_free(condition_net, uncondition_net, Z_t,None, self.sample_steps, common_feat=self.common_feat, lamda=self.lamda)

            # 解码器
            # s = to_dense_adj(edge_index).cuda()
            x_, s_ = ae.decode(reconstructed, edge_index)

            # 计算损失
            score = ae.loss_func(x, x_, s, s_, self.ae_alpha)

            self.re_A.append(s_)
            self.re_X.append(x_)

            self.latent_re_A.append(edge_index)
            self.latent_re_X.append(reconstructed)
            # pyg_auc = eval_roc_auc(y, score.cpu().detach())

            # auc.append(pyg_auc)
            # print("timestep:{},pyg_AUC: {:.4f}".format(i, pyg_auc))

def compute_condition_prob(self, condition_net, Z_t, label_one_hot,num_classes):
    """
    计算条件概率 p(y|x)，即给定类别标签y和输入x的条件下的概率
    """
    # 计算每个类别的对数似然
    likelihoods = []
    for y in range(num_classes):
        label_one_hot[:, y] = 1  # 更新当前类别的标签
        # 使用condition_net和uncondition_net计算条件似然
        condition_likelihood = torch.exp(-compute_diff_loss(condition_net, Z_t, label_one_hot))  # 计算条件似然
        likelihoods.append(condition_likelihood)

    likelihoods = torch.stack(likelihoods, dim=1)  # shape: [batch_size, num_classes]
    # 计算softmax来得到每个类别的条件概率
    return torch.nn.functional.softmax(likelihoods, dim=1)


def compute_diff_loss( net, Z_t, label_one_hot):
    """
    计算扩散模型的diffusion loss，即对数似然和扩散损失之间的差距
    """
    # 在此处实现diffusion loss的计算方式
    # 计算噪声项，利用condition_net计算损失
    noise = torch.randn_like(Z_t).cuda()
    loss = torch.mean((net(Z_t) - noise) ** 2)  # 示例的简单损失计算
    return loss

def softmax_with_temperature(input, t=1, axis=-1):
    ex = torch.exp(input/t)
    sum = torch.sum(ex, axis=axis)
    return ex/sum
