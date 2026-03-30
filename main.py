import argparse
import os

import numpy as np
import torch

import datetime
from GuiMvdiff import GuiMvdiff

from DataLoader import LoadMatData

from train import Classifier, test
from args import get_arguments

args = get_arguments()

adj, features, labels, nfeats, num_view, num_class, adj_noself = LoadMatData(
    args.dataset, args.k, 'D:\project\dataset//')
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)     # set GPU
device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
dset = args.dataset
models = []

model = GuiMvdiff(lr=args.diff_lr, ae_alpha=args.ae_alpha, ae_lr=args.ae_lr, ae_dropout=args.ae_dropout, lamda=args.lamda)
re_As, re_Xs,latent_re_As,latent_re_Xs=model(adj, features, labels, nfeats, num_view, num_class, adj_noself,args)

acc = []
f1 = []

nfeats=[]
for i in range(len(features)):
    nfeats.append(len(latent_re_Xs[i][1]))

for i in range(args.rep_num):
    print('rep_num:', i + 1)
    model, features, labels, adj, idx_test, output ,temp_acc,temp_f1= Classifier(latent_re_As, latent_re_Xs, labels, nfeats, num_view,
                                                           num_class, args, device)

    # Testing
    # acc_test, f1_score = test(args, model, features, labels, adj, idx_test)
    acc.append(temp_acc)
    f1.append(temp_f1)
    del model

# nfeats=[]
# for i in range(len(features)):
#     nfeats.append(len(re_Xs[i][1]))
#     re_Xs[i]=re_Xs[i].detach()
#
# for i in range(args.rep_num):
#     print('rep_num:', i + 1)
#     model, features, labels, adj, idx_test, output ,temp_acc,temp_f1= Classifier(re_As, re_Xs, labels, nfeats, num_view,
#                                                            num_class, args, device)
#
#     # Testing
#     acc_test, f1_score = Eva(args, model, features, labels, adj, idx_test)
#     acc.append(acc_test)
#     f1.append(f1_score)
#     del model

print("Optimization Finished!")

print("accuracy_mean= {:.4f}".format(np.array(acc).mean()),
      "accuracy_std= {:.4f}".format(np.array(acc).std()))
print("f1_mean= {:.4f}".format(np.array(f1).mean()), "f1_std= {:.4f}".format(np.array(f1).std()))

isExists = os.path.exists(args.res_path)
if not isExists:
    os.mkdir(args.res_path)
with open(args.res_path + '/{}.txt'.format(args.dataset), 'a', encoding='utf-8') as f:
    f.write(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '\n'
                                                                     'dataset:{} | layer_num:{} | rep_num：{} | Ratio：{}'.format(
        args.dataset, args.layer_num, args.rep_num, args.train_ratio) + '\n'
                                                                        'dropout:{} | epochs:{} | lr:{} | wd:{} | hidden:{}'.format(
        args.dropout, args.epoch, args.lr, args.weight_decay, args.nhid) + '\n'
                                                                           'temp_acc:{:.4f} | ACC_std: {:.4f} | ACC_max:{:.4f}'.format(
        np.array(acc).mean(), np.array(acc).std(), np.array(acc).max()) + '\n'
                                                                          'temp_f1:{:.4f} | F1_std: {:.4f} | F1_max:{:.4f}'.format(
        np.array(f1).mean(), np.array(f1).std(), np.array(f1).max()) + '\n'
                                                                       '----------------------------------------------------------------------------' + '\n')

# models.append(model)

# model = DiffGAD(lr=0.004, ae_alpha=args.ae_alpha, ae_lr=args.ae_lr, ae_dropout=args.ae_dropout, lamda=args.lamda)



