from __future__ import division
from __future__ import print_function


import time
from DataLoader import LoadMatData, generate_permutation
import torch.nn.functional as F
import torch.optim as optim

from MvGCN import MvGCN

from utils import accuracy, f1_test
from tqdm import tqdm

import torch


def Classifier(adj, features, labels, nfeats, num_view, num_class, args, device):

    idx_train, idx_test = generate_permutation(labels, args)
    # model = MvMLP(nfeats, num_class)
    model = MvGCN(nfeats, num_class)
    # model = MvFGCN(nfeats, num_class)
    # model = MvACMGCN(nfeats, num_class)
    total_para = sum(x.numel() for x in model.parameters())
    print("Total number of paramerters in networks is {}  ".format(total_para / 1e6))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for i in range(num_view):
        # exec("features_{}= torch.from_numpy(features[{}]/1.0).float().to(device)".format(i, i))
        exec("features[{}]= torch.Tensor(features[{}] / 1.0).to(device)".format(i, i))
        exec("features[{}] = F.normalize(features[{}])".format(i, i))
        exec("adj[{}]=adj[{}].to_dense().float().to(device)".format(i, i))


    model.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()  # [:10]
    idx_test = idx_test.cuda()
    # f_loss = open('./results/loss_and_acc_curve/loss/' + args.dataset + '.txt', 'w')
    # f_ACC = open('./results/loss_and_acc_curve/ACC/' + args.dataset + '.txt', 'w')
    # f_F1 = open('./results/loss_and_acc_curve/F1/' + args.dataset + '.txt', 'w')
    t1 = time.time()
    out_model=model
    with tqdm(total=args.epoch) as pbar:
        pbar.set_description('Training:')
        temp_acc=0
        temp_f1=0
        for i in range(args.epoch):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            output = F.log_softmax(output, dim=1)
            # temp1 = output[idx_train]
            # temp2 = labels[idx_train]
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            f1_train = f1_test(output[idx_train], labels[idx_train])
            loss_train.backward()
            # clip_gradient(model, clip_norm=0.5)
            optimizer.step()

            if not args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output = model(features, adj)
                output = F.log_softmax(output, dim=1)

            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            f1 = f1_test(output[idx_test], labels[idx_test])
            if acc_test > temp_acc:
                temp_acc = acc_test
                temp_f1 = f1
                out_model=model

            # # loss曲线
            # isExists = os.path.exists("./results_linear/loss/{}".format(args.dataset))
            # if not isExists:
            #     os.mkdir("./results_linear/loss/{}".format(args.dataset))
            # with open("./results_linear/loss/{}".format(args.dataset) + '/loss.txt', 'a', encoding='utf-8') as f:
            #     f.write(str(loss_test.detach().cpu().numpy()) + '\n')
            # with open("./results_linear/loss/{}".format(args.dataset) + '/acc.txt', 'a', encoding='utf-8') as f:
            #     f.write(str(acc_test.detach().cpu().numpy()) + '\n')
            # with open("./results_linear/loss/{}".format(args.dataset) + '/f1.txt', 'a', encoding='utf-8') as f:
            #     f.write(str(f1) + '\n')

            outstr = 'Epoch: {:04d} '.format(i + 1) + \
                     'loss_train: {:.4f} '.format(loss_train.item()) + \
                     'acc_train: {:.4f} '.format(acc_train.item()) + \
                     'loss_test: {:.4f} '.format(loss_test.item()) + \
                     'acc_test: {:.4f} '.format(temp_acc.item()) + \
                     'f1_test: {:.4f} '.format(temp_f1.item()) + \
                     'time: {:.4f}s'.format(time.time() - t)
            pbar.set_postfix_str(outstr)
            # print(outstr)
            # f_loss.write(str(loss_train.item()) + '\n')
            # f_ACC.write(str(acc_test.item()) + '\n')
            # f_F1.write(str(f1.item()) + '\n')
            pbar.update(1)
    # return model, loss_val.item(), acc_val.item(), loss_test.item(), acc_test.item()
    # draw_plt(output[idx_test], labels[idx_test], args.dataset)
    print('total_time:', time.time() - t1)
    return out_model, features, labels, adj, idx_test, output,temp_acc.item(),temp_f1.item()


def test(args, model, features, labels, adj, idx_test):
    model.eval()
    output = model(features, adj)

    # # TSNE
    # prediction_test = F.log_softmax(output[idx_test], dim=1)
    # labels_test = labels[idx_test]
    # prediction_test = prediction_test.detach().cpu().numpy()
    # labels_test = labels_test.detach().cpu().numpy()
    # sio.savemat('{}_MvOGCN_linear_prediction_test.mat'.format(args.dataset), {'prediction_test': prediction_test})
    # sio.savemat('{}_MvOGCN_linear_labels_test.mat'.format(args.dataset), {'labels_test': labels_test})

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    f1 = f1_test(output[idx_test], labels[idx_test])
    # temp_acc = 0
    # temp_f1 = 0
    # if acc_test > temp_acc:
    #     temp_acc = acc_test
    #     temp_f1 = f1
    print("Dataset:", args.dataset)
    print("Test set results:",
          "accuracy= {:.2f}".format(100 * acc_test.item()),
          "f1= {:.2f}".format(100 * f1.item()))
    return 100 * acc_test.item(), 100 * f1.item()
