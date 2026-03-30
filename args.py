import argparse


    # 'Caltech101-20', 'Caltech101-all', 'citeseer', 'COIL', 'flower17', 'GRAZ02', 'handwritten', 'Hdigit', 'HW', 'Mfeat', 'MNIST10k', 'MITIndoor',
# 'MSRC-v1', 'NoisyMNIST_15000', 'NUS-WIDE', '3sources', '20newsgroups', '100leaves', 'ALOI', 'animals', 'BBCnews', 'BBCSports', 'Caltech101-7',
    #  'COIL',  'GRAZ02','Notting-Hill', 'ORL2', 'scene15', 'UCI', 'WebKB',
    #  'Wikipedia', 'Youtube','AwA','NUSWIDEOBJ'
    #  'Reuters'报错
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="0", help='gpu')
    parser.add_argument('--lamda', dest='lamda', type=float, default=1)
    parser.add_argument('--dataset', dest='dataset', type=str, default='NUS-WIDE')
    parser.add_argument('--ae_lr', dest='ae_lr', type=float, default=0.001)
    parser.add_argument('--diff_lr', dest='diff_lr', type=float, default=0.1)
    parser.add_argument('--ae_alpha', dest='ae_alpha', type=float, default=0.05)
    parser.add_argument('--ae_dropout', dest='ae_dropout', type=float, default=0.1)


    parser.add_argument("--layer_num", type=int, default=2, help="Number of network layer.")
    parser.add_argument('--rep_num', type=int, default=1, help='Number of rep.')
    parser.add_argument('--epoch', type=int, default=2500, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--k', type=int, default=10, help='k of kneighbors_graph.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nhid', type=int, default=32, help='the dimension of hidden layer')

    # for early stop
    parser.add_argument("--early-stop", type=bool, default=False, help="If early stop")
    parser.add_argument("--patience", type=int, default=20, help="Patence for early stop")

    parser.add_argument('--batch_norm', type=int, default=1, choices=[0, 1], help='whether to use batch norm')
    parser.add_argument('--train_ratio', type=float, default=0.1, help='train_ratio')
    # parser.add_argument('--valid_ratio', type=float, default=0, help='valid_ratio')
    # parser.add_argument('--test_ratio', type=float, default=0.9, help='test_ratio')
    parser.add_argument('--data_split_mode', type=str, default='Ratio', help='data_split_mode, [Ratio, Num]')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--res_path', type=str, default="./results/GCN/", help='Dataset to use.')

    args = parser.parse_args()
    return args


args = get_arguments()