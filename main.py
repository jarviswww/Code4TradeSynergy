import torch
import argparse
import pickle
import time
import os
from utils import *
from model import *
from spectral_clustering import *
from attribute_graph import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica-2', help='dataset name: diginetica-2/Tmall-2/30music-2')
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.6, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=1, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1,help='split the portion of training set as validation set')
# -------- HID --------- #
parser.add_argument('--temperature', type=float, default=0.14, help='the temperature coefficient of ICLoss')
parser.add_argument('--scale', type=float, default=0.2, help='the sacle of LatEx loss')
parser.add_argument('--cluster', type=int, default=200, help='number of spectral clusters')
parser.add_argument('--penalty', type=float, default=0.2, help='threshold of  penalty')
parser.add_argument('--penalty_scale', type=float, default=0.3, help='standard of  penalty')

opt = parser.parse_args(args=[])
print(opt)


def main():
    # ----------------- Data Load -------------- #
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    TailHead = pickle.load(open('datasets/' + opt.dataset + '/TailHead' + '.pkl', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    Tail = TailHead[0]

    # ----------------- Hybrid Intent Learning -------------- #
    try:
        item_2hint = pickle.load(open('datasets/' + opt.dataset + '/item_2att.pkl', 'rb'))
    except:
        attribute_graph_generator(opt.dataset)
        spectral_clustering(opt.dataset, opt.cluster)
        item_2hint = pickle.load(open('datasets/' + opt.dataset + '/item_att.txt', 'rb'))
    item_2hint, hint_2item, mask = h_intent_process(item_2hint)

    # ----------------- Data Prepare -------------- #
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)

    if opt.dataset == 'diginetica-2':
        n_node = 43098
    elif opt.dataset == 'Tmall-2':
        n_node = 40728
    else:
        n_node = 0

    model = trans_to_cuda(SessionGraph(opt, n_node, item_2hint, hint_2item, mask))

    start = time.time()
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        train_test(model, opt, train_data, test_data, n_node, Tail)
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()