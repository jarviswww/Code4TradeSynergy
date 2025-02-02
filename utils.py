import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

def h_intent_process(item_2hint):
    item_2hint = renumber(item_2hint)
    hint_2item = {}
    for key, value in item_2hint.items():
        hint_2item.setdefault(value, []).append(key)
    item_2hint = item_to_centroids(item_2hint).cuda()

    hint_2item = {k: sorted(v) for k, v in sorted(hint_2item.items())}
    x = hint_2item.values()
    y = max(map(len, hint_2item.values()))
    new = []
    count = []
    for i in x:
        q1 = list(i) + [0] * (y - len(i))
        q2 = [1] * len(i) + [0] * (y - len(i))
        new.append(q1)
        count.append(q2)
    hint_2item = torch.tensor(new).cuda()
    mask = torch.tensor(count).cuda()

    return item_2hint, hint_2item, mask

def item_to_centroids(item_to_cat):
    sorted_keys = sorted(item_to_cat .keys())
    unique_keys_count = len(sorted_keys)
    matrix = np.zeros((unique_keys_count))
    for i, key in enumerate(sorted_keys):
        matrix[i] = item_to_cat[key]
    return torch.tensor(matrix).long()

def renumber(original_dict):
    sorted_values = np.unique(sorted(original_dict.values(), reverse=True))
    value_indices = {value: index for index, value in enumerate(sorted_values)}
    new_dict = {key: value_indices[value] for key, value in original_dict.items()}

    return new_dict

def item_to_centroids(item_to_cat):
    sorted_keys = sorted(item_to_cat.keys())
    unique_keys_count = len(sorted_keys)
    matrix = np.zeros((unique_keys_count))
    for i, key in enumerate(sorted_keys):
        matrix[i] = item_to_cat[key]
    return torch.tensor(matrix).long()


def renumber(original_dict):
    sorted_values = np.unique(sorted(original_dict.values(), reverse=True))
    value_indices = {value: index for index, value in enumerate(sorted_values)}
    new_dict = {key: value_indices[value] for key, value in original_dict.items()}

    return new_dict


def pad_list(lst, pad_value=0):
    target_len = 100  # max(map(len, lst))
    padded_lst = [
        sublist[:target_len] + [pad_value] * (target_len - len(sublist)) if len(sublist) < target_len
        else sublist[:target_len]  # 截断
        for sublist in lst
    ]
    return padded_lst


def int_to_items(x):
    n1 = x.max().item()
    B1 = [[] for _ in range(n1 + 1)]
    for i in range(len(x)):
        value = x[i].item()
        B1[value].append(i)
    B1 = torch.tensor(pad_list(B1))

    return B1


def get_ndcg(indices, targets):
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero(as_tuple=False)
    if len(hits) == 0:
        return 0
    result = 0
    for single_hit in hits:
        result += np.log2(2) / np.log2(single_hit[1].item() + 2)
    return result / targets.size(0)


def get_tndcg(pre, truth, tail):
    comparison_matrix = truth == tail
    indices = torch.nonzero(comparison_matrix, as_tuple=True)[0]
    t_truth = truth[indices, :]
    t_pre = pre[indices, :]

    targets = t_truth.view(-1, 1).expand_as(t_pre)
    hits = (targets == t_pre).nonzero(as_tuple=False)
    if len(hits) == 0:
        return 0
    result = 0
    for single_hit in hits:
        result += np.log2(2) / np.log2(single_hit[1].item() + 2)
    return result / targets.size(0)


def get_trecall(pre, truth, tail):
    """
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B,1) the truth value of test samples
    :return: recall(Float), the recall score
    """

    comparison_matrix = truth == tail
    indices = torch.nonzero(comparison_matrix, as_tuple=True)[0]
    t_truth = truth[indices, :]
    t_pre = pre[indices, :]

    t_truths = t_truth.expand_as(t_pre)
    hits = (t_pre == t_truths).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (t_pre == t_truths).nonzero().size(0)
    recall = n_hits / t_truth.size(0)
    return recall


def get_tmrr(pre, truth, tail):
    comparison_matrix = truth == tail

    indices = torch.nonzero(comparison_matrix, as_tuple=True)[0]
    t_truth = truth[indices, :]
    t_pre = pre[indices, :]

    targets = t_truth.view(-1, 1).expand_as(t_pre)
    # ranks of the targets, if it appears in your indices
    hits = (targets == t_pre).nonzero()
    if len(hits) == 0:
        return 0
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    r_ranks = torch.reciprocal(ranks)  # reciprocal ranks
    mrr = torch.sum(r_ranks).data / targets.size(0)
    return mrr


def get_recall(pre, truth):
    """
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B,1) the truth value of test samples
    :return: recall(Float), the recall score
    """
    truths = truth.expand_as(pre)
    hits = (pre == truths).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (pre == truths).nonzero().size(0)
    recall = n_hits / truths.size(0)
    return recall


def get_mrr(pre, truth):
    """
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B, 1) real label
    :return: MRR(Float), the mrr score
    """
    targets = truth.view(-1, 1).expand_as(pre)
    # ranks of the targets, if it appears in your indices
    hits = (targets == pre).nonzero()
    if len(hits) == 0:
        return 0
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    r_ranks = torch.reciprocal(ranks)  # reciprocal ranks
    mrr = torch.sum(r_ranks).data / targets.size(0)
    return mrr


def IntMetric(cat_to_item, pre, pre_cate, truth_cate):
    D = torch.zeros_like(pre)
    score = 0
    for i in range(pre.size(0)):
        cat_item_num = len(cat_to_item[truth_cate[i].item()])
        indices = (pre_cate[i] == truth_cate[i].item()).nonzero()
        D[i, indices] = pre[i, indices]

        item_set = torch.unique(D[i], return_counts=True)[0]
        item_occur = torch.unique(D[i], return_counts=True)[1]
        mask = torch.where(item_occur == 1, 1, 0)
        unique_cat_item = item_set * mask
        unique_cat_item = torch.where(unique_cat_item == 0, 0, 1)
        num = torch.sum(unique_cat_item)
        score += num / cat_item_num

    return score / pre.shape[0]


def TailCount(pre, tail):
    '''
    :param pre: dim = (sequence_count, top_k), 推荐的Top-K列表
    :param tail: dim = (number_of_tail_items,), 长尾物品集合
    :param pre_cat: dim = (sequence_count, top_k), 推荐物品的类别
    :param target_cat: dim = (sequence_count,), 每个序列的真实物品类别
    :return: Tail@K 指标值
    '''
    # 将数据移到CPU并转换为张量
    pre = pre.detach().cpu()
    tail_set = tail.detach().cpu()

    sequence_count, top_k = pre.shape  # 获取序列数和 Top-K

    # 创建一个布尔矩阵，表示推荐物品是否属于长尾集合
    tail_mask = torch.isin(pre, tail_set)  # tail_mask: dim=(sequence_count, top_k)

    # 结合两个条件，得到推荐的长尾物品个数
    tail_in_top_k = (tail_mask).sum(dim=1)  # 在每个序列中，计算长尾物品个数

    # 计算 Tail@K
    tail_at_k = (tail_in_top_k.float() / top_k).mean()  # 计算平均值

    return tail_at_k


def compute_ACLT(recommendations, tail_items):
    # Count the number of tail items in each user's recommendations
    recommendations = recommendations.detach().cpu().numpy()
    tail_items = tail_items.detach().cpu().numpy()
    tail_list = []
    for pre in tqdm(recommendations):
        tail = list(np.intersect1d(pre, tail_items))
        tail_list += tail
    count = len(set(tail_list))
    tail_list = 1
    return count / len(tail_items)


def Coverage(pre, n_node):
    score = 0
    item = []
    for i in range(pre.size(0)):
        item.append(pre[i].detach().cpu().numpy())
    item_num = len(np.unique(item))
    return item_num / n_node


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets
