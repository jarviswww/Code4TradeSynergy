import pickle
import torch
import argparse
import numpy as np
from tqdm import tqdm

def renumber(original_dict):

    sorted_values = np.unique(sorted(original_dict.values(), reverse=True))
    value_indices = {value: index for index, value in enumerate(sorted_values)}
    new_dict = {key: value_indices[value] for key, value in original_dict.items()}

    return new_dict

def attribute_graph_generator(dataset):
    dataset = 'Tmall-2'
    train_data = pickle.load(open('datasets/' + dataset + '/train.txt', 'rb'))
    item_to_att = pickle.load(open('datasets/' + dataset + '/item_att.txt', 'rb'))
    item_to_att = renumber(item_to_att)
    pickle.dump(item_to_att, open('datasets/' + dataset + '/item_att' + '.txt', 'wb'))

    att_to_item = {}
    for key, value in item_to_att.items():
        att_to_item.setdefault(value, []).append(key)

    train_x = train_data[0]
    train_y = train_data[1]
    for i in tqdm(range(len(train_x))):
        train_x[i]+=[train_y[i]]
    seq = train_x

    seq_att = [list(map(lambda item_id: item_to_att.get(item_id, "未知类别"), user_list))for user_list in seq]
    all_atts = list(att_to_item.keys())
    num = len(all_atts)
    n_hop = 1
    relation = []
    neighbor = [] * (len(all_atts)+1)

    adj1 = [dict() for _ in range(num)]
    adj = [[] for _ in range(num)]

    for i in tqdm(range(len(seq_att))):
        data = seq_att[i]
        for k in range(1,n_hop+1):
            for j in range(len(data)-k):
                relation.append([data[j], data[j+k]])
                relation.append([data[j+k], data[j]])

    for tup in tqdm(relation):
        if tup[1] in adj1[tup[0]].keys():
            adj1[tup[0]][tup[1]] += 1
        else:
            adj1[tup[0]][tup[1]] = 1

    weight = [[] for _ in range(num)]
    for t in range(num):
        x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
        adj[t] = [v[0] for v in x]
        weight[t] = [v[1] for v in x]

    for i in range(num):  # 筛选出共现次数最多的k个邻居，作为最终构图时该item的邻居节点。
        if len(weight[i]) > 4:
            weight[i] = weight[i][1:4]
            adj[i] = adj[i][1:4]
        else:
            weight[i] = weight[i][1:]
            adj[i] = adj[i][1:]

    adj_matrix = np.zeros((num, num))

    for i in tqdm(range(len(adj))):
        neighbors = adj[i]
        weights = weight[i]
        for j in range(len(neighbors)):
            neighbor_id = neighbors[j]
            weight1 = weights[j]
            adj_matrix[i][neighbor_id] = weight1

    pickle.dump(adj_matrix, open('datasets/' + dataset +  '/att_adj'  + '.pkl', 'wb'))
    print('Attribute Graph has been generated!')
