import math
import numpy as np
import torch
import faiss
from tqdm import tqdm
from torch import nn
from utils import *
from torch.nn import Module, Parameter
import torch.nn.functional as F


class ICLoss(nn.Module):
    def __init__(self, dim, batch_size, penalty, penalty_scale, temperature=0.5):
        super(ICLoss, self).__init__()
        self.batch_size = batch_size
        self.penalty = penalty
        self.scale = penalty_scale
        self.temperature = temperature
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, emb_i, emb_j, target):
        SIZE = emb_i.shape[0]
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.mm(representations, representations.t().contiguous())
        sim_ij = torch.diag(similarity_matrix, SIZE)
        sim_ji = torch.diag(similarity_matrix, -SIZE)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        mask = self.sample_mask(target)
        penalty_var = self.calculate_penalty(similarity_matrix, self.penalty)
        denominator = torch.exp(similarity_matrix / self.temperature) * mask
        loss_partial = -torch.log(nominator / (torch.sum(denominator, dim=1) * penalty_var + 1e-7))
        loss = torch.sum(loss_partial) / (2 * SIZE)
        return loss

    def calculate_penalty(self, similarities, sigma_max):
        # similarities = self.normalize_matrix(similarities)
        variance = similarities.var(dim=1, unbiased=False)
        variance = self.normalize_matrix(variance)  #
        penalty_var = torch.clamp(variance - sigma_max, min=0)
        penalty_var = (1 + self.scale * penalty_var)
        return penalty_var

    def normalize_matrix(self, matrix):
        min_vals, _ = torch.min(matrix, dim=0, keepdim=True)
        max_vals, _ = torch.max(matrix, dim=0, keepdim=True)
        normalized_matrix = (matrix - min_vals) / (max_vals - min_vals + 1e-7)  #
        return normalized_matrix

    def sample_mask(self, targets):
        targets = targets.cpu().numpy()
        targets = np.concatenate([targets, targets])

        cl_dict = {}
        for i, target in enumerate(targets):
            cl_dict.setdefault(target, []).append(i)
        mask = np.ones((len(targets), len(targets)))
        for i, target in enumerate(targets):
            for j in cl_dict[target]:
                if abs(j - i) != len(targets) / 2:  #
                    mask[i][j] = 0
        return torch.Tensor(mask).cuda().float()


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node, item_2hint, hint_2item, count):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node

        self.scale = opt.scale
        self.temperature = opt.temperature
        self.penalty = opt.penalty
        self.penalty_scale = opt.penalty_scale

        self.hint_2item = hint_2item.cuda()
        self.item_2hint = item_2hint.cuda()
        self.count = count

        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.ICLoss = ICLoss(self.hidden_size, self.batch_size, self.penalty, self.penalty_scale, self.temperature)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def e_step(self):
        items_embedding = self.embedding.weight.detach().cpu().numpy()
        self.item_centroids, self.item_2cluster = self.run_kmeans(items_embedding[:])

    def run_kmeans(self, x):
        kmeans = faiss.Kmeans(d=x.shape[-1], k=500, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        self.cluster_cents = cluster_cents
        _, I = kmeans.index.search(x, 1)
        self.items_cents = I

        cluster_dict = {}
        for i, cluster_id in enumerate(I.flatten()):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(i)
        self.cat_to_item = {k: sorted(v) for k, v in sorted(cluster_dict.items())}

        centroids = torch.Tensor(cluster_cents).cuda()
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().cuda()
        return centroids, node2cluster


    def compute_scores(self, hidden, mask, target):
        hint_2item = self.hint_2item
        item_2hint = self.item_2hint
        count = self.count
        # ----------- intent learning ------- #
        new_embeds = self.embedding(hint_2item) * count.unsqueeze(-1).repeat(1, 1, self.embedding.weight.shape[-1])
        item_centroids = torch.sum(new_embeds, dim=1) / torch.sum(count, dim=-1).unsqueeze(-1).repeat(1,new_embeds.shape[-1])
        self.hint_2embd = item_centroids
        self.item_2hint = item_2hint

        # ---------------- SBR model ----------------- #
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))

        # ---------------- loss computation ----------------- #
        CELoss, ICLoss, scores = self.loss_computation(a, target)

        return scores, CELoss, ICLoss

    def loss_computation(self, seq, targets):
        # -------- ICLoss --------- #
        target_intent = self.item_2hint[targets]
        target_intent_embed = self.hint_2embd[target_intent]
        ICLoss = self.ICLoss(seq, target_intent_embed, target_intent)
        # -------- CELoss --------- #
        l_c = seq
        l_emb = self.embedding.weight[1:]
        scores = torch.matmul(l_c, l_emb.t())
        CELoss = self.loss_function(scores, targets.cuda() - 1)

        return CELoss, ICLoss, scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    targets = trans_to_cuda(torch.Tensor(targets).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    scores, CELoss, ICLoss = model.compute_scores(seq_hidden, mask, targets)
    return scores, CELoss, ICLoss, targets

def train_test(model, opt, train_data, test_data, n_node, Tail):
    model.scheduler.step()
    print('-------------------start training ------------------')
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in tqdm(zip(slices, np.arange(len(slices))), total=len(slices)):
        model.optimizer.zero_grad()
        scores, CELoss, ICLoss, targets = forward(model, i, train_data)
        loss = CELoss + model.scale * ICLoss
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
            print('[%d/%d] LatEx_loss: %.4f' % (j, len(slices), model.scale * ICLoss))
    print('\tLoss:\t%.3f' % total_loss)

    print('------------------- start predicting -------------------')
    model.eval()
    y_pre_all = torch.LongTensor().cuda()
    y_pre_all_10 = torch.LongTensor().cuda()
    test_y = torch.LongTensor().cuda()
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        scores, CELoss, ICLoss, targets = forward(model, i, test_data)
        y_pre = scores.topk(20)[1]
        targets = torch.Tensor(targets).long().cuda()

        test_y = torch.cat((test_y, targets), 0)
        y_pre_all = torch.cat((y_pre_all, y_pre), 0)
        y_pre_all_10 = torch.cat((y_pre_all_10, y_pre[:, :10]), 0)

    tail_items = torch.tensor(Tail).cuda()

    ACLT_20 = compute_ACLT(y_pre_all, tail_items)
    Tail_20 = TailCount(y_pre_all, tail_items)
    Covt_20 = Coverage(y_pre_all, n_node)

    trecall = get_trecall(y_pre_all, test_y.long().unsqueeze(1) - 1, tail_items)
    tmrr = get_tmrr(y_pre_all, test_y.long().unsqueeze(1) - 1, tail_items)
    tndcg = get_tndcg(y_pre_all, test_y.long().unsqueeze(1) - 1, tail_items)

    recall = get_recall(y_pre_all, test_y.long().unsqueeze(1) - 1)
    ndcg = get_ndcg(y_pre_all, test_y.long().unsqueeze(1) - 1)
    mrr = get_mrr(y_pre_all, test_y.long().unsqueeze(1) - 1)

    print("Recall@20: " + "%.4f" % recall + "  MRR@20: " + "%.4f" % mrr.tolist() + "  NDCG@20: " + "%.4f" % ndcg)
    print("tRecall@20: " + "%.4f" % trecall + "  tMRR@20: " + "%.4f" % tmrr.tolist() + "  tNDCG@20: " + "%.4f" % tndcg)
    print("tCov@20: " + "%.4f" % Covt_20 + " ACLT@20: " + "%.4f" % ACLT_20 + "  Tail@20: " + "%.4f" % Tail_20)

    return recall
