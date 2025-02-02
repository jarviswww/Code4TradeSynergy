import torch
import torch.sparse
import pickle
import numpy as np
import faiss
import os
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import warnings
warnings.filterwarnings("ignore")

def spectral_clustering_faiss(adj_matrix, num_clusters):

    A = adj_matrix.float()

    degrees = torch.sum(A, dim=1)
    D = torch.diag(degrees)
    L = D - A
    L_np = csr_matrix(L.numpy())

    k = num_clusters 
    eigenvalues, eigenvectors = eigsh(L_np, k=k, which='SM')

    # 前 k 个最小的非零特征值对应的特征向量
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx[1:num_clusters+1]]
    embedding = eigenvectors.astype(np.float32)
    embedding = np.ascontiguousarray(embedding)

    kmeans = faiss.Kmeans(d=embedding.shape[1], k=num_clusters, niter=30, verbose=False, gpu=False)

    kmeans.train(embedding)
    distances, labels = kmeans.index.search(embedding, 1)

    return labels.flatten()


def spectral_clustering(dataset, num_clusters):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    adj = pickle.load(open('datasets/' + dataset +  '/att_adj'  + '.pkl', 'rb'))
    item_2att = pickle.load(open('datasets/' + dataset + '/item_att.txt', 'rb'))

    adj = torch.tensor(adj)
    att_2hint = spectral_clustering_faiss(adj, num_clusters)
    item_2hint = {}
    for item_id, att_id in item_2att.items():
        intent_id = att_2hint[att_id]
        item_2hint[item_id] = intent_id

    pickle.dump(item_2hint, open('datasets/' + dataset  + '/item_2hint' + '.pkl', 'wb'))
    print('Spectral Clustering has done! \'item_2hint.pkl\' has been generated!')