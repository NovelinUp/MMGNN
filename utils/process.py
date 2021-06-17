import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from scipy.sparse import csr_matrix

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""

def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return indices, adj.data, adj.shape


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset):
    if dataset == 'acm':
        data = sio.loadmat('./data/preprocess/new_acm.mat')
        author_features, paper_features, subject_features = data['author_feature'].toarray(), \
                                                            data['paper_feature'].toarray(), \
                                                            data['subject_feature'].toarray()

        author_N, paper_N, subject_N = author_features.shape[0], paper_features.shape[0], subject_features.shape[0]

        features_list = [author_features, subject_features, paper_features]
        homo_adj_list = [data['APA']- csr_matrix(np.eye(author_N)), data['SPS']- csr_matrix(np.eye(subject_N)), \
                         data['PAP']-csr_matrix(np.eye(paper_N)), data['PSP']-csr_matrix(np.eye(paper_N))]
        hete_adj_list = [data['PA'].toarray().astype('float32'), data['PS'].toarray().astype('float32')]

        y = data['paper_label'].toarray()
        train_idx = data['train_paper_idx']
        val_idx = data['val_paper_idx']
        test_idx = data['test_paper_idx']

    elif dataset == 'imdb':
        data = sio.loadmat('./data/preprocess/new_imdb.mat')
        movie_features, actor_features, director_features = data['movie_feature'].toarray(), \
                                                            data['actor_feature'].toarray(), \
                                                            data['director_feature'].toarray()
        movie_N, actor_N, director_N = movie_features.shape[0], actor_features.shape[0], director_features.shape[0]

        features_list = [actor_features, director_features, movie_features]
        homo_adj_list = [data['AMA']-csr_matrix(np.eye(actor_N)), data['DMD']-csr_matrix(np.eye(director_N)), \
                         data['MAM']-csr_matrix(np.eye(movie_N)), data['MDM']-csr_matrix(np.eye(movie_N))]
        hete_adj_list = [data['MA'].toarray().astype('float32'), data['MD'].toarray().astype('float32')]

        y = data['movie_label'].toarray()
        train_idx = data['train_movie_idx']
        val_idx = data['val_movie_idx']
        test_idx = data['test_movie_idx']

    train_mask = sample_mask(train_idx, y.shape[0])
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # print label and idx info
    print('train_idx:{}, val_idx:{}, test_idx:{}'.format(train_idx.shape, val_idx.shape, test_idx.shape))

    return homo_adj_list, hete_adj_list, features_list, y_train, y_val, y_test, train_mask, val_mask, test_mask
