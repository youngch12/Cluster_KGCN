import numpy as np
import os
import scipy.sparse as sp


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    # n_entity, n_relation, adj_entity, adj_relation = load_kg(args)
    n_entity, n_relation, adj_entity, idx_nodes, kg = load_kg(args)
    # node_feature = load_node_feature(args)
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, idx_nodes \
        , kg


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np, args)

    return n_user, n_item, train_data, eval_data, test_data


def dataset_split(rating_np, args):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if args.ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg, edges, idx_nodes = construct_kg(kg_np)
    # adj_entity, adj_relation = construct_adj(args, kg, edges, n_entity)
    adj_entity = construct_adj_entity(edges, n_entity)

    return n_entity, n_relation, adj_entity, idx_nodes, kg  # , adj_relation


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    edges = []
    idx_nodes = []
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        edges.append((head, relation, tail))
        # # treat the KG as an undirected graph
        # if head not in kg:
        #     kg[head] = []
        # kg[head].append((tail, relation))
        # if tail not in kg:
        #     kg[tail] = []
        # kg[tail].append((head, relation))
        key_a = str(head) + "_" + str(tail)
        key_b = str(tail) + "_" + str(head)
        if key_a not in kg:
            kg[key_a] = []
        kg[key_a].append(relation)
        if key_b not in kg:
            kg[key_b] = []
        kg[key_b].append(relation)

        if head not in idx_nodes:
            idx_nodes.append(head)
        if tail not in idx_nodes:
            idx_nodes.append(tail)
    edges = np.array(edges, dtype=np.int32)
    return kg, edges, idx_nodes


def construct_adj_entity(edges, entity_num):
    print('constructing adjacency entity ...')
    adj_entity = sp.csr_matrix((np.ones(
        (edges.shape[0]), dtype=np.int32), (edges[:, 0], edges[:, 2])),
        shape=(entity_num, entity_num))
    # adj_entity = sp.csr_matrix((np.array(
    #     (edges[:, 1]), dtype=np.int32), (edges[:, 0], edges[:, 2])),
    #     shape=(entity_num, entity_num))
    # treat the KG as an undirected graph
    adj_entity += adj_entity.transpose()
    return adj_entity


def load_node_feature(args):
    print("loading node feature ...")
    feature_file = '../data/' + args.dataset + '/node_feature'
    feature_np = np.load(feature_file + '.npy')
    return feature_np
