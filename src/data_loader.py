import numpy as np
import os
import scipy.sparse as sp
import argparse


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    n_entity, n_relation, adj_entity, adj_relation, idx_nodes, kg= load_kg(args)
    train_item_idx_dict, train_item_next_dict = get_item_idx_dict(train_data)
    eval_item_idx_dict, eval_item_next_dict = get_item_idx_dict(eval_data)
    test_item_idx_dict, test_item_next_dict = get_item_idx_dict(test_data)
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation, idx_nodes, kg, \
           train_item_idx_dict, eval_item_idx_dict, test_item_idx_dict,\
           train_item_next_dict, eval_item_next_dict, test_item_next_dict



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

    train_data = sort_data(train_data)
    eval_data = sort_data(eval_data)
    test_data = sort_data(test_data)

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
    adj_entity, adj_relation = construct_adj_entity(edges, n_entity)

    return n_entity, n_relation, adj_entity, adj_relation, idx_nodes, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    edges = []
    idx_nodes = []
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        key_h_t = str(head) + "_" + str(tail)
        if key_h_t not in kg:
            edges.append((head, relation, tail))
        if head not in idx_nodes:
            idx_nodes.append(head)
        if tail not in idx_nodes:
            idx_nodes.append(tail)
    edges = np.array(edges, dtype=np.int32)
    idx_nodes = np.sort(idx_nodes)
    return kg, edges, idx_nodes


def construct_adj_entity(edges, entity_num):
    print('constructing adjacency entity ...')
    # adj_entity = sp.csr_matrix((np.ones(
    #     (edges.shape[0]), dtype=np.int32), (edges[:, 0], edges[:, 2])),
    #     shape=(entity_num, entity_num))
    adj_relation = sp.csr_matrix((np.array(
        (edges[:, 1]+1), dtype=np.int32), (edges[:, 0], edges[:, 2])),
        shape=(entity_num, entity_num))
    # treat the KG as an undirected graph
    adj_relation += adj_relation.transpose()

    adj_entity = sp.csr_matrix((np.ones(
        (edges.shape[0]), dtype=np.int32), (edges[:, 0], edges[:, 2])),
        shape=(entity_num, entity_num))
    # treat the KG as an undirected graph
    adj_entity += adj_entity.transpose()

    print("adj_relation:", adj_relation)
    print("\n")
    print("adj_entity:", adj_entity)
    return adj_entity, adj_relation


# order by item
def sort_data(data):
    return data[data[:, 1].argsort()]


def get_item_idx_dict(data):
    item_idx_dict = dict()
    item_next_dict = dict()
    temp = data[:, 1]
    former_idx = -1

    for i in range(len(temp)):
        if temp[i] not in item_idx_dict:
            item_idx_dict[temp[i]] = i
            item_next_dict[former_idx] = temp[i]
            former_idx = temp[i]

    # print("item_idx_dict\n:", item_idx_dict)
    # print("item_nexx_dcit\n:", item_nexx_dcit)
    return item_idx_dict, item_next_dict


# if __name__ == '__main__':
#     np.random.seed(555)
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str,  default='music', help='which dataset to preprocess')
#     # parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
#     parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
#
#     args = parser.parse_args()
#     data = load_data(args)
#     train_data, eval_data, test_data = data[4], data[5], data[6]
#
#     print("!!!!get_item_idx_dict(train_data):")
#     get_item_idx_dict(train_data)
#     print("!!!!get_item_idx_dict(test_data):")
#     get_item_idx_dict(test_data)
#     print("!!!! get_item_idx_dict(eval_data):")
#     get_item_idx_dict(eval_data)
#
#     print('done!!!')
