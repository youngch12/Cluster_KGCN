# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collections of partitioning functions."""

import time
import metis
import scipy.sparse as sp
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def partition_graph(adj, idx_nodes, num_clusters):
    """partition a graph by METIS."""

    start_time = time.time()
    num_nodes = len(idx_nodes)

    train_adj_lil = adj[idx_nodes, :][:, idx_nodes].tolil()
    train_ord_map = dict()
    train_adj_lists = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        rows = train_adj_lil[i].rows[0]
        # self-edge needs to be removed for valid format of METIS
        if i in rows:
            rows.remove(i)
        train_adj_lists[i] = rows
        train_ord_map[idx_nodes[i]] = i

    if num_clusters > 1:
        _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=1)
    else:
        groups = [0] * num_nodes

    # parts = [[] for _ in range(num_clusters)]

    for nd_idx in range(num_nodes):
        gp_idx = groups[nd_idx]
        nd_orig_idx = idx_nodes[nd_idx]
        # parts[gp_idx].append(nd_orig_idx)

    tf.logging.info('Partitioning done. %f seconds.', time.time() - start_time)
    print('Partitioning done. %f seconds.', time.time() - start_time)

    return groups, train_ord_map #, parts


def preprocess_multicluster(adj, kg, idx_nodes, groups, train_ord_map,
                            num_clusters, block_size, diag_lambda=-1):
    """Generate the batch for multiple clusters."""

    start_time = time.time()

    # np.random.shuffle(parts)
    # multi_parts = []
    # for _, st in enumerate(range(0, num_clusters, block_size)):
    #     pt = parts[st]
    #     for pt_idx in range(st + 1, min(st + block_size, num_clusters)):
    #         pt = np.concatenate((pt, parts[pt_idx]), axis=0)
    #     multi_parts.append(pt)

    total_adj_entities = []
    total_adj_relations = []
    num_nodes = len(idx_nodes)
    group_ids = [i for i in range(num_clusters)]
    multi_parts_map = dict()
    max_count = 0

    np.random.shuffle(group_ids)
    map_id = 0
    for _, st in enumerate(range(0, num_clusters, block_size)):
        group_id = group_ids[st]
        # print("group_id:", group_id)
        multi_parts_map[group_id] = map_id
        for pt_idx in range(st + 1, min(st + block_size, num_clusters)):
            group_id = group_ids[pt_idx]
            multi_parts_map[group_id] = map_id
        map_id += 1

    # print("multi_parts_map:", multi_parts_map)
    for nd_idx in range(num_nodes):
        count = 0
        times = 0
        adj_entities = []
        adj_relations = []
        gp_idx = groups[nd_idx]
        nd_orig_idx = idx_nodes[nd_idx]

        for nb_orig_idx in adj[nd_orig_idx].indices:
            nb_idx = train_ord_map[nb_orig_idx]
            nb_gp_idx = groups[nb_idx]
            # if a node's neighbor and this node are still in the same concated sub-graph
            if multi_parts_map[nb_gp_idx] == multi_parts_map[gp_idx]:
                relation_times = adj[nd_orig_idx].data[count]
                key = str(nd_orig_idx) + "_" + str(nb_orig_idx)
                for i in range(relation_times):
                    adj_entities.append(nb_orig_idx)
                    adj_relations.append(kg[key][i])
                    times += 1
                if times > max_count:
                    max_count = times
            count += 1


        total_adj_entities.append(adj_entities)
        total_adj_relations.append(adj_relations)

    print("max_count:", max_count)

    total_adj_entities = np.hstack(np.insert(total_adj_entities, range(1, len(total_adj_entities) + 1), [[0] * (max_count - len(i))
                                                                            for i in total_adj_entities])).astype('int32').reshape(len(total_adj_entities), max_count)
    total_adj_relations = np.hstack(np.insert(total_adj_relations, range(1, len(total_adj_relations) + 1), [[0] * (max_count - len(i))
                                                                            for i in total_adj_relations])).astype('int32').reshape(len(total_adj_relations), max_count)
    tf.logging.info('Preprocessing multi-cluster done. %f seconds.', time.time() - start_time)
    print('Preprocessing multi-cluster done. %f seconds.', time.time() - start_time)

    return total_adj_entities, total_adj_relations


# def sparse_to_tuple(sparse_mx):
#   """Convert sparse matrix to tuple representation."""
#
#   def to_tuple(mx):
#     if not sp.isspmatrix_coo(mx):
#       mx = mx.tocoo()
#     coords = np.vstack((mx.row, mx.col)).transpose()
#     values = mx.data
#     shape = mx.shape
#     return coords, values, shape
#
#   if isinstance(sparse_mx, list):
#     for i in range(len(sparse_mx)):
#       sparse_mx[i] = to_tuple(sparse_mx[i])
#   else:
#     sparse_mx = to_tuple(sparse_mx)
#
#   return sparse_mx
#
#
# def normalize_adj(adj):
#   rowsum = np.array(adj.sum(1)).flatten()
#   d_inv = 1.0 / (np.maximum(1.0, rowsum))
#   d_mat_inv = sp.diags(d_inv, 0)
#   adj = d_mat_inv.dot(adj)
#   return adj
#
#
# def normalize_adj_diag_enhance(adj, diag_lambda):
#   """Normalization by  A'=(D+I)^{-1}(A+I), A'=A'+lambda*diag(A')."""
#   adj = adj + sp.eye(adj.shape[0])
#   rowsum = np.array(adj.sum(1)).flatten()
#   d_inv = 1.0 / (rowsum + 1e-20)
#   d_mat_inv = sp.diags(d_inv, 0)
#   adj = d_mat_inv.dot(adj)
#   adj = adj + diag_lambda * sp.diags(adj.diagonal(), 0)
#   return adj
