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
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math


def partition_graph(adj, idx_nodes, num_clusters):
    """partition a graph by METIS."""

    start_time = time.time()
    num_nodes = len(idx_nodes)
    num_all_nodes = adj.shape[0]

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

    # tf.logging.info('Partitioning done. %f seconds.', time.time() - start_time)
    print('Partitioning done. %f seconds.', time.time() - start_time)

    return groups, train_ord_map #, parts


def preprocess_multicluster(adj, kg, idx_nodes, groups, train_ord_map,
                            num_clusters, block_size,
                            train_data, eval_data, test_data,
                            train_item_idx_dict, eval_item_idx_dict, test_item_idx_dict,
                            train_item_next_dict, eval_item_next_dict, test_item_next_dict):
    """Generate the batch for multiple clusters."""

    start_time = time.time()

    num_nodes = len(idx_nodes)
    group_ids = [i for i in range(num_clusters)]
    multi_parts_map = dict()

    np.random.shuffle(group_ids)
    map_id = 0
    for _, st in enumerate(range(0, num_clusters, block_size)):
        group_id = group_ids[st]
        multi_parts_map[group_id] = map_id
        for pt_idx in range(st + 1, min(st + block_size, num_clusters)):
            group_id = group_ids[pt_idx]
            multi_parts_map[group_id] = map_id
        map_id += 1

    multi_adj_entities = [[] for i in range(math.ceil(num_clusters / block_size))]
    multi_adj_relations = [[] for i in range(math.ceil(num_clusters / block_size))]

    train_data_multi_map = [[] for i in range(math.ceil(num_clusters / block_size))]
    eval_data_multi_map = [[] for i in range(math.ceil(num_clusters / block_size))]
    test_data_multi_map = [[] for i in range(math.ceil(num_clusters / block_size))]

    train_multi_map_idx = [0 for i in range(math.ceil(num_clusters / block_size))]
    eval_multi_map_idx = [0 for i in range(math.ceil(num_clusters / block_size))]
    test_multi_map_idx = [0 for i in range(math.ceil(num_clusters / block_size))]

    multi_map_idx = [0 for i in range(math.ceil(num_clusters / block_size))]
    new_entity_id_dict = dict()

    for nd_idx in range(num_nodes):
        count = 0
        times = 0
        adj_entities = []
        adj_relations = []
        gp_idx = groups[nd_idx]
        nd_orig_idx = idx_nodes[nd_idx]
        map_id = multi_parts_map[gp_idx]

        # partition train_data
        train_map_idx = train_multi_map_idx[map_id]
        if nd_orig_idx in train_item_idx_dict:
            nd_item_idx = train_item_idx_dict[nd_orig_idx]
            if nd_orig_idx in train_item_next_dict:
                nd_next_item_id = train_item_next_dict[nd_orig_idx]
                nd_next_item_idx = train_item_idx_dict[nd_next_item_id]
                temp_list = train_data[nd_item_idx:nd_next_item_idx]
            else:
                temp_list = train_data[nd_item_idx:]

            y = [[train_map_idx]] * len(temp_list)
            temp_list = np.concatenate((temp_list, y), axis=1)
            if len(train_data_multi_map[map_id]) == 0:
                train_data_multi_map[map_id] = temp_list
            else:
                train_data_multi_map[map_id] = np.concatenate((train_data_multi_map[map_id], temp_list))
            train_multi_map_idx[map_id] += 1

        # partition eval_data
        eval_map_idx = eval_multi_map_idx[map_id]
        if nd_orig_idx in eval_item_idx_dict:
            nd_item_idx = eval_item_idx_dict[nd_orig_idx]
            if nd_orig_idx in eval_item_next_dict:
                nd_next_item_id = eval_item_next_dict[nd_orig_idx]
                nd_next_item_idx = eval_item_idx_dict[nd_next_item_id]
                temp_list = eval_data[nd_item_idx:nd_next_item_idx]
            else:
                temp_list = eval_data[nd_item_idx:]

            y = [[eval_map_idx]] * len(temp_list)
            temp_list = np.concatenate((temp_list, y), axis=1)
            if len(eval_data_multi_map[map_id]) == 0:
                eval_data_multi_map[map_id] = temp_list
            else:
                eval_data_multi_map[map_id] = np.concatenate((eval_data_multi_map[map_id], temp_list))
            eval_multi_map_idx[map_id] += 1

        # partition test_data
        test_map_idx = test_multi_map_idx[map_id]
        if nd_orig_idx in test_item_idx_dict:
            nd_item_idx = test_item_idx_dict[nd_orig_idx]
            if nd_orig_idx in test_item_next_dict:
                nd_next_item_id = test_item_next_dict[nd_orig_idx]
                nd_next_item_idx = test_item_idx_dict[nd_next_item_id]
                temp_list = test_data[nd_item_idx:nd_next_item_idx]
            else:
                temp_list = test_data[nd_item_idx:]

            y = [[test_map_idx]] * len(temp_list)
            temp_list = np.concatenate((temp_list, y), axis=1)
            if len(test_data_multi_map[map_id]) == 0:
                test_data_multi_map[map_id] = temp_list
            else:
                test_data_multi_map[map_id] = np.concatenate((test_data_multi_map[map_id], temp_list))
            test_multi_map_idx[map_id] += 1


        for nb_orig_idx in adj[nd_orig_idx].indices:
            nb_idx = train_ord_map[nb_orig_idx]
            nb_gp_idx = groups[nb_idx]
            # if a node's neighbor and this node are still in the same concated sub-graph
            if multi_parts_map[nb_gp_idx] == map_id:
                relation_times = adj[nd_orig_idx].data[count]
                key = str(nd_orig_idx) + "_" + str(nb_orig_idx)
                new_entity_id_key = str(map_id) + "_" + str(nb_orig_idx)
                if new_entity_id_key in new_entity_id_dict:
                    new_entity_id = new_entity_id_dict[new_entity_id_key]
                else:
                    new_entity_id = multi_map_idx[map_id]
                    new_entity_id_dict[new_entity_id_key] = new_entity_id
                    multi_map_idx[map_id] += 1
                for i in range(relation_times):
                    # adj_entities.append(nb_orig_idx)
                    adj_entities.append(new_entity_id)
                    adj_relations.append(kg[key][i])
                    times += 1
            count += 1

        # sample adj_value
        n_neighbors = times
        if n_neighbors == 0:
            print("No neighbor, nd_orig_idx: ", nd_orig_idx)
            adj_entities = [0]
            adj_relations = [0]
            # n_neighbors = 1

        # if n_neighbors >= neighbor_sample_size:
        #     sampled_indices = np.random.choice(list(range(n_neighbors)), size=neighbor_sample_size, replace=False)
        # else:
        #     sampled_indices = np.random.choice(list(range(n_neighbors)), size=neighbor_sample_size, replace=True)

        multi_adj_entities[map_id].append(np.array(adj_entities[i]))
        multi_adj_relations[map_id].append(np.array(adj_relations[i]))

    print('Preprocessing multi-cluster done. %f seconds.', time.time() - start_time)
    print('train_multi_map_idx:', train_multi_map_idx)

    return multi_adj_entities, multi_adj_relations, train_data_multi_map, eval_data_multi_map, test_data_multi_map


def sparse_to_tuple(sparse_mx):
  """Convert sparse matrix to tuple representation."""

  def to_tuple(mx):
    if not sp.isspmatrix_coo(mx):
      mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    shape = mx.shape
    return coords, values, shape

  if isinstance(sparse_mx, list):
    for i in range(len(sparse_mx)):
      sparse_mx[i] = to_tuple(sparse_mx[i])
  else:
    sparse_mx = to_tuple(sparse_mx)

  return sparse_mx
