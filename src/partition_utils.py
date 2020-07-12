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


def partition_graph(adj, idx_nodes, num_clusters, kg):
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

  # node_parts_map = dict()
  # parts = [[] for _ in range(num_clusters)]
  # parts_adj_entities = [[] for _ in range(num_clusters)]
  # parts_adj_relations = [[] for _ in range(num_clusters)]
  total_adj_entities = []
  total_adj_relations = []
  max_count = 0

  for nd_idx in range(num_nodes):
    adj_entities = []
    adj_relations = []
    gp_idx = groups[nd_idx]
    nd_orig_idx = idx_nodes[nd_idx]
    # parts[gp_idx].append(nd_orig_idx)
    # node_parts_map[nd_orig_idx] = gp_idx
    count = 0
    times = 0
    for nb_orig_idx in adj[nd_orig_idx].indices:
      nb_idx = train_ord_map[nb_orig_idx]
      # if a node's neighbor and this node are still in the same sub-graph
      if groups[nb_idx] == gp_idx:
        relation_times = adj[nd_orig_idx].data[count]
        key = str(nd_orig_idx) + "_" + str(nb_orig_idx)
        for i in range(relation_times):
          adj_entities.append(nb_orig_idx)
          adj_relations.append(kg[key][i])
          times += 1
      count += 1
      if times > max_count:
        max_count = times

    total_adj_entities.append(adj_entities)
    total_adj_relations.append(adj_relations)

    # parts_adj_entities[gp_idx].append(adj_entities)
    # parts_adj_relations[gp_idx].append(adj_relations)

  print("max_count:", max_count)

  total_adj_entities = np.hstack(np.insert(total_adj_entities, range(1, len(total_adj_entities) + 1), [[0] * (max_count - len(i))
                                                                          for i in total_adj_entities])).astype('int32').reshape(len(total_adj_entities), max_count)
  total_adj_relations = np.hstack(np.insert(total_adj_relations, range(1, len(total_adj_relations) + 1), [[0] * (max_count - len(i))
                                                                          for i in total_adj_relations])).astype('int32').reshape(len(total_adj_relations), max_count)
  tf.logging.info('Partitioning done. %f seconds.', time.time() - start_time)
  print('Partitioning done. %f seconds.', time.time() - start_time)

  # return parts_adj_entities, parts_adj_relations, parts, node_parts_map
  return total_adj_entities, total_adj_relations


def get_parts_n(parts_adj_relations, parts, num_clusters):
  parts_n_entities = [[] for _ in range(num_clusters)]
  parts_n_relations = [[] for _ in range(num_clusters)]

  for i in range(num_clusters):
    parts_n_entities[i].append(len(parts[i]))
    parts_n_relations[i].append(len(set(parts_adj_relations[i])))

  return parts_n_entities, parts_n_relations

