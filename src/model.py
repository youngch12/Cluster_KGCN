# import tensorflow as tf
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator
from sklearn.metrics import f1_score, roc_auc_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


class KGCN(object):
    # def __init__(self, args, n_user, parts_n_entities, parts_n_relations, parts_adj_entities, parts_adj_relations, parts, node_parts_map):
    #     #     self._parse_args(args, parts_adj_entities, parts_adj_relations, parts, node_parts_map)
    #     #     self._build_inputs()
    #     #     self._build_model(n_user, parts_n_entities, parts_n_relations)
    #     #     self._build_train()
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation):
        self._parse_args(args, adj_entity, adj_relation)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.glorot_uniform_initializer()

    # def _parse_args(self, args, parts_adj_entities, parts_adj_relations, parts, node_parts_map):
    #     # [entity_num, neighbor_sample_size]
    #     self.parts_adj_entities = parts_adj_entities
    #     self.parts_adj_relations = parts_adj_relations
    #     self.parts = parts
    #     self.node_parts_map = node_parts_map
    #     self.num_clusters = args.num_clusters
    #
    #     self.n_iter = args.n_iter
    #     self.batch_size = args.batch_size
    #
    #     # del
    #     # self.n_neighbor = args.neighbor_sample_size
    #     # del
    #
    #     self.dim = args.dim
    #     self.l2_weight = args.l2_weight
    #     self.lr = args.lr
    #     if args.aggregator == 'sum':
    #         self.aggregator_class = SumAggregator
    #     elif args.aggregator == 'concat':
    #         self.aggregator_class = ConcatAggregator
    #     elif args.aggregator == 'neighbor':
    #         self.aggregator_class = NeighborAggregator
    #     else:
    #         raise Exception("Unknown aggregator: " + args.aggregator)

    def _parse_args(self, args, adj_entity, adj_relation):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        # self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

    # def _build_model(self, n_user, parts_n_entities, parts_n_relations):
    #     self.user_emb_matrix = tf.get_variable(
    #         shape=[n_user, self.dim], initializer=KGCN.get_initializer(), name='user_emb_matrix')
    #
    #     part_entity_emb_matrix = [[] for _ in range(self.num_clusters)]
    #     part_relation_emb_matrix = [[] for _ in range(self.num_clusters)]
    #
    #     for i in range(self.num_clusters):
    #         entity_emb_matrix = tf.get_variable(
    #             shape=[parts_n_entities[i], self.dim], initializer=KGCN.get_initializer(), name='entity_emb_matrix')
    #         relation_emb_matrix = tf.get_variable(
    #             shape=[parts_n_relations[i], self.dim], initializer=KGCN.get_initializer(), name='relation_emb_matrix')
    #
    #         part_entity_emb_matrix[i].append(entity_emb_matrix)
    #         part_relation_emb_matrix[i].append(relation_emb_matrix)
    #
    #     self.part_entity_emb_matrix = part_entity_emb_matrix
    #     self.part_relation_emb_matrix = part_relation_emb_matrix
    #
    #     # self.entity_emb_matrix = tf.get_variable(
    #     #     shape=[n_entity, self.dim], initializer=KGCN.get_initializer(), name='entity_emb_matrix')
    #     # self.relation_emb_matrix = tf.get_variable(
    #     #     shape=[n_relation, self.dim], initializer=KGCN.get_initializer(), name='relation_emb_matrix')
    #
    #     # [batch_size, dim]
    #     self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
    #
    #     # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
    #     # dimensions of entities:
    #     # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
    #     entities, relations = self.get_neighbors(self.item_indices)
    #
    #     # [batch_size, dim]
    #     self.item_embeddings, self.aggregators = self.aggregate(entities, relations)
    #
    #     # [batch_size]
    #     self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
    #     self.scores_normalized = tf.sigmoid(self.scores)

    def _build_model(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=KGCN.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=KGCN.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=KGCN.get_initializer(), name='relation_emb_matrix')

        # [batch_size, dim]
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities, relations = self.get_neighbors(self.item_indices)

        # [batch_size, dim]
        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)

        # [batch_size]
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)
    # def get_neighbors(self, seeds):
    #     seeds = tf.expand_dims(seeds, axis=1)
    #     entities = [seeds]
    #     relations = []
    #     entity_vectors = []
    #     relation_vectors = []
    #     for i in range(self.n_iter):
    #
    #         neighbor_entities = []
    #         neighbor_relations = []
    #         for idx in entities[i]:
    #             gp_idx = self.node_parts_map[idx]
    #             part = self.parts[gp_idx]
    #             adj_idx = part.index(idx)
    #             adj_entities = self.parts_adj_entities[gp_idx]
    #             adj_relations = self.parts_adj_relations[gp_idx]
    #             neighbor_entities.append(adj_entities[adj_idx])
    #             neighbor_relations.append(adj_relations[adj_idx])
    #
    #         neighbor_entities = tf.reshape(neighbor_entities, [self.batch_size, -1])
    #         neighbor_relations = tf.reshape(neighbor_relations, [self.batch_size, -1])
    #
    #         entity_vectors
    #
    #         entities.append(neighbor_entities)
    #         relations.append(neighbor_relations)
    #
    #     return entities, relations

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations


    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            max_len = self.adj_entity.shape[1]
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, max_len, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)
