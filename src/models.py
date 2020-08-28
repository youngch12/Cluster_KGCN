# import tensorflow as tf
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Model(object):
    def __init__(self, placeholders, args, n_user, n_entity, n_relation):
        self._parse_args(args, n_user, n_entity, n_relation)
        # self._build_inputs()
        # self._build_model(self, n_user, n_relation)
        # self._build_train()

    @staticmethod
    def get_initializer():
        return tf.glorot_uniform_initializer()

    def _parse_args(self, args, n_user, n_entity, n_relation):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = None
        self.adj_relation = None
        self.entity_emb_matrix = None
        self.n_user = n_user
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.placeholders = {}
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.optimizer = None
        self.aggregators = []
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    # def _build_inputs(self):
    #     self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
    #     self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
    #     self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

    def build(self):
        self._build_model()
        self._build_train()

    def _build_model(self):
        self.user_emb_matrix = tf.get_variable(
            shape=[self.n_user, self.dim], initializer=KGCN.get_initializer(), name='user_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[self.n_relation, self.dim], initializer=KGCN.get_initializer(), name='relation_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[self.n_entity, self.dim], initializer=tf.glorot_uniform_initializer(), name='entity_emb_matrix')

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

        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, entities[0])]
        neighbors_entities_val = []
        neighbors_relations_val = []

        neighbors_entities_val = tf.nn.embedding_lookup(self.entity_emb_matrix, self.entities_indices)
        neighbors_relations_val = tf.nn.embedding_lookup(self.relation_emb_matrix, self.relations_indices)

        n_adj_entities = tf.cast(tf.shape(self.adj_entity)[0], tf.int32)

        neighbor_vectors = tf.sparse.SparseTensor(indices=self.neighbors_indices, values=neighbors_entities_val,
                                                  dense_shape=[self.batch_size, n_adj_entities])

        neighbor_relations = tf.sparse.SparseTensor(indices=self.neighbors_indices, values=neighbors_relations_val,
                                                    dense_shape=[self.batch_size, n_adj_entities])

        # neighbor_vectors = tf.sparse_to_dense(self.neighbors_indices,
        #                                       [self.batch_size, n_adj_entities],
        #                                       neighbors_entities_val,
        #                                       # default_value=np.zeros(self.dim),
        #                                       # default_value=tf.get_variable(shape=[1, self.dim],
        #                                       #                               initializer=tf.zeros_initializer(),
        #                                       #                               name='default_value1'),
        #                                       validate_indices=True, name='neighbor_vectors')
        # print("neighbor_vectors:", neighbor_vectors)
        # neighbor_relations = tf.sparse_to_dense(self.neighbors_indices,
        #                                         [self.batch_size, n_adj_entities],
        #                                         neighbors_relations_val,
        #                                         default_value=np.zeros(self.dim),
        #                                         # default_value=np.zeros([1, self.dim]),
        #                                         # default_value=tf.get_variable(shape=[1, self.dim],
        #                                         #                               initializer=tf.zeros_initializer(),
        #                                         #                               name='default_value2'),
        #                                         validate_indices=True, name='neighbor_relations')


        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, n_adj_entities, self.dim]
                vector = aggregator(self_vectors=entity_vectors,
                                    neighbor_vectors=neighbor_vectors,
                                    neighbor_relations=neighbor_relations,
                                    user_embeddings=self.user_embeddings)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter[-1]

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

        self.opt_op = self.optimizer.minimize(self.loss)

    def train(self, sess, feed_dict, run_options, run_metadata):
        return sess.run([self.opt_op, self.loss], feed_dict, options=run_options, run_metadata=run_metadata)

    def train(self, sess, feed_dict):
        return sess.run([self.opt_op, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)


class KGCN(Model):
  """Implementation of KGCN model."""

  def __init__(self, placeholders, args, n_user, n_entity, n_relation):
    super(KGCN, self).__init__(placeholders, args, n_user, n_entity, n_relation)

    self.adj_entity = placeholders['adj_entity']
    self.adj_relation = placeholders['adj_relation']
    self.user_indices = placeholders['user_indices']
    self.item_indices = placeholders['item_indices']
    self.cluster_item_indices = placeholders['cluster_item_indices']
    self.labels = placeholders['labels']
    self.neighbors_indices = placeholders['neighbors_indices']
    self.entities_indices = placeholders['entities_indices']
    self.relations_indices = placeholders['relations_indices']

    self.optimizer = tf.train.AdamOptimizer(args.lr)
    self.placeholders = placeholders

    self.build()