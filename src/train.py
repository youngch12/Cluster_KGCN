# import tensorflow as tf
import numpy as np
import time
import models
import partition_utils
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math


def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj, idx_nodes, kg = data[7], data[8], data[9]

    groups, train_ord_map = partition_utils.partition_graph(adj, idx_nodes, args.num_clusters)
    # pre-process multi-clusters
    # total_adj_entities, total_adj_relations = \
    #     partition_utils.preprocess_multicluster(adj, kg, idx_nodes, groups, train_ord_map, args.num_clusters, args.block_size, args.neighbor_sample_size, diag_lambda=-1)

    # pre-process multi-clusters

    group_ids = list(range(math.ceil(args.num_clusters / args.block_size)))
    multi_adj_entities, multi_adj_relations, train_data_multi_map, eval_data_multi_map, test_data_multi_map = \
        partition_utils.preprocess_multicluster(adj, kg, idx_nodes, groups, train_ord_map, args.num_clusters,
                                                args.block_size, args.neighbor_sample_size,
                                                train_data, eval_data, test_data)

    # model = KGCN(args, n_user, n_entity, n_relation, total_adj_entities, total_adj_relations)

    # Some preprocessing
    model_func = models.KGCN

    # Define placeholders
    placeholders = {
        'adj_entity':
            tf.placeholder(tf.int64),
        'adj_relation':
            tf.placeholder(tf.int64),
        'user_indices':
            tf.placeholder(tf.int64),
        'item_indices':
            tf.placeholder(tf.int64),
        'labels':
            tf.placeholder(tf.float32)
    }

    # Create model
    model = model_func(
        placeholders,
        args=args,
        n_user=n_user,
        n_entity=n_entity,
        n_relation=n_relation)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(args.n_epochs):
            # training
            t = time.time()
            np.random.shuffle(group_ids)
            for pid in group_ids:
                start = 0
                train_data = np.array(train_data_multi_map[pid])
                np.random.shuffle(train_data)
                # skip the last incomplete mini-batch if its size < batch size
                while start + args.batch_size <= train_data.shape[0]:
                    # _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                    feed_dict = construct_feed_dict(
                        multi_adj_entities[pid], multi_adj_relations[pid], train_data, start, start + args.batch_size,
                        placeholders)
                    _, loss = model.train(sess, feed_dict=feed_dict)
                    start += args.batch_size
                    if show_loss:
                        print(start, loss)

            # CTR evaluation
            train_auc, train_f1 = ctr_eval(sess, model, multi_adj_entities, multi_adj_relations, train_data_multi_map,
                                           args.batch_size, group_ids, placeholders)
            eval_auc, eval_f1 = ctr_eval(sess, model, multi_adj_entities, multi_adj_relations, eval_data_multi_map,
                                         args.batch_size, group_ids, placeholders)
            test_auc, test_f1 = ctr_eval(sess, model, multi_adj_entities, multi_adj_relations, test_data_multi_map,
                                         args.batch_size, group_ids, placeholders)
            # train_auc, train_f1 = ctr_eval(sess, model, train_data, args.batch_size, group_ids)
            # eval_auc, eval_f1 = ctr_eval(sess, model, eval_data, args.batch_size, group_ids)
            # test_auc, test_f1 = ctr_eval(sess, model, test_data, args.batch_size, group_ids)
            train_time = time.time() - t
            print(
                'epoch %d   training time: %.5f   train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
                % (step, train_time, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def construct_feed_dict(adj_entity, adj_relation, data, start, end, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['adj_entity']: adj_entity})
    feed_dict.update({placeholders['adj_relation']: adj_relation})
    feed_dict.update({placeholders['user_indices']: data[start:end, 0]})
    feed_dict.update({placeholders['item_indices']: data[start:end, 1]})
    feed_dict.update({placeholders['labels']: data[start:end, 2]})
    # feed_dict.update({placeholders['entity_emb_matrix']: tf.get_variable(
    #     shape=[len(adj_entity), dim], initializer=tf.glorot_uniform_initializer(), name='entity_emb_matrix')})
    return feed_dict


def ctr_eval(sess, model, multi_adj_entities, multi_adj_relations, data_multi_map, batch_size, group_ids,
             placeholders):
    auc_list = []
    f1_list = []
    for pid in group_ids:
        start = 0
        data = np.array(data_multi_map[pid])
        while start + batch_size <= data.shape[0]:
            feed_dict = construct_feed_dict(
                multi_adj_entities[pid], multi_adj_relations[pid], data, start, start + batch_size, placeholders)
            auc, f1 = model.eval(sess, feed_dict=feed_dict)

            auc_list.append(auc)
            f1_list.append(f1)
            start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
