# import tensorflow as tf
import numpy as np
import time
import psutil as ps
from model import KGCN
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder


def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]

    model = KGCN(args, n_user, n_entity, n_relation, adj_entity, adj_relation)

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # monitor the usage of memory while training the model
        profiler = model_analyzer.Profiler(graph=sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        # tensor-board
        writer = tf.summary.FileWriter('../data/' + args.dataset + '/logs', tf.get_default_graph())

        for step in range(args.n_epochs):
            # training
            t = time.time()
            np.random.shuffle(train_data)
            start = 0
            i = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size),
                                      run_options, run_metadata)
                # add the data into tfprofiler
                profiler.add_step(step=step, run_meta=run_metadata)
                if i == 0:
                    writer.add_run_metadata(run_metadata, 'step %d' % step)
                i += 1
                start += args.batch_size
                # if show_loss:
                #     print(start, loss)

            # CTR evaluation
            train_auc, train_f1 = ctr_eval(sess, model, train_data, args.batch_size)
            eval_auc, eval_f1 = ctr_eval(sess, model, eval_data, args.batch_size)
            test_auc, test_f1 = ctr_eval(sess, model, test_data, args.batch_size)

            # values = ps.virtual_memory()
            # used_memory = values.used / (1024.0 ** 3)
            train_time = time.time() - t

            # print('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
            #       % (step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))
            print(
                'epoch %d   training time: %.5f    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
                % (step, train_time, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))

        # # 统计模型的memory使用大小
        profile_scope_opt_builder = option_builder.ProfileOptionBuilder(
            option_builder.ProfileOptionBuilder.trainable_variables_parameter())
        # 显示字段是params，即参数
        profile_scope_opt_builder.select(['params'])
        # 根据params数量进行显示结果排序
        profile_scope_opt_builder.order_by('params')
        # 显示视图为scope view
        profiler.profile_name_scope(profile_scope_opt_builder.build())

        # ------------------------------------
        # 最耗时top 5 ops
        profile_op_opt_builder = option_builder.ProfileOptionBuilder()

        # 显示字段：op执行时间，使用该op的node的数量。 注意：op的执行时间即所有使用该op的node的执行时间总和。
        profile_op_opt_builder.select(['micros', 'occurrence'])
        # 根据op执行时间进行显示结果排序
        profile_op_opt_builder.order_by('micros')
        # 过滤条件：只显示排名top 7
        profile_op_opt_builder.with_max_depth(6)

        # 显示视图为op view
        profiler.profile_operations(profile_op_opt_builder.build())

        # ------------------------------------
        writer.close()


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


def ctr_eval(sess, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1 = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
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
