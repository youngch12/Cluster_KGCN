import argparse
import numpy as np
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

RATING_FILE_NAME = dict({'movie': 'ratings.csv', 'book': 'BX-Book-Ratings.csv', 'music': 'user_artists.dat'})
SEP = dict({'movie': ',', 'book': ';', 'music': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'music': 0})


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + DATASET + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))

    return n_entity


def form_node_feature(dim):
    print('forming node_feature ...')

    n_entity = load_kg(args)

    entity_emb_matrix = tf.get_variable(
        shape=[n_entity, dim], initializer=tf.glorot_uniform_initializer(), name='entity_emb_matrix')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        entity_emb_matrix = sess.run(entity_emb_matrix)
        print("entity_emb_matrix:", entity_emb_matrix.shape)

    file_name = '../data/' + DATASET + '/node_feature'
    np.save(file_name + '.npy', entity_emb_matrix)


def load_node_feature():
    print("loading data ...")

    rating_file = '../data/' + DATASET + '/node_feature'
    feature_np = np.load(rating_file + '.npy')



if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='music', help='which dataset to preprocess')
    parser.add_argument('-dim', type=int, default=16, help='dimension of user and entity embeddings')

    args = parser.parse_args()
    DATASET = args.d

    form_node_feature(args.dim)

    # load_node_feature()

    print('forming node_feature done')