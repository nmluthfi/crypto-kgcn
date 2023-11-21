import argparse
import numpy as np
from time import time
from data_loader import load_data
from model import KGCN
from train import train, predict_top_k_for_user
import tensorflow as tf

np.random.seed(555)


parser = argparse.ArgumentParser()

# movie
# parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
# parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
# parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
# parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
# parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

# crypto
parser.add_argument('--dataset', type=str, default='crypto', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

'''
# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=3, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
'''

'''
# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
'''


show_loss = False
show_time = True
show_topk = False

t = time()

args = parser.parse_args()
data = load_data(args)
train(args, data, show_loss, show_topk)


if show_time:
    print('time used: %d s' % (time() - t))

# reset the graph before defining a new model
tf.reset_default_graph()
# creating the mode;
model = KGCN(args, data[0], data[2], data[3], data[7], data[8])

# open the file containing the mapping from item indices to item names
index_to_name_mapping = {}
item_name_file_path = '..\data\crypto\item_index2item_name.txt'
with open(item_name_file_path, 'r') as file:
    for line in file:
        index, name = line.strip().split('\t')
        index_to_name_mapping[int(index)] = name

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Predict top-3 for a specific user (replace USER_INDEX with the desired user index)
    user_index_to_predict = 1
    max_item_id = 7277
    top_3_recommendations = predict_top_k_for_user(sess, model, user_index_to_predict, data[1], k=3, batch_size=args.batch_size, max_item_id=max_item_id)

    # Map indices to item names
    result_names = [index_to_name_mapping[index] for index in top_3_recommendations]

    print(f'Top 3 recommended cryptocurrencies for userId {user_index_to_predict}: {result_names}')