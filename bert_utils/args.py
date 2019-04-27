import os
from pathlib import Path
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# file_path = os.path.dirname(__file__)
file_path = '/home/redis/model'

model_dir = os.path.join(file_path, 'chinese_L-12_H-768_A-12/')
config_name = os.path.join(model_dir, 'bert_config.json')
ckpt_name = os.path.join(model_dir, 'bert_model.ckpt')
vocab_file = os.path.join(model_dir, 'vocab.txt')
output_dir = os.path.join(file_path, 'tmp/wiki_result/')
data_dir = os.path.join(file_path, 'dataset/WikiQACorpus/')

num_train_epochs = 10
batch_size = 8
learning_rate = 0.00005

# gpu使用率
gpu_memory_fraction = 0.8

# 默认取倒数第二层的输出值作为句向量
layer_indexes = [-2]

# 序列的最大程度，单文本建议把该值调小
max_seq_len = 32
