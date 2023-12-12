from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
import os
import sys
sys.path.append('..')
import time
import random
import uuid # for generating a unique id for the cnn

import relation_extraction.data.utils as data_utils
from relation_extraction.data.converters.converter_i2b2 import relation_dict as i2b2_relation_dict
from infer_utils import get_data
from relation_extraction.models import model_utils
import main_utils
#import argparse
from relation_extraction.models.model import CRCNN as Model
import parser
import networkx as nx

import copy
import json

from sklearn.metrics import f1_score, accuracy_score

log = logging.getLogger()

config = parser.get_config()

config.early_stop = True
config.dataset = "i2b2"
config.border_size = -1
config.use_test = True
config.preprocessing_type = "entity_blinding"

post = '_' + config.preprocessing_type
# config.data_root = "pre-processed/" + config.preprocessing_type + "/"
config.data_root = ""

relation_dict = i2b2_relation_dict
config.classnum = max(relation_dict.keys()) + 1 # we do not have an 'other' class here
config.embedding_file = 'wikipedia-pubmed-and-PMC-w2v.txt'

# config.train_text_dataset_path = 'train{post}.txt'.format(post=post)
config.test_text_dataset_path = "actual_test.txt"
def res(path): return os.path.join(config.data_root, path)

prefix = ''
mode = 'normal'
pickled_dataset = None
date_of_experiment_start = None

def prediction(scores, dataset, classnum):
    data_size = scores.shape[0]
    pred = np.zeros(data_size)
    for idx in range(data_size):
        data_line = scores[idx]
        pred[idx] = np.argmax(data_line)

    return pred


def run_infer(session, model, batch_iter, epoch, dataset, classnum):

    step = 0 #len(all_data)
    tot_data = 0
    preds = []
    scores = []


    for batch in batch_iter:
        step += 1
        tot_data += batch.shape[0]
        batch = (x for x in zip(*batch))

        sents, relations, e1, e2, dist1, dist2, position1, position2 = batch
        
        sents = np.vstack(sents)

        in_x, in_e1, in_e2, in_dist1, in_dist2, in_y, in_epoch, in_pos1, in_pos2 = model.inputs
        feed_dict = {in_x: sents, in_e1: e1, in_e2: e2, in_dist1: dist1, in_dist2: dist2, \
                in_y: relations, in_epoch: epoch, in_pos1: position1, in_pos2: position2}
    
        
        scores, = session.run(
                [model.scores],
                feed_dict=feed_dict
        )
        pred = prediction(scores, dataset, classnum)

        preds.extend(pred)

    return preds


def init(word_dict):

    if config.log_file is None:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
    else:
        main_utils.create_folder_if_not_exists(config.save_path)

        logging.basicConfig(
            filename=config.log_file, filemode='a', level=logging.DEBUG, format='%(asctime)s %(message)s',
            datefmt='%m-%d %H:%M'
        )

    dev_data, config.test_text_dataset_file = get_data(res, pickled_dataset, config, mode=mode)
    # train_data, dev_data, early_stop_data, config.train_text_dataset_file, config.test_text_dataset_file =\
    #             main_utils.get_data(res, pickled_dataset, config, mode=mode)
    # data_size = {'train': len(train_data[0]), 'dev': len(dev_data[0]), 'early_stop':
    #         len(early_stop_data[0])}
    # data = {'train': train_data, 'dev': dev_data, 'early_stop': early_stop_data}

    # word_dict = main_utils.get_word_dict(data, config.low_freq_thresh, config.early_stop)
    # json.dump(word_dict, open("word_dict_entity_blinding.json","w"))
    
    config.max_len = 190
    config.max_e1_len = 14
    config.max_e2_len = 28
 
    dev_vec = data_utils.vectorize(config, dev_data, word_dict)

    return dev_vec
    

def main(config, ent_sent_map, session, m_eval, dev_vec, save_graph_path=None):

    bz = config.batch_size

    # with tf.Graph().as_default():

    #     with tf.name_scope("Valid"):
    #         with tf.variable_scope("Model", reuse=None):
    #             m_eval = Model(config, embeddings, is_training=False)
    #             m_eval.__run__()

    #     # Start TensorFlow session
    #     sv = tf.train.Supervisor()

    # configProto = tf.ConfigProto()
    # configProto.gpu_options.allow_growth = True

    # with sv.managed_session(config=configProto) as session:

    #     sv.saver.restore(session, "output2/")        

    dev_iter   = data_utils.batch_iter(config.seed, main_utils.stack_data(dev_vec), bz, shuffle=False)

    dev_preds = run_infer(
        session, m_eval, dev_iter, 0, config.dataset,
        config.classnum
    )

    # dev_preds = [8.0, 7.0, 7.0, 8.0, 8.0, 7.0, 8.0, 8.0, 7.0, 8.0, 8.0, 8.0, 8.0, 8.0, 7.0]

    G = nx.Graph()
    file = open(config.test_text_dataset_path,"r")
    lines = file.readlines()

    for idx, line in enumerate(lines):
        splitted = line.split()
        sentence = splitted[5:]
        relation = dev_preds[idx] if idx < len(dev_preds) else 0
        
        start1, end1 = int(splitted[1]), int(splitted[2])
        start2, end2 = int(splitted[3]), int(splitted[4]) 
        ent1 = sentence[start1] if start1 == end1 else ' '.join(sentence[start1:end1+1])
        ent2 = sentence[start2] if start2 == end2 else ' '.join(sentence[start2:end2+1])
        
        ent1 = ent1.replace(".","")
        ent2 = ent2.replace(".","")

        if ent_sent_map.get(ent1) is not None and ent_sent_map.get(ent2) is not None:
            G.add_node(ent1)
            G.add_node(ent2)
        
            if int(relation) != 8:
                G.add_edge(ent1,ent2, weight=int(relation))

    if save_graph_path is not None:    
        pos = nx.spring_layout(G, k=0.3)
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx(G, pos, with_labels=True, node_color="lightblue", node_size=150, font_size=8, edge_color="gray", edge_labels=labels)

        plt.title("Graph of Multi-word Entities")
        plt.savefig(save_graph_path)

    rank_dict = nx.pagerank(G)
    sorted_list = {k: v for k, v in sorted(rank_dict.items(), key=lambda item: item[1], reverse=True)}

    return sorted_list



if __name__ == "__main__":
    main(config) 