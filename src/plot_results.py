# /data/home/aml7/co-attention/old_code/plot_results.py

import io
import json
import logging
import os
import sys
import time

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn

from data_util.data_batcher import get_batch_generator
from data_util.evaluate import exact_match_score, f1_score
# from data_util.official_eval_helper import get_json_data, generate_answers
from data_util.pretty_print import print_example
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from data_util.vocab import get_glove

from config import config
from model import CoattentionModel
from model_baseline import Baseline

from itertools import groupby
import matplotlib.pyplot as plt
import pickle


logging.basicConfig(level=logging.INFO)

use_cuda = torch.cuda.is_available()
print ('Using CUDA.' if use_cuda else 'Using CPU.')

def adaptive_load_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print('Skipped {}, not found'.format(name))
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if own_state[name].shape != param.shape:
            print('Skipped {}, shape mismatch'.format(name))
            continue
        own_state[name].copy_(param)

class Processor(object):
    def __init__(self):
        self.glove_path = os.path.join(config.data_dir, "glove.6B.{}d.txt".format(config.embedding_size))
        self.emb_matrix, self.word2id, self.id2word = get_glove(self.glove_path, config.embedding_size)

        self.train_context_path = os.path.join(config.data_dir, "train.context")
        self.train_qn_path = os.path.join(config.data_dir, "train.question")
        self.train_ans_path = os.path.join(config.data_dir, "train.span")
        self.dev_context_path = os.path.join(config.data_dir, "dev.context")
        self.dev_qn_path = os.path.join(config.data_dir, "dev.question")
        self.dev_ans_path = os.path.join(config.data_dir, "dev.span")

    def get_mask_from_seq_len(self, seq_mask):
        seq_lens = np.sum(seq_mask, 1)
        max_len = np.max(seq_lens)
        indices = np.arange(0, max_len)
        mask = (indices < np.expand_dims(seq_lens, 1)).astype(int)
        return mask

    def get_data(self, batch, is_train=True):
        qn_mask = self.get_mask_from_seq_len(batch.qn_mask)
        qn_mask_var = torch.from_numpy(qn_mask).long()

        context_mask = self.get_mask_from_seq_len(batch.context_mask)
        context_mask_var = torch.from_numpy(context_mask).long()

        qn_seq_var = torch.from_numpy(batch.qn_ids).long()
        context_seq_var = torch.from_numpy(batch.context_ids).long()

        if is_train:
            span_var = torch.from_numpy(batch.ans_span).long()

        if use_cuda:
            qn_mask_var = qn_mask_var.cuda()
            context_mask_var = context_mask_var.cuda()
            qn_seq_var = qn_seq_var.cuda()
            context_seq_var = context_seq_var.cuda()
            if is_train:
                span_var = span_var.cuda()

        if is_train:
            return qn_seq_var, qn_mask_var, context_seq_var, context_mask_var, span_var
        else:
            return qn_seq_var, qn_mask_var, context_seq_var, context_mask_var

    def get_model(self, model_file_path=None, is_eval=False):
        if config.model_type == 'co-attention':
            model = CoattentionModel(config.hidden_dim, config.maxout_pool_size,
                                 self.emb_matrix, config.max_dec_steps, config.dropout_ratio)
        else:
            model = Baseline(config.hidden_dim, self.emb_matrix, config.dropout_ratio)

        if is_eval:
            model = model.eval()
        if use_cuda:
            model = model.cuda()

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            adaptive_load_state_dict(model, state['model'])

        return model

    def test_one_batch(self, batch, model):
        model.eval()
        q_seq, q_lens, d_seq, d_lens = self.get_data(batch, is_train=False)
        pred_start_pos, pred_end_pos = model(q_seq, q_lens, d_seq, d_lens)
        return pred_start_pos.data, pred_end_pos.data

    def get_individual_f1(self, model, dataset, num_samples=0):
        logging.info("Calculating F1 for all individual samples...")

        if dataset == "train":
            context_path, qn_path, ans_path = self.train_context_path, self.train_qn_path, self.train_ans_path
        elif dataset == "dev":
            context_path, qn_path, ans_path = self.dev_context_path, self.dev_qn_path, self.dev_ans_path
        else:
            raise ('dataset is not defined')
        
        all_f1s = []

        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path, config.batch_size,
                                         context_len=config.context_len, question_len=config.question_len,
                                         discard_long=False):

            pred_start_pos, pred_end_pos = self.test_one_batch(batch, model)

            pred_start_pos = pred_start_pos.tolist()
            pred_end_pos = pred_end_pos.tolist()

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) \
                    in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                true_answer = " ".join(true_ans_tokens)

                f1 = f1_score(pred_answer, true_answer)
                all_f1s.append((len(batch.context_tokens[ex_idx]), len(batch.qn_tokens[ex_idx]), len(true_ans_tokens), batch.qn_tokens[ex_idx][0], f1))

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        return all_f1s


if __name__ == "__main__":
    if not os.path.exists('all_f1s.pkl'):
        processor = Processor()
        model_file_path = '/home/aml7/co-attention/log/train_1555145528/bestmodel/model_12000_14_1555186366'
        model = processor.get_model(model_file_path)
        all_f1s = processor.get_individual_f1(model, 'dev', 0)
        pickle.dump(all_f1s, open('all_f1s.pkl', 'wb'))
        print ('Dumped all F1 scores.')
    else:
        all_f1s = pickle.load(open('all_f1s.pkl', 'rb'))

    """ # tokens in document """
    keys, means, stds = [], [], []
    for key, group in groupby(sorted(all_f1s, key=lambda x: x[0]), lambda x: x[0]):
        f1s = np.array([np.clip(t[-1], 0, 1) for t in group])
        keys.append(key)
        means.append(np.mean(f1s))
        stds.append(np.std(f1s))
    keys, means, stds = np.array(keys), np.array(means), np.array(stds)
    plt.errorbar(keys, means, stds, linestyle='None', marker='o', capsize=3)
    plt.xlabel('#Tokens in Document')
    plt.ylabel('F1')
    # plt.show()
    plt.savefig('perf_across_doclength.png', dpi=150)

    """ # tokens in question """
    keys, means, stds = [], [], []
    for key, group in groupby(sorted(all_f1s, key=lambda x: x[1]), lambda x: x[1]):
        f1s = np.array([np.clip(t[-1], 0, 1) for t in group])
        keys.append(key)
        means.append(np.mean(f1s))
        stds.append(np.std(f1s))
    keys = np.array(keys)
    plt.errorbar(keys, means, stds, fmt='-o')
    plt.xlabel('#Tokens in Question')
    plt.ylabel('F1')
    # plt.show()
    plt.savefig('perf_across_qnlength.png', dpi=150)
    
    """ # tokens in answer """
    keys, means, stds = [], [], []
    for key, group in groupby(sorted(all_f1s, key=lambda x: x[2]), lambda x: x[2]):
        f1s = np.array([np.clip(t[-1], 0, 1) for t in group])
        keys.append(key)
        means.append(np.mean(f1s))
        stds.append(np.std(f1s))
    keys = np.array(keys)
    plt.errorbar(keys, means, stds, fmt='-o')
    plt.xlabel('#Tokens in Answer')
    plt.ylabel('F1')
    # plt.show()
    plt.savefig('perf_across_anslength.png', dpi=150)

    # TODO: (yuanhang) move all other categories to "other"
    # """ question type """
    # keys, means, stds = [], [], []
    # for key, group in groupby(sorted(all_f1s, key=lambda x: x[3]), lambda x: x[3]):
    #     f1s = np.array([np.clip(t[-1], 0, 1) for t in group])
    #     keys.append(key)
    #     means.append(np.mean(f1s))
    #     stds.append(np.std(f1s))
    #     print (key, len(list(group)))
    # keys = np.array(keys)
    # plt.errorbar(keys, means, stds, fmt='-o')
    # # plt.show()
    # plt.savefig('perf_across_qtype.png', dpi=150)
    
