import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random as r

from collections import Counter

from torchtext import datasets
from torchtext import data
from torchtext import vocab

import json

import sys

FOLDER_PATH = "./"
DEBUG = True


def DEBUG_PRINT(x):
    if DEBUG:
        print(x)

deviceCuda = torch.device("cuda")
deviceCPU = torch.device("cpu")
USE_CUDA = True 
USE_840 = True

GLOVE_VECTORS = vocab.GloVe(name='840B', dim=300, cache=FOLDER_PATH+'glove_cache')

if USE_CUDA:
    USED_DEVICE = deviceCuda

def list2dict(lst):
    it = iter(lst)
    indexes = range(len(lst))
    res_dct = dict(zip(it, indexes))
    return res_dct


class DB(object):
    def __init__(self, batch_size):
        self.data_field = data.Field(init_token='NULL', tokenize='spacy', batch_first=True, include_lengths=True)
        self.label_field = data.Field(sequential=False, batch_first=True)
      
        self.label_field.build_vocab([['contradiction'], ['entailment'], ['neutral']])

        self.train_ds, self.dev_ds, self.test_ds = datasets.SNLI.splits(self.data_field, self.label_field, root=FOLDER_PATH)

        self.data_field.build_vocab(self.train_ds, self.dev_ds, self.test_ds, vectors=GLOVE_VECTORS, unk_init=torch.Tensor.normal_)
        
        from collections import Counter

        fake_keys = Counter(list(self.data_field.vocab.stoi.keys()))
        self.glove_keys = [[key] for key in GLOVE_VECTORS.stoi.keys() if fake_keys[key] > 0]
        self.data_field.build_vocab(self.glove_keys, vectors=GLOVE_VECTORS)
        fake_keys = []

        self.train_iter, self.dev_iter, self.test_iter = data.BucketIterator.splits((self.train_ds, self.dev_ds, self.test_ds), 
                batch_size=batch_size, device=deviceCPU, sort_key=lambda d: len(d.premise), shuffle=True, sort=True)

    def getIter(self, iter_type):      
        if iter_type == "train":
            return self.train_iter
        elif iter_type == "dev":
            return self.dev_iter
        elif iter_type == "test":
            return self.test_iter
        else:
            raise Exception("Invalid type")

class Tagger(nn.Module):
    def __init__(self, embedding_dim, projected_dim, tagset_size,
                 vectors, f_dim=200, v_dim=200, dropout=False):
        super(Tagger, self).__init__()
        self.embedding_dim = embedding_dim

        # Creat Embeddings
        vecs =vectors 
        vecs = vecs/torch.norm(vecs, dim=1, keepdim=True)
        self.unknown_idx = vectors.shape[0]
        print("Unknown Index = " + str(self.unknown_idx))
        ## Add to glove vectors 2 vectors for unknown and padding:
        for i in range(100):
            pad = torch.randn((1, vecs[0].shape[0]))
            #pad = torch.normal(mean=torch.zeros(1, vecs[0].shape[0]), std=1)
            vecs = torch.cat((vecs, pad), 0)
        pad = torch.zeros((1, vecs[0].shape[0]))
        vecs = torch.cat((vecs, pad), 0)
        vecs[0] = torch.zeros(vecs[0].shape)
        vecs[1] = torch.zeros(vecs[0].shape)
        self.wembeddings = nn.Embedding.from_pretrained(embeddings=vecs, freeze=True)
        ## project down the vectors to 200dim
        self.project = nn.Linear(embedding_dim, projected_dim)
        self.G = self.feedForward(f_dim * 2, v_dim, 0.2)
        self.H = self.feedForward(v_dim * 2, v_dim, 0.2)
        self.linear = nn.Linear(v_dim, tagset_size)
        self.hidden_dim = projected_dim
        self.f_dim = f_dim
        
        self.F = self.feedForward(self.hidden_dim, f_dim, 0.2)
        self.softmax = nn.Softmax(dim=1)

    def feedForward(self, i_dim, o_dim, dropout):
        use_dropout = dropout > 0
        layers = []

        layers.append(nn.Linear(i_dim, o_dim))
        if use_dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(o_dim, o_dim))
        if use_dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

        layers = nn.Sequential(*layers)
        return layers

    def forward(self, premise_data, hyp_data):
        prem_rand_idx = torch.randint(self.unknown_idx, self.unknown_idx+99,
                                      premise_data.shape)
        hyp_rand_idx = torch.randint(self.unknown_idx, self.unknown_idx+99,
                                     hyp_data.shape)

        premise_data[premise_data == 0] = prem_rand_idx[premise_data == 0]
        hyp_data[hyp_data == 0] = hyp_rand_idx[hyp_data == 0]

        premise_mask = torch.ones(premise_data.shape)
        hyp_mask = torch.ones(hyp_data.shape)
        premise_mask[premise_data == 1] = 0
        hyp_mask[hyp_data == 1] = 0

        tmp = list(premise_mask.shape)
        tmp.append(1)
        premise_mask = premise_mask.reshape(tmp).repeat(1,1,200)

        tmp = list(hyp_mask.shape)
        tmp.append(1)
        hyp_mask = hyp_mask.reshape(tmp).repeat(1,1,200)


        if USE_CUDA:
            padded_premise_w = premise_data.to(deviceCuda)
            padded_hyp_w = hyp_data.to(deviceCuda)
            premise_mask = premise_mask.to(deviceCuda)
            hyp_mask = hyp_mask.to(deviceCuda)
        else:
            padded_premise_w = premise_data
            padded_hyp_w = hyp_data


        prem_w_e = self.wembeddings(padded_premise_w)
        hyp_w_e = self.wembeddings(padded_hyp_w)

        # Project the embeddings to smaller vector
        prem_w_e = self.project(prem_w_e)
        hyp_w_e = self.project(hyp_w_e)

        #beta, alpha = self.attention(prem_w_e, hyp_w_e)
        a = prem_w_e 
        b = hyp_w_e
        fa = self.F(a)
        fb = self.F(b)

        # We want ato calculate e_ij = fa_i * fb_j
        # fa shape: batch x sentence_a x hidden_dim
        # fb shape: batch x sentence_b x hidden_dim
        ## Per batch calculation:
        ## calc fa x fb.transpose() gives sentence_a x sentence_b
        E = torch.bmm(fa, torch.transpose(fb, 1, 2))

        # E shape: batch x sentence_a x sentence_b
        ## for beta needs: (batch*sentence_a)*sentence_b
        E4beta = self.softmax(E.view(-1, b.shape[1]))
        E4beta = E4beta.view(E.shape)
        beta = torch.bmm(E4beta, b)

        E4alpha = torch.transpose(E, 1, 2)
        saved_shape = E4alpha.shape
        E4alpha = self.softmax(E4alpha.reshape(-1, a.shape[1]))
        # alpha is (batch*sentence_b) x sentence a
        E4alpha = E4alpha.view(saved_shape)
        alpha = torch.bmm(E4alpha, a)

        # Compare
        ##Concat to each it's weights
        weighted_a = torch.cat((prem_w_e, beta), 2)
        weighted_b = torch.cat((hyp_w_e, alpha), 2)

        ##Feedforward
        v1 = self.G(weighted_a)*premise_mask
        v2 = self.G(weighted_b)*hyp_mask

        # Aggregate
        v1 = torch.sum(v1, 1)
        v2 = torch.sum(v2, 1)

        h_in = torch.cat((v1, v2), 1)
        y = self.H(h_in)
        y = self.linear(y)

        if USE_CUDA:
          y = y.to(deviceCPU)

        return y

    def getLabel(self, data):
        _, prediction_argmax = torch.max(data, 1)
        return prediction_argmax

class Run(object):
    def __init__(self, params):
        self.edim = params['EMBEDDING_DIM']
        self.rnn_h_dim = params['RNN_H_DIM']
        self.num_epochs = params['EPOCHS']
        self.batch_size = params['BATCH_SIZE']
        self.test_file = params['TEST_FILE']
        self.test_o_file = params['TEST_O_FILE']
        self.model_file = params['MODEL_FILE']
        self.save_to_file = params['SAVE_TO_FILE']
        self.run_dev = params['RUN_DEV']
        self.learning_rate = params['LEARNING_RATE']
        self.dropout = params['DROPOUT']
        self.acc_data_list = []
        self.acc_test_list = []
        self.load_params = params['LOAD_PARAMS']

    def _load_epoch(self):
        params = torch.load(self.model_file + "_epoch")
        return params['epoch']

    def _saveAccData(self, epoch):
        try:
            acc_data = torch.load(FOLDER_PATH + 'accuracy_graphs_data')
        except FileNotFoundError:
            print("No accuracy data file found - creating new")
            acc_data = {}

        acc_data.update({str(epoch): (self.acc_data_list[-1], self.train_accuracy,
                                      self.train_loss, self.acc_test_list[-1])})
        acc_data.update({'epoch':epoch})
        torch.save(acc_data, FOLDER_PATH + 'accuracy_graphs_data')
        params = {'epoch':epoch}
        torch.save(params, self.model_file + "_epoch")

    def _calc_batch_acc(self, tagger, flatten_tag, flatten_label):
        predicted_tags = tagger.getLabel(flatten_tag)
        diff = predicted_tags - flatten_label
        correct_cntr = len(diff[diff == 0])  # tmp
        total_cntr = len(predicted_tags)  # - to_ignore
        return correct_cntr, total_cntr

    def _flat_vecs(self, batch_tag_score, batch_label_list):
        # print(batch_tag_score.shape)
        # print(batch_label_list)
        flatten_tag = batch_tag_score  # .reshape(-1, batch_tag_score.shape[2])
        flatten_label = torch.LongTensor(batch_label_list)  # .reshape(-1))
        # print(flatten_tag)
        # print(flatten_label)
        return flatten_tag, flatten_label

    def runOnDev(self, tagger, data_iter, acc_list, d_type):
        tagger.eval()

        with torch.no_grad():
            correct_cntr = 0
            total_cntr = 0
            data_iter.init_epoch()
            for sample in data_iter:
          
                premise_data, _ = sample.premise
                hyp_data, _ = sample.hypothesis
                batch_label = (sample.label - torch.ones(sample.label.shape)).long()

                batch_tag_score = tagger.forward(premise_data, hyp_data)
                # flatten_tag, flatten_label = self._flat_vecs(batch_tag_score, batch_label_list)

                # calc accuracy
                batch_label_tensor = torch.LongTensor(batch_label)
                c, t = self._calc_batch_acc(tagger, batch_tag_score, batch_label_tensor)
                correct_cntr += c
                total_cntr += t

        acc = correct_cntr / total_cntr
        acc_list.append(acc)
        print(d_type + " accuracy " + str(acc))
        
        tagger.train()

    def train(self):

        print("Loading data")
        db = DB(self.batch_size);
        train_iter = db.getIter("train")
        print("Done loading data")

        if self.load_params:
          epoch_base = int(self._load_epoch())
        else:
          epoch_base = 0

        print("init tagger")
        tagger = Tagger(embedding_dim=self.edim, projected_dim=self.rnn_h_dim,
                        tagset_size=3, vectors = db.data_field.vocab.vectors, 
                        dropout=self.dropout)

        print("done")

        if USE_CUDA:
          tagger.to(deviceCuda)

        print("define loss and optimizer")
        loss_function = nn.CrossEntropyLoss()  # ignore_index=len(lTran.tag_dict))
        optimizer = torch.optim.Adagrad(tagger.parameters(), lr=self.learning_rate,
                                        initial_accumulator_value=0.1, weight_decay=0.000002)  # 0.01)
        print("done")

        if self.run_dev:
            self.runOnDev(tagger, db.getIter('dev'), 
                          self.acc_data_list, "Validation")
            self.runOnDev(tagger, db.getIter('test'), 
                          self.acc_test_list, "Testset")
        for epoch in range(self.num_epochs):
            train_iter.init_epoch()
            loss_acc = 0
            progress1 = 0
            progress2 = 0
            correct_cntr = 0
            total_cntr = 0
            sentences_seen = 0
            for sample in train_iter:
                if progress1 / 100000 > progress2:
                    print("reached " + str(progress2 * 100000))
                    progress2 += 1
                progress1 += self.batch_size
                sentences_seen += self.batch_size

                tagger.zero_grad()

                premise_data, _ = sample.premise
                hyp_data, _ = sample.hypothesis
                batch_label = (sample.label - torch.ones(sample.label.shape)).long()

                batch_tag_score = tagger.forward(premise_data, hyp_data)
                # flatten_tag, flatten_label = self._flat_vecs(batch_tag_score, batch_label_list)

                # calc accuracy
                batch_label_tensor = torch.LongTensor(batch_label)
                c, t = self._calc_batch_acc(tagger, batch_tag_score, batch_label_tensor)
                correct_cntr += c
                total_cntr += t

                # loss = loss_function(flatten_tag, flatten_label)
                loss = loss_function(batch_tag_score, batch_label_tensor)
                loss_acc += loss.item()
                loss.backward()
                optimizer.step()
                
                tagger.zero_grad()

            if self.run_dev:
                self.runOnDev(tagger, db.getIter('dev'), 
                              self.acc_data_list, "Validation")

            print("epoch: " + str(epoch) + " " + str(loss_acc))
            self.train_accuracy = correct_cntr/total_cntr
            self.train_loss = loss_acc
            print("Train accuracy " + str(correct_cntr/total_cntr))

            if self.run_dev:
                self._saveAccData(epoch_base + epoch)
            if self.run_dev and (self.acc_data_list[-1] > 0.868):
                break;

        self.runOnDev(tagger, db.getIter('test'), 
                      self.acc_test_list, "Testset")


FAVORITE_RUN_PARAMS = {
    'EMBEDDING_DIM': 300,
    'RNN_H_DIM': 200,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 0.05
}

if __name__ == "__main__":
    model_file = FOLDER_PATH + '32Bbatch' #sys.argv[2]
    epochs = 200 
    run_dev = True 

    RUN_PARAMS = FAVORITE_RUN_PARAMS
    RUN_PARAMS.update({
                'TEST_FILE': None, 
                'TEST_O_FILE': None,
                'MODEL_FILE': model_file,
                'SAVE_TO_FILE': False,
                'RUN_DEV' : run_dev,
                'EPOCHS' : epochs,
                'LOAD_PARAMS': False,
                'DROPOUT' : True})

    run = Run(RUN_PARAMS)

    run.train()
