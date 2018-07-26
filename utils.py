import os
import re
import h5py
import argparse
import jieba
import pickle
import numpy as np
from collections import Counter

import pdb

np.seed = 88

class DataLoader(object):
    def __init__(self, c_vocab_size, w_vocab_size, num_classes, n_steps, batch_size, max_cv_epoch, data_dir='./data'):
        self.c_vocab_size = c_vocab_size
        self.w_vocab_size = w_vocab_size
        self.num_classes = num_classes
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.max_cv_epoch = max_cv_epoch

        self.raw_path = os.path.join(data_dir, 'ent-char.bieo')

        self.c_vocab_path = os.path.join(data_dir, 'c-vocab-%d.pkl'%c_vocab_size)
        self.w_vocab_path = os.path.join(data_dir, 'w-vocab-%d.pkl'%w_vocab_size)
        self.label_path = os.path.join(data_dir, 'label2id.pkl')

        self.train_file_path = os.path.join(data_dir, 'train_file.txt')
        self.test_file_path = os.path.join(data_dir, 'test_file.txt')
        self.trainset_path = os.path.join(data_dir, 'trainset.h5')
        self.testset_path = os.path.join(data_dir, 'testset.h5')

        self.pad_id = 0
        self.unk_id = 1

        self.data_ratio = [0.8, 0.2]

        self.get_vocab()
        self.label2id = {'PAD': 0, 'O': 1, 'S': 2, 'B': 3, 'I': 4, 'E': 5}

        self.id2char = {i: char for char, i in self.char2id.items()}
        self.id2word = {i: word for word, i in self.word2id.items()}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
    def get_vocab(self):
        ## char vocab
        if not os.path.exists(self.c_vocab_path):
            char_list = []
            label_list = []
            c_sen_len = []
            with open(self.raw_path) as fr:
                sentences = fr.read().strip().split('\n\n')
                cnt = 0
                for sen in sentences:
                    cnt += 1
                    if cnt % 100000 == 0:
                        print('parsed %d lines.'%cnt)

                    try:
                        chars = [x.split('\t')[0] for x in sen.split('\n')]
                        labels = [x.split('\t')[1] for x in sen.split('\n')]
                    except:
                        pdb.set_trace()

                    chars = self.get_digits_norm(chars)
                    char_list.extend(chars)
                    label_list.extend(labels)
                    c_sen_len.append(len(chars))
            char_list = Counter(char_list).most_common()
            c_sen_len = sorted(c_sen_len)
            print('total %d unique words, choose %d words, cover %.3f'%(len(char_list), self.c_vocab_size, self.c_vocab_size*1.0/len(char_list)))
            print('average char length: %d, 90%% sentences length is %d, max length is %d'%(sum(c_sen_len)*1.0/len(c_sen_len), c_sen_len[int(0.9*len(c_sen_len))], c_sen_len[-1]))
            self.char2id = {c[0]: i+2 for i, c in enumerate(char_list[:self.c_vocab_size-2])}
            self.char2id['_pad_'] = 0
            self.char2id['_unk_'] = 1

            with open(self.c_vocab_path, 'wb') as fw:
                pickle.dump(self.char2id, fw)

        else:
            self.char2id = pickle.load(open(self.c_vocab_path, "rb"))
        ## word vocab
        if not os.path.exists(self.w_vocab_path):
            word_list = []
            w_sen_len = []
            with open(self.raw_path) as fr:
                sentences = fr.read().strip().split('\n\n')
                cnt = 0
                for sen in sentences:
                    cnt += 1
                    if cnt % 100000 == 0:
                        print('parsed %d lines.'%cnt)

                    try:
                        chars = [x.split('\t')[0] for x in sen.split('\n')]
                        sentence = ''.join(chars)
                        # To add data normalization, such as numbers.
                        ws = [x for x in jieba.cut(sentence)]
                        word_list.extend(ws)
                    except:
                        pdb.set_trace()
                    chars = self.get_digits_norm(chars)
                    w_sen_len.append(len(ws))

            word_list = Counter(word_list).most_common()
            w_sen_len = sorted(w_sen_len)
            print('total %d unique words, choose %d words, cover %.3f'%(len(word_list), self.w_vocab_size, self.w_vocab_size*1.0/len(word_list)))
            print('average word length: %d, 90%% sentences length is %d, max length is %d'%(sum(w_sen_len)*1.0/len(w_sen_len), w_sen_len[int(0.9*len(w_sen_len))], w_sen_len[-1]))
            self.word2id = {w[0]: i+2 for i, w in enumerate(word_list[:self.w_vocab_size-2])}
            self.word2id['_pad_'] = 0
            self.word2id['_unk_'] = 1 

            with open(self.w_vocab_path, 'wb') as fw:
                pickle.dump(self.word2id, fw)
        else:
            self.word2id = pickle.load(open(self.w_vocab_path, "rb"))
    
    def get_digits_norm(self, chars):
        res = []
        for char in chars:
            if re.match(r'^\d+$', char):
                res.append('_num_')
            else:
                res.append(char)
        return res

    def get_dataset(self, mode):
        if not os.path.exists(self.trainset_path):
            # data to id
            out_train = open(self.train_file_path, 'w')
            out_test = open(self.test_file_path, 'w')
            train_seq_list = []
            train_label_list = []
            test_seq_list = []
            test_label_list = []
            with open(self.raw_path) as fr:
                # split train/test set
                sentences = fr.read().strip().split('\n\n')
                shuffle_idx = np.random.permutation(np.arange(len(sentences)))
                train_num = int(self.data_ratio[0] * len(sentences))
                test_num = len(sentences) - train_num

                for train_idx in shuffle_idx[:train_num]:
                    out_train.write(sentences[train_idx]+'\n\n')
                    chars = [x.split('\t')[0] for x in sentences[train_idx].split('\n')]
                    labels = [x.split('\t')[1] for x in sentences[train_idx].split('\n')]
                    # Digits norm
                    chars = self.get_digits_norm(chars)
                    seq_id = [self.char2id.get(t, self.unk_id) for t in chars]
                    label_id = [self.label2id[t] for t in labels]
                    if len(seq_id) < self.n_steps:
                        seq_id.extend([self.pad_id]*(self.n_steps-len(seq_id)))
                        label_id.extend([self.label2id['PAD']]*(self.n_steps-len(label_id))) # PAD -> 0
                    elif len(seq_id) > self.n_steps:
                        seq_id = seq_id[-self.n_steps:]
                        label_id = label_id[-self.n_steps:]

                    train_seq_list.append(seq_id)
                    train_label_list.append(label_id)
                for test_idx in shuffle_idx[train_num:]:
                    out_test.write(sentences[test_idx]+'\n\n')
                    chars = self.get_digits_norm(chars)
                    chars = [x.split('\t')[0] for x in sentences[test_idx].split('\n')]
                    labels = [x.split('\t')[1] for x in sentences[test_idx].split('\n')]
                    seq_id = [self.char2id.get(t, self.unk_id) for t in chars]
                    label_id = [self.label2id[t] for t in labels]
                    if len(seq_id) < self.n_steps:
                        seq_id.extend([self.pad_id]*(self.n_steps-len(seq_id)))
                        label_id.extend([self.label2id['PAD']]*(self.n_steps-len(label_id))) # PAD -> 0
                    elif len(seq_id) > self.n_steps:
                        seq_id = seq_id[-self.n_steps:]
                        label_id = label_id[-self.n_steps:]                    
                    test_seq_list.append(seq_id)
                    test_label_list.append(label_id)

                self.train_inputs = np.array(train_seq_list, dtype=np.int32)
                self.train_labels = np.array(train_label_list, dtype=np.int32)

                self.test_inputs = np.array(test_seq_list, dtype=np.int32)
                self.test_labels = np.array(test_label_list, dtype=np.int32)
            with h5py.File(self.trainset_path, 'w') as fw:
                fw.create_dataset('inputs', data=self.train_inputs)
                fw.create_dataset('labels', data=self.train_labels)
            with h5py.File(self.testset_path, 'w') as fw:
                fw.create_dataset('inputs', data=self.test_inputs)
                fw.create_dataset('labels', data=self.test_labels)            

            out_train.close()
            out_test.close()
        else:
            with h5py.File(self.trainset_path) as fr:
                self.train_inputs = fr['inputs'][:]
                self.train_labels = fr['labels'][:]
            with h5py.File(self.testset_path) as fr:
                self.test_inputs = fr['inputs'][:]
                self.test_labels = fr['labels'][:]
        self.train_batch_num = len(self.train_inputs) / self.batch_size
        self.test_batch_num = len(self.test_inputs) / self.batch_size

    def get_cross_validation_dataset(self, cv_epoch):
        self.get_dataset('train')
        self.get_dataset('test')

    def get_pre_embedding(self):
        pass
    
    def get_transition(self, y_train_batch):
        transition_batch = []
        for m in range(len(y_train_batch)):
                y = [self.num_classes] + list(y_train_batch[m]) + [0]
                for t in range(len(y)):
                    if t + 1 == len(y):
                        continue
                    i = y[t]
                    j = y[t + 1]
                    if i == 0:
                        break
                    transition_batch.append(i * (self.num_classes+1) + j)
        transition_batch = np.array(transition_batch)
        return transition_batch

    def next_batch(self, mode='train'):
        if mode == 'train':
            inputs = self.train_inputs[self.train_ptr*self.batch_size: (self.train_ptr+1)*self.batch_size]
            labels = self.train_labels[self.train_ptr*self.batch_size: (self.train_ptr+1)*self.batch_size]
            self.train_ptr += 1
        else:
            inputs = self.test_inputs[self.test_ptr*self.batch_size: (self.test_ptr+1)*self.batch_size]
            labels = self.test_labels[self.test_ptr*self.batch_size: (self.test_ptr+1)*self.batch_size]
            self.test_ptr = (self.test_ptr+1) % self.test_batch_num
        return inputs, labels

    def shuffle(self, mode='train'):
        if mode == 'train':
            self.train_ptr = 0
            train_shuffle = np.random.permutation(np.arange(len(self.train_inputs)))
            self.train_inputs = self.train_inputs[train_shuffle]
            self.train_labels = self.train_labels[train_shuffle]
        else:
            self.test_ptr = 0
            test_shuffle = np.random.permutation(np.arange(len(self.test_inputs)))
            self.test_inputs = self.train_inputs[test_shuffle]
            self.test_labels = self.train_labels[test_shuffle]
        print('shuffle %s-set'%mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument('-m', '--mode', type=str, default="train", help='run mode, train/test')
    parser.add_argument('-o', '--order', type=str, default="train", help='train order')
    parser.add_argument('--res_iter', type=int, default=11, help='restore model iteration ')
    parser.add_argument('--embed_mode', type=str, default='char', help='sequence embed mode, char/word/hybrid')
    parser.add_argument('--cv_epoch', type=int, default=0, help='cross validation epoch, 0 ~ max_epoch-1')
    parser.add_argument('--code', type=str, default='gbk', help='code to encode/decode')
    # model args
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--op', type=str, default='adam', help='optimizer type, adam/sgd')
    parser.add_argument('--c_vocab_size', type=int, default=3000, help='char vocab size')
    parser.add_argument('--w_vocab_size', type=int, default=9000, help='word vocab size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--emb_dim', type=int, default=200, help='embbedding dimension for char/word')
    parser.add_argument('--hidden_dim', type=int, default=400, help='hidden dimension for bi-rnn')
    parser.add_argument('--n_steps', type=int, default=16, help='max sequence length')
    parser.add_argument('--max_cv_epoch', type=int, default=5, help='max cross validation epoch')
    parser.add_argument('--n_epoches', type=int, default=15, help='max train epoches')
    parser.add_argument('--n_classes', type=int, default=6, help='number of label classes.')  # [PAD, B, I, E, O, S]
    parser.add_argument('--rnn_type', type=str, default='lstm', help='rnn type, lstm/gru')
    parser.add_argument('--layers', type=int, default=1, help='rnn layers for bi-rnn')
    parser.add_argument('--dropout', type=float, default=1.0, help='dropout keep rate.')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping norm')
    parser.add_argument('--pre_embed', type=int, default=0, help='whether to use pre-trained embedding.')

    # data args
    parser.add_argument('--data_dir', type=str, default='./data', help='data dir')
    parser.add_argument('--save_dir', type=str, default='./save', help='save dir')
    parser.add_argument('--train_print_iters', type=int, default=100, help='training print iterations')
    parser.add_argument('--test_print_iters', type=int, default=100, help='test print iterations')
    # parser.add_argument('--dev_print_iters', type=int, default=500, help='training print iterations')

    args = parser.parse_args()
    data_loader = DataLoader(args.c_vocab_size, args.w_vocab_size, args.n_classes, args.n_steps, args.batch_size,
                             args.max_cv_epoch)
    data_loader.get_dataset("train")
    data_loader.shuffle()

    m = data_loader.next_batch("train")
    print("done")
