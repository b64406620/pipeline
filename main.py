# coding:utf-8
import os 
import sys
import argparse
import numpy as np  

import tensorflow as tf  

from utils import DataLoader
from models.model import BiRNNCRF

import pdb

parser = argparse.ArgumentParser()

# main args
parser.add_argument('-m', '--mode', type=str, required=True, help='run mode, train/test')
parser.add_argument('-o', '--order', type=str, required=True, help='train order')
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
parser.add_argument('--n_classes', type=int, default=6, help='number of label classes.') # [PAD, B, I, E, O, S]
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
#parser.add_argument('--dev_print_iters', type=int, default=500, help='training print iterations')

args = parser.parse_args()

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)

if args.mode == 'train':
    data_loader = DataLoader(args.c_vocab_size, args.w_vocab_size, args.n_classes, args.n_steps, args.batch_size, args.max_cv_epoch)
    data_loader.get_cross_validation_dataset(args.cv_epoch)
    
    label2id, id2label, id2char = data_loader.label2id, data_loader.id2label, data_loader.id2char

    pre_embed = None
    if args.pre_embed:
        pre_embed = data_loader.get_pre_embedding()

    train_batch_num = data_loader.train_batch_num
    test_batch_num = data_loader.test_batch_num

    model = BiRNNCRF(args, pre_embed)
    saver = tf.train.Saver(max_to_keep=30)

    summary_path = os.path.join(args.save_dir, 'summary', args.order, 'train')
    if not os.path.exists(summary_path):
        os.system('mkdir -p %s'%summary_path)

    train_writer = tf.summary.FileWriter(summary_path, sess.graph)  
    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()

    cnt = -1
    print 'train batch num: %d, test batch num: %d'%(data_loader.train_batch_num, data_loader.test_batch_num)
    for epoch in range(args.n_epoches):
        data_loader.shuffle('train')
        data_loader.shuffle('test')
        for _ in range(train_batch_num):
            cnt += 1
            x_batch, y_batch = data_loader.next_batch('train')
            if not len(y_batch):
                pdb.set_trace()            
            #y_weight_batch = 1 + np.array((y_batch == label2id['B']) | (y_batch == label2id['E']), float) # Not used in the model!
            #fd = {model.targets_transition:transition_batch, model.inputs:x_batch, model.targets:y_batch, model.targets_weight:y_weight_batch}
            transition_batch = data_loader.get_transition(y_batch)
            fd = {model.targets_transition:transition_batch, model.inputs:x_batch, model.targets:y_batch}
            _, train_loss, max_scores, max_scores_pre, length, train_summary = sess.run([model.train_op, model.loss, model.max_scores, 
                                                                            model.max_scores_pre, model.length, model.summary], 
                                                                            fd)

            predicts_train = model.viterbi(max_scores, max_scores_pre, length, predict_size=args.batch_size)

            if cnt % args.train_print_iters == 0:
                precision_train, recall_train, f1_train, _ = model.evaluate(x_batch, y_batch, predicts_train, id2char, id2label, length=length)
                train_writer.add_summary(train_summary, cnt)
                print "  train: %d, iteration: %5d, train loss: %5d, train precision: %.5f, train recall: %.5f, train f1: %.5f" % (epoch, cnt, train_loss, precision_train, recall_train, f1_train)  
        
        ### after one epoch, begin test
        hit_num = []
        pred_num = []
        true_num = []
        test_loss = []
        for _ in range(test_batch_num):
            x_batch, y_batch = data_loader.next_batch('test')
            if not len(y_batch):
                pdb.set_trace()
            transition_batch = data_loader.get_transition(y_batch)
            fd = {model.targets_transition:transition_batch, model.inputs:x_batch, model.targets:y_batch}
            
            loss, max_scores, max_scores_pre, length = sess.run([model.loss, model.max_scores, model.max_scores_pre, model.length], fd)
            predicts_test = model.viterbi(max_scores, max_scores_pre, length, predict_size=args.batch_size)
            _, _, _, test_tuple = model.evaluate(x_batch, y_batch, predicts_test, id2char, id2label)
            # test_tuple: (hit, pred_num, true_num)
            hit_num.append(test_tuple[0])
            pred_num.append(test_tuple[1])
            true_num.append(test_tuple[2])
            test_loss.append(loss)
        precision = sum(hit_num) * 1.0 / sum(pred_num)
        recall = sum(hit_num) * 1.0 / sum(true_num)
        f1 = 2 * precision * recall / (precision + recall)
        print "epoch: %d, test loss: %.4f, precision: %.4f, recall: %.4f, f1: %.4f" % (epoch, np.mean(test_loss), precision, recall, f1)        
        save_path = os.path.join(args.save_dir, 'ckpt', args.order, 'model-%d'%epoch)
        if not os.path.exists(os.path.dirname(save_path)):
            os.system('mkdir -p %s'%(os.path.dirname(save_path)))
        saver.save(sess, save_path)
        
        # validation

elif args.mode == 'debug':
    data_loader = DataLoader(args.c_vocab_size, args.w_vocab_size, args.n_classes, args.n_steps, args.batch_size, args.max_cv_epoch)
    label2id, id2label, id2char, char2id = data_loader.label2id, data_loader.id2label, data_loader.id2char, data_loader.char2id

    model = BiRNNCRF(args)
    saver = tf.train.Saver(max_to_keep=30)
    res_path = os.path.join(args.save_dir, 'ckpt', args.order, 'model-%d'%args.res_iter)    
    saver.restore(sess, res_path)
    
    sentence = raw_input('Please input the sentence here:')
    while sentence:
	sent_utf = sentence.decode(args.code)
        sent_id = [char2id.get(char.encode(args.code), data_loader.unk_id) for char in sent_utf]
        if len(sent_id) > args.n_steps:
            sent_id = sent_id[-args.n_steps]
        else:
            sent_id += [data_loader.pad_id] * (args.n_steps - len(sent_id))
        input_sens = [sent_id] * args.batch_size
        fd = {model.inputs:input_sens}
        max_scores, max_scores_pre, length = sess.run([model.max_scores, model.max_scores_pre, model.length], fd)
        predicts_labels = model.viterbi(max_scores, max_scores_pre, length, predict_size=args.batch_size)
        label_str = [id2label[i] for i in predicts_labels[0]]
        print label_str
	sentence = raw_input("Please input the next sentence here:")
else:
    pass

