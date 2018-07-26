# coding:utf-8
import re
import tensorflow as tf 
import numpy as np
import pdb

class BiRNNCRF(object):
    def __init__(self, args, pre_embed=None, is_training=True, is_crf=True):
        # Parameter
        self.max_f1 = 0
        self.learning_rate = args.lr
        self.dropout_rate = args.dropout
        self.batch_size = args.batch_size
        self.num_layers = args.layers  
        self.emb_dim = args.emb_dim
        self.hidden_dim = args.hidden_dim
        #self.num_epochs = args.num_epochs
        self.num_steps = args.n_steps
        self.num_chars = args.c_vocab_size
        self.num_words = args.w_vocab_size
        self.num_classes = args.n_classes
        self.code = "gbk"
        # placeholder of x, y and weight
        self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets_weight = tf.placeholder(tf.float32, [None, self.num_steps])
        self.targets_transition = tf.placeholder(tf.int32, [None])
        
        if args.embed_mode in ['word', 'hybrid']:
            if pre_embed is not None:
                self.embedding = tf.Variable(pre_embed, trainable=True, name="emb", dtype=tf.float32)
            else:
                self.embedding = tf.get_variable("emb", [self.num_words, self.emb_dim])     
        
        if args.embed_mode in ['char', 'hybrid']:
            self.c_embedding = tf.get_variable('char_embed', [self.num_chars, self.emb_dim], dtype=tf.float32)

        # get the length of each sample
        seq_len = tf.reduce_sum(tf.sign(self.inputs), axis=1)
        self.length = tf.cast(seq_len, tf.int32)  

        inputs_c_emb = tf.nn.embedding_lookup(self.c_embedding, self.inputs)

        self.inputs_emb = inputs_c_emb

        outputs = self.get_bi_rnn_embed(self.inputs_emb, seq_len, rnn_type=args.rnn_type, layers=args.layers, dropout_keep_rate=args.dropout)
        
        # softmax
        self.outputs = tf.reshape(tf.concat(outputs, axis=1), [-1, self.hidden_dim * 2])
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes])
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        self.logits = tf.nn.xw_plus_b(self.outputs, self.softmax_w, self.softmax_b)

        if not is_crf:
            pass
        else:
            self.tags_scores = tf.reshape(self.logits, [self.batch_size, self.num_steps, self.num_classes])
            self.transitions = tf.get_variable("transitions", [self.num_classes + 1, self.num_classes + 1])
            
            dummy_val = -1000
            class_pad = tf.Variable(dummy_val * np.ones((self.batch_size, self.num_steps, 1)), dtype=tf.float32)
            self.observations = tf.concat([self.tags_scores, class_pad], axis=2)

            begin_vec = tf.Variable(np.array([[dummy_val] * self.num_classes + [0] for _ in range(self.batch_size)]), trainable=False, dtype=tf.float32)
            end_vec = tf.Variable(np.array([[0] + [dummy_val] * self.num_classes for _ in range(self.batch_size)]), trainable=False, dtype=tf.float32) 
            begin_vec = tf.reshape(begin_vec, [self.batch_size, 1, self.num_classes + 1])
            end_vec = tf.reshape(end_vec, [self.batch_size, 1, self.num_classes + 1])

            self.observations = tf.concat([begin_vec, self.observations, end_vec], axis=1)

            self.mask = tf.cast(tf.reshape(tf.sign(self.targets),[self.batch_size * self.num_steps]), tf.float32)
            
            # point score
            self.point_score = tf.gather(tf.reshape(self.tags_scores, [-1]), tf.range(0, self.batch_size * self.num_steps) * self.num_classes + tf.reshape(self.targets,[self.batch_size * self.num_steps]))
            self.point_score *= self.mask
            
            # transition score
            self.trans_score = tf.gather(tf.reshape(self.transitions, [-1]), self.targets_transition)
            
            # real score
            self.target_path_score = tf.reduce_sum(self.point_score) + tf.reduce_sum(self.trans_score)
            
            # all path score
            self.total_path_score, self.max_scores, self.max_scores_pre  = self.forward(self.observations, self.transitions, self.length)
            
            # loss
            self.loss = - (self.target_path_score - self.total_path_score)
        
        if is_training:
            if args.op == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)#.minimize(self.loss)
            elif args.op == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)#.minimize(self.loss)
            else:
                raise ValueError('unknown optimizer: %s'%args.op)

            gradients, variables = zip(*optimizer.compute_gradients(self.loss)) ## why *?
            gradients, _ = tf.clip_by_global_norm(gradients, args.grad_clip)
            self.train_op = optimizer.apply_gradients(zip(gradients, variables))

            tf.summary.scalar("loss", self.loss)
            self.summary = tf.summary.merge_all()

    def get_bi_rnn_embed(self, input_embed, seq_len, rnn_type='lstm', layers=1, dropout_keep_rate=1.0):
        if rnn_type == 'lstm':
            cell_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
        elif rnn_type == 'gru':
            cell_fw = tf.contrib.rnn.GRUCell(self.hidden_dim)
            cell_bw = tf.contrib.rnn.GRUCell(self.hidden_dim)
        else:
            raise ValueError("unknown rnn type %s!"%rnn_type)           

        if dropout_keep_rate < 1.0:
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout_keep_rate)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout_keep_rate)

        if layers > 1:
            cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw] for _ in range(self.num_layers))
            cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw] for _ in range(self.num_layers))

        # forward and backward
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, 
            cell_bw,
            input_embed, 
            dtype=tf.float32,
            sequence_length=seq_len
        )

        return outputs

    def logsumexp(self, x, axis=None):
        x_max = tf.reduce_max(x, reduction_indices=axis, keep_dims=True)
        x_max_ = tf.reduce_max(x, reduction_indices=axis)
        return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), reduction_indices=axis))

    def forward(self, observations, transitions, length, is_viterbi=True, return_best_seq=True):
        length = tf.reshape(length, [self.batch_size])
        transitions = tf.reshape(tf.concat([transitions]*self.batch_size, axis=0), [self.batch_size, (self.num_classes+1), (self.num_classes+1)])
        observations = tf.reshape(observations, [self.batch_size, self.num_steps + 2, (self.num_classes+1), 1])
        observations = tf.transpose(observations, [1, 0, 2, 3])
        previous = observations[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]
        for t in range(1, self.num_steps + 2):
            previous = tf.reshape(previous, [self.batch_size, (self.num_classes+1), 1])
            current = tf.reshape(observations[t, :, :, :], [self.batch_size, 1, (self.num_classes+1)])
            alpha_t = previous + current + transitions
            if is_viterbi:
                max_scores.append(tf.reduce_max(alpha_t, reduction_indices=1))
                max_scores_pre.append(tf.argmax(alpha_t, dimension=1))
            alpha_t = tf.reshape(self.logsumexp(alpha_t, axis=1), [self.batch_size, (self.num_classes+1), 1])
            alphas.append(alpha_t)
            previous = alpha_t           
            
        alphas = tf.reshape(tf.concat(alphas, axis=0), [self.num_steps + 2, self.batch_size, (self.num_classes+1), 1])
        alphas = tf.transpose(alphas, [1, 0, 2, 3])
        alphas = tf.reshape(alphas, [self.batch_size * (self.num_steps + 2), (self.num_classes+1), 1])

        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.num_steps + 2) + length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, (self.num_classes+1), 1])

        max_scores = tf.reshape(tf.concat(max_scores, axis=0), (self.num_steps + 1, self.batch_size, (self.num_classes+1)))
        max_scores_pre = tf.reshape(tf.concat(max_scores_pre, axis=0), (self.num_steps + 1, self.batch_size, (self.num_classes+1)))
        max_scores = tf.transpose(max_scores, [1, 0, 2])
        max_scores_pre = tf.transpose(max_scores_pre, [1, 0, 2])

        return tf.reduce_sum(self.logsumexp(last_alphas, axis=1)), max_scores, max_scores_pre        

    def train(self, sess, save_file, X_train, y_train, X_val, y_val):
        saver = tf.train.Saver()

        char2id, id2char = helper.loadMap("char2id")
        label2id, id2label = helper.loadMap("label2id")

        merged = tf.merge_all_summaries()
        summary_writer_train = tf.train.SummaryWriter('loss_log/train_loss', sess.graph)  
        summary_writer_val = tf.train.SummaryWriter('loss_log/val_loss', sess.graph)     
        
        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))

        cnt = 0
        for epoch in range(self.num_epochs):
            # shuffle train in each epoch
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)
            X_train = X_train[sh_index]
            y_train = y_train[sh_index]
            print "current epoch: %d" % (epoch)
            for iteration in range(num_iterations):
                # train
                X_train_batch, y_train_batch = helper.nextBatch(X_train, y_train, start_index=iteration * self.batch_size, batch_size=self.batch_size)
                y_train_weight_batch = 1 + np.array((y_train_batch == label2id['B']) | (y_train_batch == label2id['E']), float)
                transition_batch = helper.getTransition(y_train_batch)
                
                _, loss_train, max_scores, max_scores_pre, length, train_summary =\
                    sess.run([
                        self.optimizer, 
                        self.loss, 
                        self.max_scores, 
                        self.max_scores_pre, 
                        self.length,
                        self.train_summary
                    ], 
                    feed_dict={
                        self.targets_transition:transition_batch, 
                        self.inputs:X_train_batch, 
                        self.targets:y_train_batch, 
                        self.targets_weight:y_train_weight_batch
                    })

                predicts_train = self.viterbi(max_scores, max_scores_pre, length, predict_size=self.batch_size)
                if iteration % 10 == 0:
                    cnt += 1
                    precision_train, recall_train, f1_train = self.evaluate(X_train_batch, y_train_batch, predicts_train, id2char, id2label)
                    summary_writer_train.add_summary(train_summary, cnt)
                    print "iteration: %5d, train loss: %5d, train precision: %.5f, train recall: %.5f, train f1: %.5f" % (iteration, loss_train, precision_train, recall_train, f1_train)  
                    
                # validation
                if iteration % 100 == 0:
                    X_val_batch, y_val_batch = helper.nextRandomBatch(X_val, y_val, batch_size=self.batch_size)
                    y_val_weight_batch = 1 + np.array((y_val_batch == label2id['B']) | (y_val_batch == label2id['E']), float)
                    transition_batch = helper.getTransition(y_val_batch)
                    
                    loss_val, max_scores, max_scores_pre, length, val_summary =\
                        sess.run([
                            self.loss, 
                            self.max_scores, 
                            self.max_scores_pre, 
                            self.length,
                            self.val_summary
                        ], 
                        feed_dict={
                            self.targets_transition:transition_batch, 
                            self.inputs:X_val_batch, 
                            self.targets:y_val_batch, 
                            self.targets_weight:y_val_weight_batch
                        })
                    
                    predicts_val = self.viterbi(max_scores, max_scores_pre, length, predict_size=self.batch_size)
                    precision_val, recall_val, f1_val = self.evaluate(X_val_batch, y_val_batch, predicts_val, id2char, id2label)
                    summary_writer_val.add_summary(val_summary, cnt)
                    print "iteration: %5d, valid loss: %5d, valid precision: %.5f, valid recall: %.5f, valid f1: %.5f" % (iteration, loss_val, precision_val, recall_val, f1_val)

                    if f1_val > self.max_f1:
                        self.max_f1 = f1_val
                        save_path = saver.save(sess, save_file)
                        print "saved the best model with f1: %.5f" % (self.max_f1)

    def test(self, sess, X_test, X_test_str, output_path):
        char2id, id2char = helper.loadMap("char2id")
        label2id, id2label = helper.loadMap("label2id")
        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        print "number of iteration: " + str(num_iterations)
        with open(output_path, "wb") as outfile:
            for i in range(num_iterations):
                print "iteration: " + str(i + 1)
                results = []
                X_test_batch = X_test[i * self.batch_size : (i + 1) * self.batch_size]
                X_test_str_batch = X_test_str[i * self.batch_size : (i + 1) * self.batch_size]
                if i == num_iterations - 1 and len(X_test_batch) < self.batch_size:
                    X_test_batch = list(X_test_batch)
                    X_test_str_batch = list(X_test_str_batch)
                    last_size = len(X_test_batch)
                    X_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_test_str_batch += [['x' for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_test_batch = np.array(X_test_batch)
                    X_test_str_batch = np.array(X_test_str_batch)
                    results = self.predictBatch(sess, X_test_batch, X_test_str_batch, id2label)
                    results = results[:last_size]
                else:
                    X_test_batch = np.array(X_test_batch)
                    results = self.predictBatch(sess, X_test_batch, X_test_str_batch, id2label)
                
                for i in range(len(results)):
                    doc = ''.join(X_test_str_batch[i])
                    outfile.write(doc + "<@>" +" ".join(results[i]).encode(self.code) + "\n")

    def viterbi(self, max_scores, max_scores_pre, length, predict_size=128):
        best_paths = []
        for m in range(predict_size):
            path = []
            last_max_node = np.argmax(max_scores[m][length[m]])
            # last_max_node = 0
            for t in range(1, length[m] + 1)[::-1]:
                last_max_node = max_scores_pre[m][t][last_max_node]
                path.append(last_max_node)
            path = path[::-1]
            best_paths.append(path)
        return best_paths

    def predictBatch(self, sess, X, X_str, id2label):
        results = []
        length, max_scores, max_scores_pre = sess.run([self.length, self.max_scores, self.max_scores_pre], feed_dict={self.inputs:X})
        predicts = self.viterbi(max_scores, max_scores_pre, length, self.batch_size)
        for i in range(len(predicts)):
            x = ''.join(X_str[i]).decode(self.code)
            y_pred = ''.join([id2label[val] for val in predicts[i] if val != 5 and val != 0])
            entitys = helper.extractEntity(x, y_pred)
            results.append(entitys)
        return results

    def evaluate(self, X, y_true, y_pred, id2char, id2label, length=None):
        precision = -1.0
        recall = -1.0
        f1 = -1.0
        hit_num = 0
        pred_num = 0
        true_num = 0
        for i in range(len(y_true)):
            #x = ''.join([str(id2char[val].encode("utf-8")) for val in X[i]])
            #y = ''.join([str(id2label[val].encode("utf-8")) for val in y_true[i]])
            #x = ' '.join([id2char[val] for val in X[i]])
            #y = ' '.join([id2label[val] for val in y_true[i]])            
            #y_hat = ' '.join([id2label[val] for val in y_pred[i]  if val != 5])
            x = [id2char[val] for val in X[i] if val] # delete pad
            y = [id2label[val] for val in y_true[i] if val] # delete pad
            y_hat = [id2label[val] for val in y_pred[i]] # len(y_pred[i]) == len(y)
            if len(x) != len(y) or len(x)!= len(y_hat):
                pdb.set_trace()
            true_labels = extractEntity_new(x, y)
            pred_labels = extractEntity_new(x, y_hat)            
            hit_num += len(set(true_labels) & set(pred_labels))
            pred_num += len(set(pred_labels))
            true_num += len(set(true_labels))
        if pred_num != 0:
            precision = 1.0 * hit_num / pred_num
        if true_num != 0:
            recall = 1.0 * hit_num / true_num
        if precision > 0 and recall > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        return precision, recall, f1, [hit_num, pred_num, true_num]                                                                

def extractEntity_new(sentence, labels):
    entitys = []
    i = 0
    has_begin = False
    if len(sentence) != len(labels):
        pdb.set_trace()
    while i < len(sentence):
        if not has_begin:
            if labels[i] == 'S':
                entitys.append(sentence[i])
            else:
                if labels[i] == 'B':
                    e_begin = i
                    has_begin = True
        else:
            if labels[i] == 'E':
                entitys.append(''.join(sentence[e_begin: i+1]))
                has_begin = False
        i += 1
    return entitys

def extractEntity(sentence, labels):
    entitys = []
    re_entity = re.compile(r'BI*E') #
    m = re_entity.search(labels)
    while m:
        entity_labels = m.group()
        start_index = labels.find(entity_labels)
        entity = sentence[start_index:start_index + len(entity_labels)]
        labels = list(labels)
        # replace the "BM*E" with "OO*O"
        labels[start_index: start_index + len(entity_labels)] = ['O' for i in range(len(entity_labels))] 
        entitys.append(entity)
        labels = ''.join(labels)
        m = re_entity.search(labels)
    return entitys