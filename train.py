import time
import numpy as np
import tensorflow as tf

from models import spMMGNN
from utils import process
from metrics import eva_SVM, eva_Kmeans

config = tf.ConfigProto(allow_soft_placement=True)

dataset = 'imdb'
checkpt_file = 'pre_trained/{}/spMMGNN.ckpt'.format(dataset)
# training params
batch_size = 1
nb_epochs = 500
patience = 30
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [8]
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = spMMGNN

# load data
homo_adj_list, hete_adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)

nb_nodes = [fea.shape[0] for fea in fea_list]
ft_size = [fea.shape[1] for fea in fea_list]
nb_classes = y_train.shape[1]

# one graph
fea_list = [fea[np.newaxis] for fea in fea_list]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

# get bias
biases_list = [process.preprocess_adj_bias(adj) for adj in homo_adj_list]

print('build graph...')
with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in_list = [tf.placeholder(dtype=tf.float32,
                                      shape=(batch_size, nb_nodes[i], ft_size[i]),
                                      name='ftr_in_{}'.format(i))
                       for i in range(len(fea_list))]
        bias_in_list = [tf.sparse_placeholder(dtype=tf.float32) for _ in biases_list]
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes[2], nb_classes), name='lbl_in')
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes[2]),name='msk_in')
        attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
        is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
    # forward
    logits, final_embedding, att_val = model.inference(inputs_list=ftr_in_list, nb_classes=nb_classes,
                                                       nb_nodes=nb_nodes, training=is_train,
                                                       attn_drop=attn_drop, ffd_drop=ffd_drop,
                                                       hete_adj_list=hete_adj_list,
                                                       bias_mat_list=bias_in_list,
                                                       hid_units=hid_units, n_heads=n_heads,
                                                       residual=residual, activation=nonlinearity)

    # cal masked_loss
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    # optimzie
    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = fea_list[0].shape[0]

            # ================   training    ============
            while tr_step * batch_size < tr_size:

                fd1 = {i: d[tr_step * batch_size: (tr_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_train[tr_step * batch_size: (tr_step + 1) * batch_size],
                       msk_in: train_mask[tr_step * batch_size: (tr_step + 1) * batch_size],
                       is_train: True,
                       attn_drop: 0.5,
                       ffd_drop: 0.5}
                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                _, loss_value_tr, acc_tr, att_val_train = sess.run([train_op,loss,accuracy,att_val], feed_dict=fd)

                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = fea_list[0].shape[0]
            # =============   val       =================
            while vl_step * batch_size < vl_size:

                fd1 = {i: d[vl_step * batch_size:(vl_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d for i, d in zip(bias_in_list, biases_list)}
                fd3 = {lbl_in: y_val[vl_step * batch_size:(vl_step + 1) * batch_size],
                       msk_in: val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
                       is_train: False,
                       attn_drop: 0.0,
                       ffd_drop: 0.0}

                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                loss_value_vl, acc_vl = sess.run([loss, accuracy], feed_dict=fd)
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print('Epoch: {}, att_val: {}'.format(epoch, np.mean(att_val_train, axis=0)))
            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                  (train_loss_avg / tr_step, train_acc_avg / tr_step,
                   val_loss_avg / vl_step, val_acc_avg / vl_step))

            if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
                if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg / vl_step
                    vlss_early_model = val_loss_avg / vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn,
                          ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ',
                          vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)
        print('load model from : {}'.format(checkpt_file))
        ts_size = fea_list[0].shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(ftr_in_list, fea_list)}
            fd2 = {i: d for i, d in zip(bias_in_list, biases_list)}
            fd3 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                   msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                   is_train: False,
                   attn_drop: 0.0,
                   ffd_drop: 0.0}

            fd = fd1
            fd.update(fd2)
            fd.update(fd3)
            loss_value_ts, acc_ts, final_embed = sess.run([loss, accuracy, final_embedding], feed_dict=fd)
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss / ts_step,
              '; Test accuracy:', ts_acc / ts_step)

        print('start svm, kmean...')
        xx = np.expand_dims(final_embed, axis=0)[test_mask]
        yy = y_test[test_mask]
        eva_SVM(xx, yy)
        eva_Kmeans(xx, yy)

        sess.close()