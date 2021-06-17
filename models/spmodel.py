import numpy as np
import tensorflow as tf
from utils import layers
from models.base_gattn import BaseGAttN


class spMMGNN(BaseGAttN):
    def inference(inputs_list, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, hete_adj_list,
                  activation=tf.nn.elu, residual=False, mp_att_size=128):
        embed_list = []
        # diff-type
        for i, bias_mat in enumerate(bias_mat_list[:1]):
            muti_paper_embed = []
            for _ in range(n_heads[0]):
                projectW = tf.Variable(tf.random_normal([hid_units[0], hid_units[0]], stddev=0.1))
                author_embed = layers.sp_attn_head(inputs_list[i], bias_mat=bias_mat,
                                              out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes[i],
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False)

                author_embed = tf.squeeze(author_embed, 0)
                paper_embed = tf.nn.tanh(tf.tensordot(hete_adj_list[i], author_embed, axes=1))
                paper_embed = tf.tensordot(paper_embed, projectW, axes=1)
                paper_embed = tf.expand_dims(paper_embed, 0)
                muti_paper_embed.append(paper_embed)
            h_1 = tf.concat(muti_paper_embed, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))

        # same-type
        for bias_mat in bias_mat_list[2:]:
            muti_paper_embed = []
            for _ in range(n_heads[0]):
                paper_embed = layers.sp_attn_head(inputs_list[2], bias_mat=bias_mat,
                                              out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes[2],
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False)

                muti_paper_embed.append(paper_embed)
            h_2 = tf.concat(muti_paper_embed, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_2), axis=1))

        all_paper_embed = tf.concat(embed_list, axis=1)
        print(all_paper_embed.shape)
        final_embed, att_val = layers.SimpleAttLayer(all_paper_embed, mp_att_size,time_major=False,return_alphas=True)

        out = []
        for i in range(n_heads[-1]):
            out.append(tf.layers.dense(final_embed, nb_classes, activation=None))

        logits = tf.add_n(out) / n_heads[-1]
        print('all right')

        logits = tf.expand_dims(logits, axis=0)
        return logits, final_embed, att_val