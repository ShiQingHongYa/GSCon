
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
import numpy as np
import os
from .Sparsemax import Sparsemax

class GSCon(nn.Module):
    def __init__(self, args):
        super(GSCon, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        if args.dataset_name == 'sims':
            self.len_l, self.len_v, self.len_a = 39, 39, 39

        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        self.relu = nn.ReLU(inplace=True)
        output_dim = 1
        dim_fe = self.len_l - args.conv1d_kernel_size_l + 1
        uni_head = dim_fe * self.d_l
        cat_head = 2*uni_head

        # 1. Temporal convolutional layers for initial feature
        self.proj_language = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0,stride=args.conv1d_stride_size_l,bias=False)
        self.proj_audio = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0,stride=args.conv1d_stride_size_a,bias=False)
        self.proj_visual = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0,stride=args.conv1d_stride_size_v,bias=False)

        self.self_attentions_l_first = self.get_network(self_type='l')
        self.self_attentions_v_first = self.get_network(self_type='v')
        self.self_attentions_a_first = self.get_network(self_type='a')
        self.self_attentions_l_sec = self.get_network(self_type='l')
        self.self_attentions_v_sec = self.get_network(self_type='v')
        self.self_attentions_a_sec = self.get_network(self_type='a')
        self.self_attentions_l_thi = self.get_network(self_type='l')
        self.self_attentions_v_thi = self.get_network(self_type='v')
        self.self_attentions_a_thi = self.get_network(self_type='a')

        # common head
        self.conv_common1 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.conv_common2 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.class_proj_comm1 = nn.Linear(uni_head, uni_head)
        self.class_proj_comm2 = nn.Linear(uni_head, uni_head)
        self.class_out_comm = nn.Linear(uni_head, output_dim)

        # specific head
        self.class_proj3_l1 = nn.Linear(cat_head, cat_head)  # self.d_l, self.d_l 1760,1760   4600, 4600
        self.class_proj3_l2 = nn.Linear(cat_head, cat_head)
        self.class_proj3_v1 = nn.Linear(cat_head, cat_head)
        self.class_proj3_v2 = nn.Linear(cat_head, cat_head)
        self.class_proj3_a1 = nn.Linear(cat_head, cat_head)
        self.class_proj3_a2 = nn.Linear(cat_head, cat_head)

        self.class_proj3_l = nn.Linear(cat_head, output_dim)
        self.class_proj3_v = nn.Linear(cat_head, output_dim)
        self.class_proj3_a = nn.Linear(cat_head, output_dim)

        # LVAC head
        self.class_proj3_cat1 = nn.Linear(4 * self.d_l , 4 * self.d_l)  # 3 * self.d_l
        self.class_proj3_cat2 = nn.Linear(4 * self.d_l, 4 * self.d_l)  # 
        self.class_out_layer_cat = nn.Linear(4 * self.d_l, output_dim)

        # LVA head
        self.class_proj3_c1 = nn.Linear(3 * self.d_l, 3 * self.d_l)  # 3 * self.d_l
        self.class_proj3_c2 = nn.Linear(3 * self.d_l, 3 * self.d_l)  # 
        self.class_out_layer = nn.Linear(3 * self.d_l, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 3 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text, audio, video, is_distill=False):
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        proj_l = x_l if self.orig_d_l == self.d_l else self.proj_language(x_l)   # [16,50,46]
        proj_v = x_v if self.orig_d_v == self.d_v else self.proj_visual(x_v)
        proj_a = x_a if self.orig_d_a == self.d_a else self.proj_audio(x_a)



        proj_l = proj_l.permute(0, 2, 1)  # [16,46,50]  2, 0, 1
        proj_v = proj_v.permute(0, 2, 1)
        proj_a = proj_a.permute(0, 2, 1)

        # encoder
        l_att = self.self_attentions_l_first(proj_l)     # [16,46,50]
        v_att = self.self_attentions_v_first(proj_v)
        a_att = self.self_attentions_a_first(proj_a)

        # second four encoders
        sec_l = self.self_attentions_l_sec(l_att)    # [16,46,50]
        sec_v = self.self_attentions_v_sec(v_att)
        sec_a = self.self_attentions_a_sec(a_att)

        thi_l = self.self_attentions_l_thi(sec_l)  # [16,46,50]
        thi_v = self.self_attentions_v_thi(sec_v)
        thi_a = self.self_attentions_a_thi(sec_a)

        common_l = self.conv_common2(F.relu(self.conv_common1(thi_l.unsqueeze(1)), inplace=True)).squeeze()
        common_v = self.conv_common2(F.relu(self.conv_common1(thi_v.unsqueeze(1)), inplace=True)).squeeze()
        common_a = self.conv_common2(F.relu(self.conv_common1(thi_a.unsqueeze(1)), inplace=True)).squeeze()

        common = (common_l + common_v + common_a)/3

        cat_l = torch.cat((thi_l, common), dim=2)
        cat_v = torch.cat((thi_v, common), dim=2)
        cat_a = torch.cat((thi_a, common), dim=2)
        thi_cat_max1 = torch.cat((thi_l.transpose(1, 0)[-1], thi_v.transpose(1, 0)[-1], thi_a.transpose(1, 0)[-1], common.transpose(1, 0)[-1]),dim=1)

        final_cat_max1 = self.class_proj3_cat2(F.dropout(F.relu(self.class_proj3_cat1(thi_cat_max1), inplace=True), p=self.output_dropout,training=self.training))
        output_final_cat_max = self.class_out_layer_cat(final_cat_max1)

        out_l = cat_l.contiguous().view(cat_l.size(0), -1)
        out_v = cat_v.contiguous().view(cat_v.size(0), -1)
        out_a = cat_a.contiguous().view(cat_a.size(0), -1)
        common1 = common.contiguous().view(common.size(0), -1)

        output_l = self.class_proj3_l2(
            F.dropout(F.relu(self.class_proj3_l1(out_l), inplace=True), p=self.output_dropout, training=self.training))
        # output_l += out_l
        output_l = self.class_proj3_l(output_l)

        output_v = self.class_proj3_v2(
            F.dropout(F.relu(self.class_proj3_v1(out_v), inplace=True), p=self.output_dropout, training=self.training))
        # output_v += out_v
        output_v = self.class_proj3_v(output_v)

        output_a = self.class_proj3_a2(
            F.dropout(F.relu(self.class_proj3_a1(out_a), inplace=True), p=self.output_dropout, training=self.training))
        # output_a += out_a
        output_a = self.class_proj3_a(output_a)

        output_comm = self.class_proj_comm2(
            F.dropout(F.relu(self.class_proj_comm1(common1), inplace=True), p=self.output_dropout, training=self.training))
        # output_comm += common1
        output_comm = self.class_out_comm(output_comm)

        thi_l1 = thi_l.permute(1, 0, 2)
        thi_v1 = thi_v.permute(1, 0, 2)
        thi_a1 = thi_a.permute(1, 0, 2)

        final = torch.cat((thi_l1[-1], thi_v1[-1], thi_a1[-1]), dim=1)

        final_m = self.class_proj3_c2(
            F.dropout(F.relu(self.class_proj3_c1(final), inplace=True), p=self.output_dropout, training=self.training))
        final_m = torch.sigmoid(final_m)
        output_final = self.class_out_layer(final_m)

        res = {
            'thi_cat_max': final_cat_max1,
            'thi_l': thi_l,
            'thi_v': thi_v,
            'thi_a': thi_a,
            'common_l': common_l,
            'common_v': common_v,
            'common_a': common_a,
            'common': common,
            'output_final_l': output_l,
            'output_final_v': output_v,
            'output_final_a': output_a,
            'output_common': output_comm,
            'final_output_three': output_final,
            'final_output': output_final_cat_max
        }

        return res