"""
here is the mian backbone for DMD containing feature decoupling and multimodal transformers
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder
import numpy as np
import os
from .Sparsemax import Sparsemax

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class DMD(nn.Module):
    def __init__(self, args):
        super(DMD, self).__init__()
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
        self.share = nn.Parameter(torch.randn(46, dst_feature_dims))
        self.activate = Sparsemax(dim=-1)    # nn.Softmax(dim=-1)
        self.layernor = nn.LayerNorm(normalized_shape = [3, dst_feature_dims, dst_feature_dims])
        self.batchnor = nn.BatchNorm2d(num_features=3)
        self.share_attention = self.get_network(self_type='l')   # l_mem
        # self.activate_sparse = nn.Sp
        combined_dim_low = self.d_a
        combined_dim_high = 2 * self.d_a
        combined_dim = 2 * (self.d_l + self.d_a + self.d_v) + self.d_l * 3
        output_dim = 1

        # 1. Temporal convolutional layers for initial feature
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # self.learn_impor_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        # self.learn_impor_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        # self.learn_impor_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # # 2.1 Modality-specific encoder
        # self.encoder_s_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        # self.encoder_s_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        # self.encoder_s_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        #
        # # 2.2 Modality-invariant encoder
        # self.encoder_c = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        #
        # # 3. Decoder for reconstruct three modalities
        # self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        # self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        # self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)
        #
        # # for calculate cosine sim between s_x
        # self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        # self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        # self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)
        #
        # # for align c_l, c_v, c_a
        # self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        # self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        # self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        self.self_attentions_first_c_l = self.get_network(self_type='l')
        self.self_attentions_first_c_v = self.get_network(self_type='v')
        self.self_attentions_first_c_a = self.get_network(self_type='a')
        self.self_attentions_sec_c_l = self.get_network(self_type='l')
        self.self_attentions_sec_c_v = self.get_network(self_type='v')
        self.self_attentions_sec_c_a = self.get_network(self_type='a')
        self.self_attentions_sec_c_l2 = self.get_network(self_type='l')
        self.self_attentions_sec_c_v2 = self.get_network(self_type='v')
        self.self_attentions_sec_c_a2 = self.get_network(self_type='a')
        # self.self_attentions_thi_c_l = self.get_network(self_type='l')
        # self.self_attentions_thi_c_v = self.get_network(self_type='v')
        # self.self_attentions_thi_c_a = self.get_network(self_type='a')
        # self.self_attentions_third_f1 = self.get_network(self_type='l_mem')   # l_mem
        # self.self_attentions_third_f2 = self.get_network(self_type='l_mem')

        # self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        # self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        # self.out_layer_c = nn.Linear(2300, output_dim)     # (self.d_l, output_dim)  2300

        #

        self.weight1_l = nn.Linear(2300, 2300)   # (self.d_l, self.d_l)
        self.weight1_v = nn.Linear(2300, 2300)
        self.weight1_a = nn.Linear(2300, 2300)

        self.out_layer_l = nn.Linear(2300, output_dim)
        self.out_layer_v = nn.Linear(2300, output_dim)
        self.out_layer_a = nn.Linear(2300, output_dim)
        # self.weight_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
        # second project
        # self.weight2_l = nn.Linear(2300, 2300)   # (self.d_l, self.d_l)
        # self.weight2_v = nn.Linear(2300, 2300)
        # self.weight2_a = nn.Linear(2300, 2300)
        # first encoder
        # self.proj1_l1 = nn.Linear(2300, 2300)  # self.d_l, self.d_l
        # self.proj1_l2 = nn.Linear(2300, 2300)
        # self.proj1_v1 = nn.Linear(2300, 2300)
        # self.proj1_v2 = nn.Linear(2300, 2300)
        # self.proj1_a1 = nn.Linear(2300, 2300)
        # self.proj1_a2 = nn.Linear(2300, 2300)

        # self.out_layer_first_l = nn.Linear(2300, output_dim)
        # self.out_layer_first_v = nn.Linear(2300, output_dim)
        # self.out_layer_first_a = nn.Linear(2300, output_dim)


        # second encoder
        # self.proj2_l1 = nn.Linear(2300, 2300)  # self.d_l, self.d_l
        # self.proj2_l2 = nn.Linear(2300, 2300)
        # self.proj2_v1 = nn.Linear(2300, 2300)
        # self.proj2_v2 = nn.Linear(2300, 2300)
        # self.proj2_a1 = nn.Linear(2300, 2300)
        # self.proj2_a2 = nn.Linear(2300, 2300)
        #
        # self.out_layer_sec_l = nn.Linear(2300, output_dim)
        # self.out_layer_sec_v = nn.Linear(2300, output_dim)
        # self.out_layer_sec_a = nn.Linear(2300, output_dim)

        # final project
        # self.weight_l = nn.Linear(self.d_l, self.d_l)  # (self.d_l, self.d_l)
        # self.weight_v = nn.Linear(self.d_l, self.d_l)
        # self.weight_a = nn.Linear(self.d_l, self.d_l)
        # self.weight_fuse = nn.Linear(3 * self.d_l, 3 * self.d_l)

        self.proj3_c1 = nn.Linear(3 * self.d_l, 3 * self.d_l)  # 3 * self.d_l
        self.proj3_c2 = nn.Linear(3 * self.d_l, 3 * self.d_l)   # (combined_dim, combined_dim)
        self.out_layer = nn.Linear(3 * self.d_l, output_dim)

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

        proj_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)   # [16,50,46]
        proj_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        # original encoder+encoder+encoder
        # proj_l = proj_l.permute(2, 0, 1)  # [46,16,50]
        # proj_v = proj_v.permute(2, 0, 1)
        # proj_a = proj_a.permute(2, 0, 1)
        #
        # l_att = self.self_attentions_first_c_l(proj_l)
        # v_att = self.self_attentions_first_c_v(proj_v)
        # a_att = self.self_attentions_first_c_a(proj_a)
        #
        # sec_l = self.self_attentions_sec_c_l(l_att)    # [16,46,50]
        # sec_v = self.self_attentions_sec_c_v(v_att)
        # sec_a = self.self_attentions_sec_c_a(a_att)
        #
        # third_l = self.self_attentions_thi_c_l(sec_l)  # [16,46,50]
        # third_v = self.self_attentions_thi_c_v(sec_v)
        # third_a = self.self_attentions_thi_c_a(sec_a)
        #
        # fuse = torch.cat((sec_l, sec_v, sec_a), 2)
        # # fuse = torch.cat((third_l, third_v, third_a), 2)
        #
        # # third four encoders
        # final1 = self.self_attentions_third_f1(fuse)
        # # final = self.self_attentions_third_f2(final1)
        # # final = final1.permute(1, 0, 2)
        # final_l = final1[-1]
        # # final = final.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        # final_m = self.proj3_c2(
        #     F.dropout(F.relu(self.proj3_c1(final_l), inplace=True), p=self.output_dropout, training=self.training))
        #
        # final_f = torch.sigmoid(final_m)
        #
        # output_final = self.out_layer(final_f)
        # end

        proj_l = proj_l.permute(0, 2, 1)  # [46,16,50]  2, 0, 1
        proj_v = proj_v.permute(0, 2, 1)
        proj_a = proj_a.permute(0, 2, 1)

        # first encoder
        l_att = self.self_attentions_first_c_l(proj_l)     # [46,16,50]
        a_att = self.self_attentions_first_c_a(proj_a)
        v_att = self.self_attentions_first_c_v(proj_v)

        l_att = l_att.contiguous().view(x_l.size(0), -1)    # [16,2300]  .transpose(0, 1)
        v_att = v_att.contiguous().view(x_v.size(0), -1)  # [16,2300]
        a_att = a_att.contiguous().view(x_a.size(0), -1)  # [16,2300]
        #
        # h_l = torch.sigmoid(self.proj1_l2(
        #     F.dropout(F.relu(self.proj1_l1(l_att1), inplace=True), p=self.output_dropout, training=self.training)))
        # midd_sup_l = self.out_layer_first_l(h_l)
        #
        # h_v = torch.sigmoid(self.proj1_v2(
        #     F.dropout(F.relu(self.proj1_v1(v_att1), inplace=True), p=self.output_dropout, training=self.training)))
        # midd_sup_v = self.out_layer_first_l(h_v)
        #
        # h_a = torch.sigmoid(self.proj1_a2(
        #     F.dropout(F.relu(self.proj1_a1(a_att1), inplace=True), p=self.output_dropout, training=self.training)))
        # midd_sup_a = self.out_layer_first_l(h_a)
        #
        # midd_l = l_att * proj_l      # attention score   [46,16,50]
        # midd_v = v_att * proj_v
        # midd_a = a_att * proj_a
        #
        # midd_l = midd_l.permute(1, 0, 2).contiguous().view(l_att.size(1), -1)
        # midd_v = midd_v.permute(1, 0, 2).contiguous().view(v_att.size(1), -1)
        # midd_a = midd_a.permute(1, 0, 2).contiguous().view(a_att.size(1), -1)

        h_l = torch.sigmoid(self.weight1_l(l_att))  # [16, 2300]
        h_v = torch.sigmoid(self.weight1_v(v_att))
        h_a = torch.sigmoid(self.weight1_a(a_att))

        midd_sup_l = self.out_layer_l(h_l)  # output [16, 1]  h_l
        midd_sup_v = self.out_layer_v(h_v)
        midd_sup_a = self.out_layer_a(h_a)

        midd_l = h_l * (proj_l.contiguous().view(x_l.size(0), -1))  # attention score   [16, 2300]
        midd_v = h_v * (proj_v.contiguous().view(x_v.size(0), -1))
        midd_a = h_a * (proj_a.contiguous().view(x_a.size(0), -1))

        # noise
        noise_l = torch.from_numpy(np.random.randn(midd_l.size(0), midd_l.size(1)).astype(np.float32)).cuda()
        noise_v = torch.from_numpy(np.random.randn(midd_v.size(0), midd_v.size(1)).astype(np.float32)).cuda()
        noise_a = torch.from_numpy(np.random.randn(midd_a.size(0), midd_a.size(1)).astype(np.float32)).cuda()

        index_l = torch.argsort(midd_l,dim=1,descending=False)  # small to big
        index_v = torch.argsort(midd_v,dim=1,descending=False)
        index_a = torch.argsort(midd_a,dim=1,descending=False)

        important_ratio = 0.5     # mask important part
        unimportant = int(midd_l.size(1) * (1-important_ratio))

        # ids_unimportant = index_l[:, 0:unimportant]   # mask unimportant    # [:, unimportant:midd_l.size(1)] mask important
        ids_important_l = index_l[:, 0:unimportant]
        ids_l = torch.LongTensor(ids_important_l.cpu()).cuda()
        noise_l_important = torch.scatter(midd_l, dim=1, index=ids_l, src=noise_l)

        ids_important_v = index_v[:, 0:unimportant]
        ids_v = torch.LongTensor(ids_important_v.cpu()).cuda()
        noise_v_important = torch.scatter(midd_v, dim=1, index=ids_v, src=noise_v)

        ids_important_a = index_a[:, 0:unimportant]
        ids_a = torch.LongTensor(ids_important_a.cpu()).cuda()
        noise_a_important = torch.scatter(midd_a, dim=1, index=ids_a, src=noise_a)

        # second four encoders
        sec_l1 = self.self_attentions_sec_c_l(noise_l_important.contiguous().view(noise_l_important.size(0),46,50))    # [16,46,50]
        sec_v1 = self.self_attentions_sec_c_v(noise_v_important.contiguous().view(noise_v_important.size(0),46,50))
        sec_a1 = self.self_attentions_sec_c_a(noise_a_important.contiguous().view(noise_a_important.size(0),46,50))

        # # noise for sec
        # noise_sec_l = torch.from_numpy(np.random.randn(sec_l1.size(0), sec_l1.size(1), sec_l1.size(2)).astype(np.float32)).cuda()
        # noise_sec_v = torch.from_numpy(np.random.randn(sec_v1.size(0), sec_v1.size(1), sec_v1.size(2)).astype(np.float32)).cuda()
        # noise_sec_a = torch.from_numpy(np.random.randn(sec_a1.size(0), sec_a1.size(1), sec_a1.size(2)).astype(np.float32)).cuda()
        #
        # mask_l = torch.randint(0, 2, size=sec_l1.shape).bool()
        # mask_v = torch.randint(0, 2, size=sec_v1.shape).bool()
        # mask_a = torch.randint(0, 2, size=sec_a1.shape).bool()
        #
        # sec_l1[mask_l] += noise_sec_l[mask_l]
        # sec_v1[mask_v] += noise_sec_v[mask_v]
        # sec_a1[mask_a] += noise_sec_a[mask_a]

        #
        # sec_l1 = sec_l1 + 0.5*noise_sec_l
        # sec_v1 = sec_v1 + 0.5*noise_sec_v
        # sec_a1 = sec_a1 + 0.5*noise_sec_a

        sec_l = self.self_attentions_sec_c_l2(sec_l1)  # [16,46,50]
        sec_v = self.self_attentions_sec_c_v2(sec_v1)
        sec_a = self.self_attentions_sec_c_a2(sec_a1)

        sec_l = sec_l.permute(1, 0, 2)
        sec_v = sec_v.permute(1, 0, 2)
        sec_a = sec_a.permute(1, 0, 2)

        final = torch.cat((sec_l[-1], sec_v[-1], sec_a[-1]), dim=1)

        final_m = self.proj3_c2(
            F.dropout(F.relu(self.proj3_c1(final), inplace=True), p=self.output_dropout, training=self.training))

        final_f = torch.sigmoid(final_m)
        output_final = self.out_layer(final_f)

        # # all zeros
        # noise_l = torch.full((midd_l.size(0), midd_l.size(1)), 9999).float().cuda()
        # noise_v = torch.full((midd_v.size(0), midd_v.size(1)), 9999).float().cuda()
        # noise_a = torch.full((midd_a.size(0), midd_a.size(1)), 9999).float().cuda()
        #
        #
        # index_l = torch.argsort(midd_l, dim=1, descending=False)  # small to big
        # index_v = torch.argsort(midd_v, dim=1, descending=False)
        # index_a = torch.argsort(midd_a, dim=1, descending=False)
        #
        # important_ratio = 0.5  # mask important part
        # unimportant = int(midd_l.size(1) * (1 - important_ratio))
        #
        # # ids_unimportant = index_l[:, 0:unimportant]
        # # ids_important_l = index_l[:, unimportant:midd_l.size(1)]   # 替换重要的数据
        # ids_important_l = index_l[:, 0:unimportant]  # 替换不重要的部分
        # ids_l = torch.LongTensor(ids_important_l.cpu()).cuda()
        # noise_l_important = torch.scatter(midd_l, dim=1, index=ids_l, src=noise_l)
        #
        # # ids_important_v = index_v[:, unimportant:midd_v.size(1)]
        # ids_important_v = index_v[:, 0:unimportant]
        # ids_v = torch.LongTensor(ids_important_v.cpu()).cuda()
        # noise_v_important = torch.scatter(midd_v, dim=1, index=ids_v, src=noise_v)
        #
        # # ids_important_a = index_a[:, unimportant:midd_a.size(1)]
        # ids_important_a = index_a[:, 0:unimportant]
        # ids_a = torch.LongTensor(ids_important_a.cpu()).cuda()
        # noise_a_important = torch.scatter(midd_a, dim=1, index=ids_a, src=noise_a)
        #
        # # delete 9999
        # noise_l_important = noise_l_important[noise_l_important != 9999].view(midd_l.size(0), -1)   # [16, 1150]
        # noise_v_important = noise_v_important[noise_v_important != 9999].view(midd_v.size(0), -1)
        # noise_a_important = noise_a_important[noise_a_important != 9999].view(midd_a.size(0), -1)
        #
        #
        # # second four encoders
        # noise_l_important = noise_l_important.view(noise_l_important.size(0), 23, 50)
        # noise_l_important = noise_l_important.permute(1,0,2)
        #
        # noise_v_important = noise_v_important.view(noise_v_important.size(0), 23, 50)
        # noise_v_important = noise_v_important.permute(1,0,2)
        #
        # noise_a_important = noise_a_important.view(noise_a_important.size(0), 23, 50)
        # noise_a_important = noise_a_important.permute(1,0,2)    # [23, 16,50]
        #
        # sec_l = self.self_attentions_c_v(noise_l_important)  # [23,16,50]
        # sec_v = self.self_attentions_c_v(noise_v_important)
        # sec_a = self.self_attentions_c_v(noise_a_important)
        #
        # sec_out_l = sec_l.transpose(0,1).contiguous().view(sec_l.size(1), -1)  # [16,1150]
        # sec_out_v = sec_v.transpose(0,1).contiguous().view(sec_v.size(1), -1)
        # sec_out_a = sec_a.transpose(0,1).contiguous().view(sec_a.size(1), -1)
        #
        # sec_out_l = self.proj2_l2(
        #     F.dropout(F.relu(self.proj2_l1(sec_out_l), inplace=True), p=self.output_dropout, training=self.training))
        # sec_out_l_sig = torch.sigmoid(sec_out_l)
        # sec_out_l = self.out_layer_sec_l(sec_out_l_sig)
        #
        # sec_out_v = self.proj2_v2(
        #     F.dropout(F.relu(self.proj2_v1(sec_out_v), inplace=True), p=self.output_dropout, training=self.training))
        # sec_out_v_sig = torch.sigmoid(sec_out_v)
        # sec_out_v = self.out_layer_sec_v(sec_out_v_sig)
        #
        # sec_out_a = self.proj2_a2(
        #     F.dropout(F.relu(self.proj2_a1(sec_out_a), inplace=True), p=self.output_dropout, training=self.training))
        # sec_out_a_sig = torch.sigmoid(sec_out_a)
        # sec_out_a = self.out_layer_sec_a(sec_out_a_sig)
        #
        # sec_important_l = sec_out_l_sig * noise_l_important.contiguous().view(x_l.size(0), -1)   # [16, 1150]
        # sec_important_v = sec_out_v_sig * noise_v_important.contiguous().view(x_v.size(0), -1)
        # sec_important_a = sec_out_a_sig * noise_a_important.contiguous().view(x_a.size(0), -1)
        #
        # mean_important_l = torch.mean(sec_important_l, dim=1)
        # mean_important_v = torch.mean(sec_important_v, dim=1)
        # mean_important_a = torch.mean(sec_important_a, dim=1)
        #
        # # length_fuse = int(3*sec_important_l.size(1)/2)
        # length_fuse = 2100
        # fuse = torch.zeros(sec_important_l.size(0), length_fuse).cuda()
        # for i in range(len(mean_important_l)):
        #     sample_l = sec_important_l[i][sec_important_l[i] > mean_important_l[i]]
        #     sample_v = sec_important_v[i][sec_important_v[i] > mean_important_v[i]]
        #     sample_a = sec_important_a[i][sec_important_a[i] > mean_important_a[i]]
        #     sample_length = len(sample_l)+len(sample_v)+len(sample_a)
        #     if sample_length != length_fuse:
        #         if sample_length > length_fuse:
        #             min_sample = min([len(sample_l), len(sample_v), len(sample_a)])
        #             sample_need = length_fuse - (sample_length - min_sample)
        #             if sample_need < 0:
        #                 if min_sample == len(sample_l):
        #                     sample_a = sample_a[:sample_need]
        #                     sample_fuse = torch.cat([sample_v, sample_a], dim=0)
        #                 elif min_sample == len(sample_v):
        #                     sample_a = sample_a[:sample_need]
        #                     sample_fuse = torch.cat([sample_l, sample_a], dim=0)
        #                 else:
        #                     sample_v = sample_v[:sample_need]
        #                     sample_fuse = torch.cat([sample_l, sample_v], dim=0)
        #             else:
        #                 if min_sample == len(sample_l):
        #                     sample_l = sample_l[:sample_need]
        #                 elif min_sample == len(sample_v):
        #                     sample_v = sample_v[:sample_need]
        #                 else:
        #                     sample_a = sample_a[:sample_need]
        #                 sample_fuse = torch.cat([sample_l, sample_v, sample_a],dim=0)
        #         else:
        #             sample_fuse = torch.zeros(length_fuse)
        #             sample_fuse[:sample_length] = torch.cat([sample_l, sample_v, sample_a],dim=0)    # supple zero
        #     else:
        #         sample_fuse = torch.cat([sample_l, sample_v, sample_a], dim=0)
        #
        #     fuse[i] = sample_fuse
        #
        # fuse = fuse.view(fuse.size(0),42,50)
        # fuse = fuse.permute(1,0,2)

        # third four encoders
        # final = self.self_attentions_c_a(fuse)  # [46,16,50]
        # final = final[-1]
        # final_f = torch.sigmoid(self.proj3_c2(
        #     F.dropout(F.relu(self.proj3_c1(final), inplace=True), p=self.output_dropout, training=self.training)))
        #
        # output_final = self.out_layer(final_f)
        # res = {
        #     'logit_sec_l': final_l,
        #     'logit_sec_v': final_v,
        #     'logit_sec_a': final_a,
        #     'logit_sec_c': final_c,
        #     'final_output': output_final
        # }
        res = {
            'midd_sup_l': midd_sup_l,
            'midd_sup_v': midd_sup_v,
            'midd_sup_a': midd_sup_a,
            'final_output': output_final
        }
        # res = {
        #     'midd_sup_l': midd_sup_l,
        #     'midd_sup_v': midd_sup_v,
        #     'midd_sup_a': midd_sup_a,
        #     'sec_out_l': sec_out3_l,
        #     'sec_out_v': sec_out3_v,
        #     'sec_out_a': sec_out3_a,
        #     'final_output': output_final
        # }
        return res