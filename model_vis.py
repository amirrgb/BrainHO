'''----------------------------------------------mymodel vis----------------------------------------------
for vis, modified output
'''

import torch
import torch.nn as nn
from functools import partial
from timm.layers import DropPath
from timm.models.layers import trunc_normal_
from sparsemax import Sparsemax


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class HierarchicalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_sg=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.num_sg = num_sg
        self.dim = dim
        '''----------------------------------------------SG 2 Node----------------------------------------------'''
        self.q_sg = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_node_cross = nn.Linear(dim, dim * 2, bias=qkv_bias)
        '''----------------------------------------------Node 2 Node----------------------------------------------'''
        self.qkv_node_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        '''----------------------------------------------graph 2 sg----------------------------------------------'''
        self.q_global = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_gsg_reproj = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sparsemax = Sparsemax(dim=-1)

        self.norm_sg = nn.LayerNorm(dim)
        self.norm_node = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        # x: [Global(1), SG(8), Nodes(200)]
        x_g = x[:, 0:1, :]  # [B, 1, C]
        x_sg = x[:, 1:self.num_sg + 1, :]  # [B, 8, C]
        x_n = x[:, self.num_sg + 1:, :]  # [B, 200, C]

        '''----------------------------------------------Node 2 Node----------------------------------------------'''
        qkv_n = self.qkv_node_self(x_n).reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_n_self, k_n_self, v_n_self = qkv_n[0], qkv_n[1], qkv_n[2]

        attn_node = (q_n_self @ k_n_self.transpose(-2, -1)) * self.scale
        # attn_node = (q_n_self @ k_n_self.transpose(-2, -1))
        # for vis
        attn_node_dtc = attn_node.clone().detach()
        # attn_node = attn_node*self.scale
        attn_node = attn_node.softmax(dim=-1)
        attn_node_dtc = attn_node.clone().detach()
        attn_node = self.attn_drop(attn_node)
        x_n_new = (attn_node @ v_n_self).transpose(1, 2).reshape(B, x_n.shape[1], C)
        x_n_updated = self.norm_node(x_n + x_n_new)

        '''----------------------------------------------SG 2 Node----------------------------------------------'''
        # q_sg: [B, 8, C] -> [B, H, 8, D]
        q_sg_vec = self.q_sg(x_sg).reshape(B, self.num_sg, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # k_n, v_n: [B, 200, C] -> [B, H, 200, D]
        kv_n = self.kv_node_cross(x_n_updated).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k_n_cross, v_n_cross = kv_n[0], kv_n[1]
        attn_sgn = (q_sg_vec @ k_n_cross.transpose(-2, -1)) * self.scale
        attn_sgn = self.sparsemax(attn_sgn)
        attn_sgn_dtc = attn_sgn.clone().detach()  # 可视化
        attn_sgn = self.attn_drop(attn_sgn)
        x_sg_new = (attn_sgn @ v_n_cross).transpose(1, 2).reshape(B, self.num_sg, C)
        x_sg_updated = self.norm_sg(x_sg + x_sg_new)

        '''----------------------------------------------G 2 SG----------------------------------------------'''
        x_gsg_concat = torch.cat([x_g, x_sg_updated], dim=1)  # [B, 9, C]
        kv_gsg = self.kv_gsg_reproj(x_gsg_concat).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1,
                                                                                                           4)
        k_gsg_new, v_gsg_new = kv_gsg[0], kv_gsg[1]
        q_g_vec = self.q_global(x_g).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_gsg = (q_g_vec @ k_gsg_new.transpose(-2, -1)) * self.scale
        attn_gsg = attn_gsg.softmax(dim=-1)
        attn_gsg_dtc = attn_gsg.clone().detach()  # hook
        attn_gsg_dtc = attn_gsg_dtc[:, :, :, 1:]
        attn_gsg = self.attn_drop(attn_gsg)
        x_g_new = (attn_gsg @ v_gsg_new).transpose(1, 2).reshape(B, 1, C)

        final_x = torch.cat([x_g_new, x_sg_updated, x_n_updated], dim=1)

        attn_g2node_step1 = torch.matmul(attn_gsg_dtc, attn_sgn_dtc)
        attn_g2node = torch.matmul(attn_g2node_step1, attn_node_dtc)

        return final_x, attn_node_dtc, attn_sgn_dtc, attn_gsg_dtc, attn_g2node


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-5)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = HierarchicalAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = 1024
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        o, attn_node, attn_sgn, attn_gsg, attn_g2node = self.attn(self.norm1(x))
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn_node, attn_sgn, attn_gsg, attn_g2node


class mymodel_vis(torch.nn.Module):

    def __init__(self,
                 input_dim=200,
                 embed_dim=200,
                 hidden_dim=1024,
                 reduct_dim=8,
                 num_classes=2,
                 num_heads=8,
                 depth=2,
                 dropout=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 num_node=200,
                 num_sg=8
                 ):
        super(mymodel_vis, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.reduct_dim = reduct_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.depth = depth
        self.dropout = dropout
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.num_node = num_node
        self.num_sg = num_sg

        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                drop=self.dropout,
                attn_drop=self.attn_drop,
            )
            for i in range(depth)])

        self.input_pj = nn.Linear(in_features=self.input_dim, out_features=self.embed_dim)

        self.gtoken = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.sgtoken = nn.Parameter(torch.zeros(1, self.num_sg, self.embed_dim))

        self.head = nn.Linear(in_features=self.embed_dim, out_features=self.num_classes)

        trunc_normal_(self.gtoken, std=.02)
        trunc_normal_(self.sgtoken, std=.02)

        self.head_aux = nn.Linear(in_features=self.embed_dim, out_features=self.num_classes)



    def forward(self, x):
        b, n, d = x.shape  # torch.Size([b, 200, 200])
        x_pj = self.input_pj(x)
        gtoken = self.gtoken.expand(b, -1, -1)  # torch.Size([b, num_gtoken, embed_dim])
        sgtoken = self.sgtoken.expand(b, -1, -1)  # torch.Size([b, num_gtoken, embed_dim])
        x = torch.cat((gtoken, sgtoken, x_pj), dim=1)

        attn_node_list = []
        attn_sgn_list = []
        attn_gsg_list = []
        attn_g2node_list = []
        for i, blk in enumerate(self.blocks):
            x, attn_node, attn_sgn, attn_gsg, attn_g2node = blk(x) # [64, 8, 209, 209]
            attn_node_list.append(attn_node)
            attn_sgn_list.append(attn_sgn)
            attn_gsg_list.append(attn_gsg)
            attn_g2node_list.append(attn_g2node)

        x_g = x[:, 0, :]

        pred_g = self.head(x_g)

        return pred_g, attn_node_list, attn_sgn_list, attn_gsg_list, attn_g2node_list







