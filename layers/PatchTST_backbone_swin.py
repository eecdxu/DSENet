__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
from layers.groupattention import Aggregator

# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, c_kernel=5, c_stride=4, inner_dim=16, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., embedding_dropout=0.1,act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=False, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.res_attention = res_attention
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            # self.padding_patch_layer = nn.ReplicationPad1d((stride, 0)) 
            patch_num += 1
        
        self.num_words = math.floor((patch_len+2-c_kernel) / c_stride)+1 # W+2*padding-Kernel_size / stride +1
        self.d_model = inner_dim*self.num_words
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, inner_dim=inner_dim, c_kernel=c_kernel, c_stride=c_stride, d_model=self.d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, embedding_dropout=embedding_dropout,act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = self.d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        if self.res_attention:
            z_backbone, atten1, atten2, atten3 = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        else:
            z_backbone, atten1, atten2, atten3 = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]

        z, z_liner = self.head(z_backbone)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z, z_liner, atten1, atten2, atten3
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, 100*target_window)
            self.dropout = nn.Dropout(head_dropout)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(100*target_window, 2*target_window)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = x.view(x.size(0), -1)
            x_inter = self.linear(x)
            x = self.gelu(self.dropout(x_inter))
            x = self.linear2(x)
        return x, x_inter
        
        
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, inner_dim=16, c_kernel=3, c_stride=2, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., embedding_dropout=0.1,act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.res_attention = res_attention
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, inner_dim)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len
        self.proj = nn.Conv1d(1, inner_dim, kernel_size=c_kernel, padding=1, stride=c_stride)

        # Positional encoding
        # self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.num_words = int(d_model / inner_dim)
        # self.d_model = inner_dim*self.num_words
        self.W_pos = positional_encoding(pe, learn_pe, self.num_words, inner_dim)

        # Residual dropout
        self.dropout = nn.Dropout(embedding_dropout)

        # Encoder
        self.encoder1 = TSTEncoder1(pe, learn_pe, patch_num, q_len, self.num_words, inner_dim, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,embedding_dropout=embedding_dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn,
                                   )
        self.encoder2 = TSTEncoder2(pe, learn_pe, patch_num, q_len, self.num_words//2, inner_dim*2, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,embedding_dropout=embedding_dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn,
                                   )
        self.encoder3 = TSTEncoder3(pe, learn_pe, patch_num, q_len, self.num_words//4, inner_dim*4, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,embedding_dropout=embedding_dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn,
                                   )

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        B, n_vars = x.shape[0], x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        # x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        inner_tokens = torch.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2],1,x.shape[3]))      # u: [bs * nvars * patch_num x patch_len]
        inner_tokens = self.proj(inner_tokens)
        # outer_tokens = inner_tokens.reshape(B, n_vars, self.patch_num, -1)
        inner_tokens = inner_tokens.permute(0,2,1)
        inner_tokens = self.dropout(inner_tokens + self.W_pos)                                         # u: [bs * nvars * patch_num x d_model]

        # Encoder
        if self.res_attention:
            z, atten1, score1 = self.encoder1(inner_tokens, B, n_vars)                                                      # z: [bs * nvars x patch_num x d_model]
            z, atten2, score2 = self.encoder2(z, B, n_vars)
            z, atten3, score3 = self.encoder3(z, B, n_vars)
            z = torch.reshape(z, (B, n_vars, self.patch_num, -1))                # z: [bs x nvars x patch_num x d_model]
            z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
            return z, atten1, atten2, atten3
        else:
            z, atten1 = self.encoder1(inner_tokens, B, n_vars)                                                      # z: [bs * nvars x patch_num x d_model]
            z, atten2 = self.encoder2(z, B, n_vars)
            z, atten3 = self.encoder3(z, B, n_vars)
            z = torch.reshape(z, (B, n_vars, self.patch_num, -1))                # z: [bs x nvars x patch_num x d_model]
            z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
            return z, atten1, atten2, atten3 
            
class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, bias=False):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(2*dim, 4*dim, bias=bias),
                                nn.GELU(),
                                nn.Dropout(0.3),
                                nn.Linear(4*dim, 2*dim, bias=bias))
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L % 2 == 0

        x0 = x[:, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 2 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.proj(x)

        return x            
    
# Cell
class TSTEncoder1(nn.Module):
    def __init__(self, pe, learn_pe, patch_num, q_len, num_words, inner_dim, d_model, n_heads, d_k=None, d_v=None, d_ff=None, norm='BatchNorm', attn_dropout=0., dropout=0.,embedding_dropout=0.1,
                 activation='gelu', res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(pe, learn_pe, patch_num, q_len, num_words, inner_dim, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,embedding_dropout=embedding_dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention
        self.patch_merging = PatchMerging(dim=20)

    def forward(self, src:Tensor, B, n_vars, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores1 = None
        scores2 = None
        if self.res_attention:
            for mod in self.layers:
                output, scores1, scores2 = mod(output, B, n_vars, prev1=scores1, prev2=scores2, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                output = self.patch_merging(output)
            return output, scores1, scores2
        else:
            for mod in self.layers:
                output, atten = mod(output, B, n_vars, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                output = self.patch_merging(output)
            return output, atten

class TSTEncoder2(nn.Module):
    def __init__(self, pe, learn_pe, patch_num, q_len, num_words, inner_dim, d_model, n_heads, d_k=None, d_v=None, d_ff=None, norm='BatchNorm', attn_dropout=0., dropout=0.,embedding_dropout=0.1,
                 activation='gelu', res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(pe, learn_pe, patch_num, q_len, num_words, inner_dim, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,embedding_dropout=embedding_dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention
        self.patch_merging = PatchMerging(dim=40)

    def forward(self, src:Tensor, B, n_vars, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores1 = None
        scores2 = None
        if self.res_attention:
            for mod in self.layers:
                output, scores1, scores2 = mod(output, B, n_vars, prev1=scores1, prev2=scores2, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                output = self.patch_merging(output)
            return output, scores1, scores2
        else:
            for mod in self.layers:
                output, atten = mod(output, B, n_vars, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                output = self.patch_merging(output)
            return output, atten

class TSTEncoder3(nn.Module):
    def __init__(self, pe, learn_pe, patch_num, q_len, num_words, inner_dim, d_model, n_heads, d_k=None, d_v=None, d_ff=None, norm='BatchNorm', attn_dropout=0., dropout=0.,embedding_dropout=0.1,
                 activation='gelu', res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(pe, learn_pe, patch_num, q_len, num_words, inner_dim, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,embedding_dropout=embedding_dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention
        self.patch_merging = PatchMerging(dim=80)

    def forward(self, src:Tensor, B, n_vars, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores1 = None
        scores2 = None
        if self.res_attention:
            for mod in self.layers:
                output, scores1, scores2 = mod(output, B, n_vars, prev1=scores1, prev2=scores2, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output, scores1, scores2
        else:
            for mod in self.layers:
                output, atten = mod(output, B, n_vars, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                # output = self.patch_merging(output)
            return output, atten

class TSTEncoderLayer(nn.Module):
    def __init__(self, pe, learn_pe, patch_num, q_len, num_words, inner_dim, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,norm='BatchNorm', attn_dropout=0, dropout=0.,embedding_dropout=0.1,
                 bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        # print(num_words, d_model)
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.patch_num = patch_num
        self.inner_dim = inner_dim
        self.W_pos = positional_encoding(pe, learn_pe, self.patch_num, num_words*inner_dim)
        # d_k = d_model // n_heads if d_k is None else d_k
        # d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn1 = _MultiheadAttention(inner_dim, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        self.self_attn2 = _MultiheadAttention2(q_len, d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, embedding_dropout=embedding_dropout, res_attention=res_attention, act=activation, pe=pe, learn_pe=learn_pe)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

        self.proj_norm1 = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj_norm2 = nn.LayerNorm(d_model)
        # self.proj1 = nn.Sequential(nn.LayerNorm(d_model),
        #                         nn.Linear(d_model, d_ff, bias=bias),
        #                         get_activation_fn(activation),
        #                         nn.Dropout(dropout),
        #                         nn.Linear(d_ff, d_model, bias=bias),
        #                         nn.LayerNorm(d_model))
        self.proj1 = nn.Sequential(
                                nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                # nn.LayerNorm(d_ff),
                                nn.Linear(d_ff, d_model, bias=bias)
                                )


    def forward(self, src:Tensor, B, n_vars, prev1:Optional[Tensor]=None, prev2:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        
        outer = src.reshape(B * n_vars, self.patch_num, -1)
        # outer = self.dropout(outer + self.W_pos)    
        ## Multi-Head attention
        if self.res_attention:
            src2, attn1, scores1 = self.self_attn1(src, src, src, prev1, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            # outer = outer + self.proj_norm2(self.proj(self.proj_norm1(src2.reshape(B*n_vars, self.patch_num, -1)))) # B, N, C
            outer = outer + self.proj1(src2.reshape(B*n_vars, self.patch_num, -1)) # B, N, C
            src2_outer, attn2, scores2 = self.self_attn2(outer, outer, outer, prev2, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn1 = self.self_attn1(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            # outer = outer + self.proj_norm2(self.proj(self.proj_norm1(src2.reshape(B*n_vars, self.patch_num, -1)))) # B, N, C
            outer = outer + self.proj1(src2.reshape(B*n_vars, self.patch_num, -1)) # B, N, C
            src2_outer, attn2 = self.self_attn2(outer, outer, outer, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn2
        ## Add & Norm
        outer = outer + self.dropout_attn(src2_outer) # Add: residual connection with residual dropout
        if not self.pre_norm:
            outer = self.norm_attn(outer)

        # Feed-forward sublayer
        if self.pre_norm:
            outer = self.norm_ffn(outer)
        ## Position-wise Feed-Forward
        src2_outer = self.ff(outer)
        ## Add & Norm
        outer = outer + self.dropout_ffn(src2_outer) # Add: residual connection with residual dropout
        if not self.pre_norm:
            outer = self.norm_ffn(outer)

        if self.res_attention:
            return outer.reshape(B*n_vars*self.patch_num, -1, self.inner_dim), attn2, scores2
        else:
            return outer.reshape(B*n_vars*self.patch_num, -1, self.inner_dim), attn2


class _MultiheadAttention(nn.Module):
    def __init__(self, dim, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        self.dim = dim
        d_k = dim // n_heads if d_k is None else d_k
        d_v = dim // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(dim, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(dim, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(dim, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(dim, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, dim), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class _MultiheadAttention2(nn.Module):
    def __init__(self, q_len, dim, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., embedding_dropout=0.1,qkv_bias=True, lsa=False, act='gelu',
                 pe='zeros', learn_pe=True):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        self.dim = dim
        self.q_len = q_len
        d_k = dim // n_heads if d_k is None else d_k
        d_v = dim // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.aggregator = Aggregator(dim=dim, seg=5, act=act)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.W_Q = nn.Linear(dim, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(dim, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(dim, d_v * n_heads, bias=qkv_bias)
        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, dim//5*4)
        # Residual dropout
        self.dropout = nn.Dropout(embedding_dropout)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(dim, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, dim), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        # q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        # k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        # v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]
        q_s = self.W_Q(Q).view(bs, -1, self.dim)
        k_s = self.W_K(K).view(bs, -1, self.dim)
        v_s = self.W_V(V).view(bs, -1, self.dim)
        qkv = torch.cat([q_s, k_s, v_s], dim=0)
        qkv, x_agg0, C = self.aggregator(qkv, self.n_heads)
        q_s, k_s, v_s = qkv[0], qkv[1], qkv[2]
        q_s = self.dropout(q_s + self.W_pos)
        k_s = self.dropout(k_s + self.W_pos)
        v_s = self.dropout(v_s + self.W_pos)
        q_s, k_s, v_s = q_s.view(bs, self.q_len, self.n_heads, C//self.n_heads).transpose(1,2), k_s.view(bs, self.q_len, self.n_heads, C//self.n_heads).permute(0,2,3,1), v_s.view(bs, self.q_len, self.n_heads, C//self.n_heads).transpose(1,2)

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, C) # output: [bs x q_len x n_heads * d_v]
        output = torch.cat([output, x_agg0], dim=-1)
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights