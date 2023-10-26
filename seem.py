from typing import Optional

import imgviz
import numpy
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .registry import register_decoder
from ...utils import configurable
from ...modules import PositionEmbeddingSine
from PIL import Image
import numpy as np


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    color_map = imgviz.label_colormap()
    lbl_pil.putpalette(color_map.flatten())
    lbl_pil.save(save_path)


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads, idx):
        super(CrossAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * self.heads == self.embed_size), 'embed_size should be divided by heads'

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.head_dim * self.heads, embed_size)
        self._reset_parameters(idx)

    def forward(self, values, keys, query, mask):
        ###### b, n, d ########
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum('nqhd, nkhd -> nhqk', [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, torch.tensor(-1e10))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum('nhql, nlhd -> nqhd', [attention, values])

        out = out.reshape(N, query_len, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out

    def _reset_parameters(self, idx):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if idx == 6 or idx == 7 or idx == 8:
            nn.init.constant_(self.fc_out.bias.data, 0)  # nn.init.constant_()表示将偏差定义为常量0
            nn.init.constant_(self.fc_out.weight.data, 0)  # nn.init.constant_()表示将偏差定义为常量0


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MLP_1(nn.Module):
    def __init__(self, dim):
        super(MLP_1, self).__init__()
        self.fc1 = nn.Linear(dim, dim*4)
        self.fc2 = nn.Linear(dim*4, dim)
        self.act = nn.functional.gelu
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class InterTransBlock(nn.Module):
    def __init__(self, dim):
        super(InterTransBlock, self).__init__()
        self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
        self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.Attention = MultiScaleAtten(dim)
        self.FFN = MLP_1(dim)

    def forward(self, x):
        h = x  # (B, N, H)
        x = self.SlayerNorm_1(x)

        x = self.Attention(x)  # padding 到right_size
        x = h + x

        h = x
        x = self.SlayerNorm_2(x)

        x = self.FFN(x)
        x = h + x

        return x


class MultiScaleAtten(nn.Module):
    def __init__(self, dim):
        super(MultiScaleAtten, self).__init__()
        self.qkv_linear = nn.Linear(dim, dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.num_head = 8
        self.scale = (dim // self.num_head) ** 0.5

    def forward(self, x):
        B, num_blocks, _, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
        qkv = self.qkv_linear(x).reshape(B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(4,
                                                                                                                      0,
                                                                                                                      1,
                                                                                                                      2,
                                                                                                                      5,
                                                                                                                      3,
                                                                                                                      6).contiguous()  # (3, B, num_block, num_block, head, N, C)
        q, k, v = qkv[0], qkv[1], qkv[2]
        atten = q @ k.transpose(-1, -2).contiguous()
        atten = self.softmax(atten)
        atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
        atten_value = self.proj(atten_value)  # (B, num_block, num_block, N, C)
        return atten_value


class SpatialAwareTrans(nn.Module):
    def __init__(self, dim=256, num=1):
        super(SpatialAwareTrans, self).__init__()
        self.ini_win_size = 2
        self.channels = [512 + 10, 512 + 10, 1024 + 10]
        self.dim = dim
        self.depth = 3
        self.fc_module = nn.ModuleList()
        self.fc_rever_module = nn.ModuleList()
        self.num = num
        for i in range(self.depth):
            self.fc_module.append(nn.Linear(self.channels[i], self.dim))

        for i in range(self.depth):
            self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))

        self.group_attention = []
        for i in range(self.num):
            self.group_attention.append(InterTransBlock(dim))
        self.group_attention = nn.Sequential(*self.group_attention)
        self.split_list = [8 * 8, 8 * 8, 4 * 4]

    def forward(self, x):
        x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
        # Patch Matching
        for j, item in enumerate(x):
            B, H, W, C = item.shape
            win_size = self.ini_win_size ** (self.depth - j)
            item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4,
                                                                                                5).contiguous()
            item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
            x[j] = item
        x = tuple(x)
        x = torch.cat(x, dim=-2)  # (B, H // win, W // win, N, C)
        # Scale fusion
        for i in range(self.num):
            x = self.group_attention[i](x)  # (B, H // win_size, W // win_size, win_size*win_size, C)

        x = torch.split(x, self.split_list, dim=-2)
        x = list(x)
        # patch reversion
        for j, item in enumerate(x):
            B, num_blocks, _, N, C = item.shape
            win_size = self.ini_win_size ** (self.depth - j - 1)
            item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4,
                                                                                          5).contiguous().reshape(B,
                                                                                                                  num_blocks * win_size,
                                                                                                                  num_blocks * win_size,
                                                                                                                  C)
            item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
            x[j] = item
        return x


class MultiScaleMaskedTransformerDecoder(nn.Module):
    _version = 2

    @configurable
    def __init__(
            self,
            lang_encoder: nn.Module,
            in_channels,
            mask_classification=True,
            *,
            hidden_dim: int,
            dim_proj: int,
            num_queries: int,
            contxt_len: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            task_switch: dict,
            enforce_input_project: bool,
            max_spatial_len: int,
            attn_arch: dict,
    ):

        super().__init__()
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.layer_norm = nn.LayerNorm(1024)
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.contxt_len = contxt_len

        self.num_feature_levels = 3

        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.final_predict = nn.Sequential(
            nn.BatchNorm2d(512 + 20),
            nn.Conv2d(512 + 20, 10, 3, 1, 1))

        self.final_fuse = nn.Sequential(
            nn.Conv2d(1300, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.hidden_dim_list = [512, 512, 512, 512, 1024, 1024, 512, 512, 1024]
        for idx in range(self.num_layers):
            self.transformer_cross_attention_layers.append(
                CrossAttention(
                    embed_size=self.hidden_dim_list[idx],
                    heads=nheads,
                    idx=idx
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=self.hidden_dim_list[idx],
                    dim_feedforward=self.hidden_dim_list[idx] * 2,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.channel_reduction = nn.ModuleList()
        self.channel_reduction.append(
            nn.Sequential(
                nn.Conv2d(1024, 512, 3, 1),
                nn.BatchNorm2d(512)
            ))
        self.channel_reduction.append(
            nn.Sequential(
                nn.Conv2d(1024, 512, 3, 1),
                nn.BatchNorm2d(512)
            ))

        self.spatial_list = [64, 64, 32]
        self.channel_list = [512, 512, 1024]
        self.skip_connection = nn.ModuleList()

        self.skip_connection.append(
            nn.Sequential(
                nn.Conv2d(self.channel_list[0] + self.channel_list[1], self.channel_list[1], 3, 1, 1),
                nn.BatchNorm2d(self.channel_list[1])
            ))
        self.skip_connection.append(
            nn.Sequential(
                nn.Conv2d(self.channel_list[1] + self.channel_list[2], self.channel_list[2], 3, 1, 1),
                nn.BatchNorm2d(self.channel_list[2])
            )
        )

        self.inter_scale = SpatialAwareTrans(256, 1)

    @classmethod
    def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
        ret = {}

        ret["lang_encoder"] = lang_encoder
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
        ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
        ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
        ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
        ret["nheads"] = dec_cfg['NHEADS']
        ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']

        assert dec_cfg['DEC_LAYERS'] >= 1
        ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
        ret["pre_norm"] = dec_cfg['PRE_NORM']
        ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
        ret["mask_dim"] = enc_cfg['MASK_DIM']
        ret["task_switch"] = extra['task_switch']
        ret["max_spatial_len"] = dec_cfg['MAX_SPATIAL_LEN']
        ret["attn_arch"] = cfg['ATTENTION_ARCH']

        return ret

    def forward(self, ref_information, query_information, extra={}, task='seg'):
        query_multi_scale = query_information
        ref_multiscale_feature, ref_mask = ref_information

        out_predict_list = []
        ########## reference mask进行缩放，在三个尺度 #################
        bs, c, h, w = ref_mask.tensor.shape
        ref_mask_list = []
        for i in range(self.num_feature_levels):
            ref_mask_si = F.interpolate(ref_mask.tensor, (self.spatial_list[i], self.spatial_list[i]), mode='nearest')
            ref_mask_list.append(ref_mask_si.reshape(bs, c, -1).permute(0, 2, 1))

        ######### 特征插值到128, 64, 32 ##############################
        query_stage_list = []
        ref_stage_list = []

        for i in range(self.num_feature_levels):
            if i != 2:
                query_multi_si = F.interpolate(self.channel_reduction[i](query_multi_scale[i]),
                                               (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
                                               mode='bilinear')
            else:
                query_multi_si = query_multi_scale[i]

            query_stage_list.append(query_multi_si)

        for i in range(self.num_feature_levels):
            if i != 2:
                ref_multi_si = F.interpolate(self.channel_reduction[i](ref_multiscale_feature[i]),
                                             (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
                                             mode='bilinear')
            else:
                ref_multi_si = ref_multiscale_feature[i]

            ref_stage_list.append(ref_multi_si)

        ######### 经过cross-attention结构进行特征增强 ##################3
        for level_index in range(self.num_feature_levels):
            if level_index != 0:
                pre_feature = F.interpolate(pre_feature, (query_stage_list[level_index].shape[-1], query_stage_list[level_index].shape[-2]), align_corners=True,
                                             mode='bilinear')
                query_stage_list[level_index] = torch.cat((query_stage_list[level_index], pre_feature), dim=1)
                query_stage_list[level_index] = self.skip_connection[level_index - 1](query_stage_list[level_index])

            for j in range(2):
                src_mask_features = query_stage_list[level_index]
                spatial_tokens = ref_stage_list[level_index]
                bs, d, _, _ = src_mask_features.shape
                src_mask_features = src_mask_features.view(bs, d, -1).permute(0, 2, 1)
                spatial_tokens = spatial_tokens.view(bs, d, -1).permute(0, 2, 1)

                output_pos = self.transformer_cross_attention_layers[level_index * 2 + j](
                    spatial_tokens, spatial_tokens, src_mask_features,
                    mask=None
                )
                y = self.transformer_ffn_layers[level_index * 2 + j](output_pos.permute(1, 0, 2)).permute(1, 0, 2)  ### b, n, d
                query_stage_list[level_index] = y.reshape(bs, self.spatial_list[level_index],
                                                       self.spatial_list[level_index], d).permute(0, 3, 1, 2)

            pre_feature = query_stage_list[level_index]

        #############  增强后的特征各自进行检索过程  ############################################
        for i in range(len(query_stage_list)):
            src_mask_features = query_stage_list[i]
            spatial_tokens = ref_stage_list[i]

            bs, d, _, _ = src_mask_features.shape
            src_mask_features = src_mask_features.view(bs, d, -1).permute(0, 2, 1)
            spatial_tokens = spatial_tokens.view(bs, d, -1).permute(0, 2, 1)

            src_norm = src_mask_features / (torch.norm(src_mask_features, dim=-1, keepdim=True) + 1e-12)
            spatial_norm = spatial_tokens / (torch.norm(spatial_tokens, dim=-1, keepdim=True) + 1e-12)

            avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
            avg_atten = avg_atten.softmax(dim=-1)

            out_predict = avg_atten @ ref_mask_list[i]
            out_predict_list.append(out_predict)

        results = self.forward_prediction_heads(query_stage_list, out_predict_list)
        return results

    def forward_prediction_heads(self, src, out_predict_list):
        '''

        :param src: [(b, 512, 64, 64), (b, 512, 64, 64), (b, 1024, 32, 32)]
        :param out_predict_list: [(b, 10, 64, 64), (b, 10, 64, 64), (b, 10, 32, 32)]
        :return:
        '''
        bs, dim1, h1, w1 = src[0].shape
        bs, dim2, h2, w2 = src[1].shape
        bs, dim3, h3, w3 = src[2].shape

        feature_1 = torch.cat((src[0], out_predict_list[0].reshape(bs, -1, h1, w1)), dim=1)
        feature_2 = torch.cat((src[1], out_predict_list[1].reshape(bs, -1, h2, w2)), dim=1)
        feature_3 = torch.cat((src[2], out_predict_list[2].reshape(bs, -1, h3, w3)), dim=1)

        f_1_aug, f_2_aug, f_3_aug = self.inter_scale((feature_1, feature_2, feature_3))

        f_3_aug = F.interpolate(f_3_aug, (f_1_aug.shape[-1], f_1_aug.shape[-2]), mode='bilinear', align_corners=True)
        out_predict_3 = F.interpolate(out_predict_list[2], (f_1_aug.shape[-1], f_1_aug.shape[-2]), mode='bilinear',
                                      align_corners=True)

        final_fuse = self.final_fuse(torch.cat((f_1_aug, f_3_aug), dim=1))
        out_predict = 1 / 2 * (out_predict_list[0] + out_predict_3)

        outputs_mask = self.final_predict(
            torch.cat((final_fuse, out_predict.reshape(bs, 32, 32, 10).permute(0, 3, 1, 2)), dim=1))

        results = {
            "predictions_mask": outputs_mask
        }
        return results


@register_decoder
def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
    return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)

# #######################################current version ####################################################
# from typing import Optional
#
# import imgviz
# import numpy
# import torch
# from torch import nn, Tensor
# from torch.nn import functional as F
#
# from timm.models.layers import trunc_normal_
# from detectron2.layers import Conv2d
# import fvcore.nn.weight_init as weight_init
#
# from .utils.utils import rand_sample, prepare_features
# from .utils.attn import MultiheadAttention
# from .utils.attention_data_struct import AttentionDataStruct
# from .registry import register_decoder
# from ...utils import configurable
# from ...modules import PositionEmbeddingSine
# from einops import rearrange
# from PIL import Image
# import numpy as np
#
#
# def save_colored_mask(mask, save_path):
#     lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
#     color_map = imgviz.label_colormap()
#     lbl_pil.putpalette(color_map.flatten())
#     lbl_pil.save(save_path)
#
#
# class FFNLayer(nn.Module):
#
#     def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
#                  activation="relu", normalize_before=False):
#         super().__init__()
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#
#         self.norm = nn.LayerNorm(d_model)
#
#         self.activation = _get_activation_fn(activation)
#         self.normalize_before = normalize_before
#
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         return tensor if pos is None else tensor + pos
#
#     def forward_post(self, tgt):
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout(tgt2)
#         tgt = self.norm(tgt)
#         return tgt
#
#     def forward_pre(self, tgt):
#         tgt2 = self.norm(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
#         tgt = tgt + self.dropout(tgt2)
#         return tgt
#
#     def forward(self, tgt):
#         if self.normalize_before:
#             return self.forward_pre(tgt)
#         return self.forward_post(tgt)
#
#
# class CrossAttention(nn.Module):
#     def __init__(self, embed_size, heads, idx):
#         super(CrossAttention, self).__init__()
#
#         self.embed_size = embed_size
#         self.heads = heads
#         self.head_dim = embed_size // heads
#
#         assert (self.head_dim * self.heads == self.embed_size), 'embed_size should be divided by heads'
#
#         self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.fc_out = nn.Linear(self.head_dim * self.heads, embed_size)
#         self._reset_parameters(idx)
#
#     def forward(self, values, keys, query, mask):
#         ###### b, n, d ########
#
#         N = query.shape[0]
#         value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
#
#         values = values.reshape(N, value_len, self.heads, self.head_dim)
#         keys = keys.reshape(N, key_len, self.heads, self.head_dim)
#         queries = query.reshape(N, query_len, self.heads, self.head_dim)
#
#         values = self.values(values)
#         keys = self.keys(keys)
#         queries = self.queries(queries)
#
#         energy = torch.einsum('nqhd, nkhd -> nhqk', [queries, keys])
#
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, torch.tensor(-1e10))
#
#         attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
#
#         out = torch.einsum('nhql, nlhd -> nqhd', [attention, values])
#
#         out = out.reshape(N, query_len, self.heads * self.head_dim)
#
#         out = self.fc_out(out)
#         return out
#
#     def _reset_parameters(self, idx):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#         if idx == 6 or idx == 7 or idx == 8:
#             nn.init.constant_(self.fc_out.bias.data, 0)  # nn.init.constant_()表示将偏差定义为常量0
#             nn.init.constant_(self.fc_out.weight.data, 0)  # nn.init.constant_()表示将偏差定义为常量0
#
#
# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
#
#
# class MLP(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""
#
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
#
#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x
#
#
# class MultiScaleMaskedTransformerDecoder(nn.Module):
#     _version = 2
#
#     @configurable
#     def __init__(
#             self,
#             lang_encoder: nn.Module,
#             in_channels,
#             mask_classification=True,
#             *,
#             hidden_dim: int,
#             dim_proj: int,
#             num_queries: int,
#             contxt_len: int,
#             nheads: int,
#             dim_feedforward: int,
#             dec_layers: int,
#             pre_norm: bool,
#             mask_dim: int,
#             task_switch: dict,
#             enforce_input_project: bool,
#             max_spatial_len: int,
#             attn_arch: dict,
#     ):
#
#         super().__init__()
#         assert mask_classification, "Only support mask classification model"
#         self.mask_classification = mask_classification
#
#         N_steps = hidden_dim // 2
#         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
#         self.layer_norm = nn.LayerNorm(1024)
#         # define Transformer decoder here
#         self.num_heads = nheads
#         self.num_layers = dec_layers
#         self.contxt_len = contxt_len
#
#         self.num_feature_levels = 3
#
#         self.transformer_cross_attention_layers = nn.ModuleList()
#         self.transformer_ffn_layers = nn.ModuleList()
#
#         self.final_predict = nn.Sequential(
#             nn.BatchNorm2d(1024 + 5),
#             nn.Conv2d(1024 + 5, 5, 3, 1, 1))
#
#         self.final_fuse = nn.Sequential(
#             nn.Conv2d(1024 * 3, 1024, 3, 1, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.ReLU())
#
#         for idx in range(self.num_layers):
#             self.transformer_cross_attention_layers.append(
#                 CrossAttention(
#                     embed_size=hidden_dim,
#                     heads=nheads,
#                     idx=idx
#                 )
#             )
#
#             self.transformer_ffn_layers.append(
#                 FFNLayer(
#                     d_model=hidden_dim,
#                     dim_feedforward=dim_feedforward,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )
#
#     @classmethod
#     def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
#         ret = {}
#
#         ret["lang_encoder"] = lang_encoder
#         ret["in_channels"] = in_channels
#         ret["mask_classification"] = mask_classification
#
#         enc_cfg = cfg['MODEL']['ENCODER']
#         dec_cfg = cfg['MODEL']['DECODER']
#
#         ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
#         ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
#         ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
#         ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
#
#         # Transformer parameters:
#         ret["nheads"] = dec_cfg['NHEADS']
#         ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']
#
#         assert dec_cfg['DEC_LAYERS'] >= 1
#         ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
#         ret["pre_norm"] = dec_cfg['PRE_NORM']
#         ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
#         ret["mask_dim"] = enc_cfg['MASK_DIM']
#         ret["task_switch"] = extra['task_switch']
#         ret["max_spatial_len"] = dec_cfg['MAX_SPATIAL_LEN']
#
#         # attn data struct
#         ret["attn_arch"] = cfg['ATTENTION_ARCH']
#
#         return ret
#
#     def forward(self, ref_information, query_information, extra={}, task='seg'):
#         query_multi_scale = query_information
#         ref_multiscale_feature, ref_mask = ref_information
#
#         support_list = []
#         src = []
#         out_predict_list = []
#         src_copy = []
#
#         bs, c, h, w = ref_mask.tensor.shape
#         ref_mask_scale = F.interpolate(ref_mask.tensor, (32, 32), mode='nearest')
#         ref_mask_scale = ref_mask_scale.reshape(bs, c, -1).permute(0, 2, 1)
#
#         for i in range(len(query_multi_scale)):
#             ref_feature = ref_multiscale_feature[i]
#             bs, d, _, _ = ref_feature.shape
#             ref_feature = ref_feature.view(bs, d, -1).permute(0, 2, 1)  ### bs, n, d
#
#             support_sets = ref_feature
#             support_list.append(support_sets)
#
#             query_feature = query_multi_scale[i].view(bs, d, -1).permute(0, 2, 1)
#             src.append(query_feature)
#             src_copy.append(query_feature.clone())  ### -0.2233
#
#         for i in range(self.num_layers):
#             level_index = i % self.num_feature_levels
#             src_mask_features = src[level_index]  ### b, n, d
#             spatial_tokens = support_list[level_index]  ### b, n, d
#             output_pos = self.transformer_cross_attention_layers[i](
#                 spatial_tokens, spatial_tokens, src_mask_features,
#                 mask=None
#             )
#
#             y = self.transformer_ffn_layers[i](output_pos.permute(1, 0, 2)).permute(1, 0, 2)
#             src[level_index] = y
#
#         for i in range(len(src)):
#             src_mask_features = src[i]
#             spatial_tokens = support_list[i]
#
#             src_mask_features = self.layer_norm(src_mask_features + src_copy[i])
#
#             src_norm = src_mask_features / (torch.norm(src_mask_features, dim=-1, keepdim=True) + 1e-12)
#             spatial_norm = spatial_tokens / (torch.norm(spatial_tokens, dim=-1, keepdim=True) + 1e-12)
#
#             avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
#             avg_atten = avg_atten.softmax(dim=-1)
#
#             out_predict = avg_atten @ ref_mask_scale
#             out_predict_list.append(out_predict)
#
#         results = self.forward_prediction_heads(src, out_predict_list)
#         return results
#
#     def forward_prediction_heads(self, src, out_predict_list):
#         bs, num_1, dim = src[0].shape
#         _, num_2, _ = src[1].shape
#         _, num_3, _ = src[2].shape
#
#         feature_1 = src[0].reshape(bs, int(numpy.sqrt(num_1)),
#                                    int(numpy.sqrt(num_1)), dim).permute(0, 3, 1, 2)  ####(32, 32)
#         feature_2 = src[1].reshape(bs, int(numpy.sqrt(num_2)),
#                                    int(numpy.sqrt(num_2)), dim).permute(0, 3, 1, 2)  ####(32, 32)
#         feature_3 = src[2].reshape(bs, int(numpy.sqrt(num_3)),
#                                    int(numpy.sqrt(num_3)), dim).permute(0, 3, 1, 2)  ####(32, 32)
#
#         final_fuse = self.final_fuse(torch.cat((feature_1, feature_2, feature_3), dim=1))
#
#         out_predict = 1 / 3 * (out_predict_list[0] + out_predict_list[1] + out_predict_list[2])
#
#         outputs_mask = self.final_predict(
#             torch.cat((final_fuse, out_predict.reshape(bs, 32, 32, 5).permute(0, 3, 1, 2)), dim=1))
#
#         results = {
#             "predictions_mask": outputs_mask
#         }
#         return results
#
#
# @register_decoder
# def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
#     return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
#
# ######################################################## previous version ###########################################################
# # from typing import Optional
#
# # import imgviz
# # import numpy
# # import torch
# # from torch import nn, Tensor
# # from torch.nn import functional as F
#
# # from timm.models.layers import trunc_normal_
# # from detectron2.layers import Conv2d
# # import fvcore.nn.weight_init as weight_init
#
# # from .utils.utils import rand_sample, prepare_features
# # from .utils.attn import MultiheadAttention
# # from .utils.attention_data_struct import AttentionDataStruct
# # from .registry import register_decoder
# # from ...utils import configurable
# # from ...modules import PositionEmbeddingSine
# # from einops import rearrange
# # from PIL import Image
# # import numpy as np
#
# # def save_colored_mask(mask, save_path):
# #     lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
# #     color_map = imgviz.label_colormap()
# #     lbl_pil.putpalette(color_map.flatten())
# #     lbl_pil.save(save_path)
# # # class SelfAttentionLayer(nn.Module):
#
# # #     def __init__(self, d_model, nhead, dropout=0.0,
# # #                  activation="relu", normalize_before=False):
# # #         super().__init__()
# # #         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
#
# # #         self.norm = nn.LayerNorm(d_model)
# # #         self.dropout = nn.Dropout(dropout)
#
# # #         self.activation = _get_activation_fn(activation)
# # #         self.normalize_before = normalize_before
#
# # #         self._reset_parameters()
#
# # #     def _reset_parameters(self):
# # #         for p in self.parameters():
# # #             if p.dim() > 1:
# # #                 nn.init.xavier_uniform_(p)
#
# # #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# # #         return tensor if pos is None else tensor + pos
#
# # #     def forward_post(self, tgt,
# # #                      tgt_mask: Optional[Tensor] = None,
# # #                      tgt_key_padding_mask: Optional[Tensor] = None,
# # #                      query_pos: Optional[Tensor] = None):
# # #         q = k = self.with_pos_embed(tgt, query_pos)
# # #         tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
# # #                               key_padding_mask=tgt_key_padding_mask)[0]
# # #         tgt = tgt + self.dropout(tgt2)
# # #         tgt = self.norm(tgt)
#
# # #         return tgt
#
# # #     def forward_pre(self, tgt,
# # #                     tgt_mask: Optional[Tensor] = None,
# # #                     tgt_key_padding_mask: Optional[Tensor] = None,
# # #                     query_pos: Optional[Tensor] = None):
# # #         tgt2 = self.norm(tgt)
# # #         q = k = self.with_pos_embed(tgt2, query_pos)
# # #         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
# # #                               key_padding_mask=tgt_key_padding_mask)[0]
# # #         tgt = tgt + self.dropout(tgt2)
#
# # #         return tgt
#
# # #     def forward(self, tgt,
# # #                 tgt_mask: Optional[Tensor] = None,
# # #                 tgt_key_padding_mask: Optional[Tensor] = None,
# # #                 query_pos: Optional[Tensor] = None):
# # #         if self.normalize_before:
# # #             return self.forward_pre(tgt, tgt_mask,
# # #                                     tgt_key_padding_mask, query_pos)
# # #         return self.forward_post(tgt, tgt_mask,
# # #                                  tgt_key_padding_mask, query_pos)
#
#
# # # class CrossAttentionLayer(nn.Module):
#
# # #     def __init__(self, d_model, nhead, dropout=0.0,
# # #                  activation="relu", normalize_before=False):
# # #         super().__init__()
# # #         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#
# # #         self.norm = nn.LayerNorm(d_model)
# # #         self.dropout = nn.Dropout(dropout)
#
# # #         self.activation = _get_activation_fn(activation)
# # #         self.normalize_before = normalize_before
#
# # #         self._reset_parameters()
#
# # #     def _reset_parameters(self):
# # #         for p in self.parameters():
# # #             if p.dim() > 1:
# # #                 nn.init.xavier_uniform_(p)
# # #         # for p in self.parameters():
# # #         #     if p.dim() > 1:
# # #         #         nn.init.constant_(p, 0.)
#
# # #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# # #         return tensor if pos is None else tensor + pos
#
# # #     def forward_post(self, tgt, memory,
# # #                      memory_mask: Optional[Tensor] = None,
# # #                      memory_key_padding_mask: Optional[Tensor] = None,
# # #                      pos: Optional[Tensor] = None,
# # #                      query_pos: Optional[Tensor] = None):
# # #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
# # #                                              key=self.with_pos_embed(memory, pos),
# # #                                              value=memory, attn_mask=memory_mask,
# # #                                              key_padding_mask=memory_key_padding_mask)
# # #         tgt = tgt + self.dropout(tgt2)
# # #         tgt = self.norm(tgt)
# # #         return tgt, avg_attn
#
# # #     def forward_pre(self, tgt, memory,
# # #                     memory_mask: Optional[Tensor] = None,
# # #                     memory_key_padding_mask: Optional[Tensor] = None,
# # #                     pos: Optional[Tensor] = None,
# # #                     query_pos: Optional[Tensor] = None):
# # #         tgt2 = self.norm(tgt)
# # #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
# # #                                              key=self.with_pos_embed(memory, pos),
# # #                                              value=memory, attn_mask=memory_mask,
# # #                                              key_padding_mask=memory_key_padding_mask)
# # #         tgt = tgt + self.dropout(tgt2)
#
# # #         return tgt, avg_attn
#
# # #     def forward(self, tgt, memory,
# # #                 memory_mask: Optional[Tensor] = None,
# # #                 memory_key_padding_mask: Optional[Tensor] = None,
# # #                 pos: Optional[Tensor] = None,
# # #                 query_pos: Optional[Tensor] = None):
# # #         if self.normalize_before:
# # #             return self.forward_pre(tgt, memory, memory_mask,
# # #                                     memory_key_padding_mask, pos, query_pos)
# # #         return self.forward_post(tgt, memory, memory_mask,
# # #                                  memory_key_padding_mask, pos, query_pos)
#
#
# # class FFNLayer(nn.Module):
#
# #     def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.linear1 = nn.Linear(d_model, dim_feedforward)
# #         self.dropout = nn.Dropout(dropout)
# #         self.linear2 = nn.Linear(dim_feedforward, d_model)
#
# #         self.norm = nn.LayerNorm(d_model)
#
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
#
# #         self._reset_parameters()
#
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
#
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
#
# #     def forward_post(self, tgt):
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt
#
# #     def forward_pre(self, tgt):
# #         tgt2 = self.norm(tgt)
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
# #         tgt = tgt + self.dropout(tgt2)
# #         return tgt
#
# #     def forward(self, tgt):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt)
# #         return self.forward_post(tgt)
#
#
# # class CrossAttention(nn.Module):
# #     def __init__(self, embed_size, heads):
# #         super(CrossAttention, self).__init__()
#
# #         self.embed_size = embed_size
# #         self.heads = heads
# #         self.head_dim = embed_size // heads
#
# #         assert (self.head_dim * self.heads == self.embed_size), 'embed_size should be divided by heads'
#
# #         self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
# #         self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
# #         self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
# #         self.fc_out = nn.Linear(self.head_dim * self.heads, embed_size)
#
# #     def forward(self, values, keys, query, mask):
#
# #         N = query.shape[0]
# #         value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
#
# #         values = values.reshape(N, value_len, self.heads, self.head_dim)
# #         keys = keys.reshape(N, key_len, self.heads, self.head_dim)
# #         queries = query.reshape(N, query_len, self.heads, self.head_dim)
#
# #         values = self.values(values)
# #         keys = self.keys(keys)
# #         queries = self.queries(queries)
#
# #         energy = torch.einsum('nqhd, nkhd -> nhqk', [queries, keys])
#
# #         if mask is not None:
# #             energy = energy.masked_fill(mask==0, torch.tensor(-1e10))
#
# #         attention = torch.softmax(energy/(self.embed_size**(1/2)), dim=3)
#
# #         out = torch.einsum('nhql, nlhd -> nqhd', [attention, values])
#
# #         out = out.reshape(N, query_len, self.heads*self.head_dim)
#
# #         out = self.fc_out(out)
# #         return out
#
# # # class TransformerBlock(nn.Module):
# # #     def __init__(self, embed_size, heads, dropout, forward_expansion):
# # #         super(TransformerBlock, self).__init__()
# # #         self.attention = crossattention(embed_size, heads)
# # #         self.norm1 = nn.LayerNorm(embed_size)
# # #         self.norm2 = nn.LayerNorm(embed_size)
# # #         self.feed_forward = nn.Sequential(
# # #             nn.Linear(embed_size, embed_size * forward_expansion),
# # #             nn.ReLU(),
# # #             nn.Linear(embed_size * forward_expansion, embed_size)
# # #         )
# # #         self.dropout = nn.Dropout(dropout)
#
# # #     def forward(self, value, key, query, mask):
#
# # #         attention = self.attention(value, key, query, mask)
# # #         x = query + attention
# # #         x = self.norm1(x)
# # #         x = self.dropout(x)
#
# # #         ffn = self.feed_forward(x)
# # #         forward = ffn + x
# # #         out = self.dropout(self.norm2(forward))
# # #         return out
#
# # def _get_activation_fn(activation):
# #     """Return an activation function given a string"""
# #     if activation == "relu":
# #         return F.relu
# #     if activation == "gelu":
# #         return F.gelu
# #     if activation == "glu":
# #         return F.glu
# #     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
#
#
# # class MLP(nn.Module):
# #     """ Very simple multi-layer perceptron (also called FFN)"""
#
# #     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
# #         super().__init__()
# #         self.num_layers = num_layers
# #         h = [hidden_dim] * (num_layers - 1)
# #         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
#
# #     def forward(self, x):
# #         for i, layer in enumerate(self.layers):
# #             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
# #         return x
#
#
# # class MultiScaleMaskedTransformerDecoder(nn.Module):
# #     _version = 2
#
# #     @configurable
# #     def __init__(
# #             self,
# #             lang_encoder: nn.Module,
# #             in_channels,
# #             mask_classification=True,
# #             *,
# #             hidden_dim: int,
# #             dim_proj: int,
# #             num_queries: int,
# #             contxt_len: int,
# #             nheads: int,
# #             dim_feedforward: int,
# #             dec_layers: int,
# #             pre_norm: bool,
# #             mask_dim: int,
# #             task_switch: dict,
# #             enforce_input_project: bool,
# #             max_spatial_len: int,
# #             attn_arch: dict,
# #     ):
#
# #         super().__init__()
# #         assert mask_classification, "Only support mask classification model"
# #         self.mask_classification = mask_classification
#
# #         N_steps = hidden_dim // 2
# #         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
# #         self.layer_norm = nn.LayerNorm(1024)
# #         # define Transformer decoder here
# #         self.num_heads = nheads
# #         self.num_layers = dec_layers
# #         self.contxt_len = contxt_len
#
# #         self.transformer_self_attention_layers = nn.ModuleList()
# #         self.transformer_cross_attention_layers = nn.ModuleList()
# #         self.transformer_ffn_layers = nn.ModuleList()
#
# #         self.linear_list = nn.ModuleList()
# #         self.src_linear_list = nn.ModuleList()
# #         for _ in range(3):
# #             self.linear_list.append(
# #             nn.Sequential(nn.Linear(1024, 512))
# #         )
#
# #         for _ in range(3):
# #             self.src_linear_list.append(
# #             nn.Sequential(nn.Linear(1024, 512))
# #         )
#
#
# #         self.final_predict = nn.Sequential(
# #             nn.BatchNorm2d(1024+5),
# #             nn.Conv2d(1024+5, 5, 3, 1, 1))
#
# #         self.final_fuse = nn.Sequential(
# #             nn.Conv2d(1024 * 3, 1024, 3, 1, padding=1),
# #             nn.BatchNorm2d(1024),
# #             nn.ReLU())
#
# #         for _ in range(self.num_layers):
#
# #             self.transformer_cross_attention_layers.append(
# #                 CrossAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
#
# #             self.transformer_ffn_layers.append(
# #                 FFNLayer(
# #                     d_model=hidden_dim,
# #                     dim_feedforward=dim_feedforward,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
#
# #         self.decoder_norm = nn.LayerNorm(hidden_dim)
# #         self.num_queries = num_queries
# #         self.query_feat = nn.Embedding(num_queries, hidden_dim)
#
# #         self.query_embed = nn.Embedding(num_queries, hidden_dim)
# #         self.pn_indicator = nn.Embedding(2, hidden_dim)
#
# #         self.num_feature_levels = 3
# #         self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
# #         self.input_proj = nn.ModuleList()
#
# #         for _ in range(self.num_feature_levels):
# #             if in_channels != hidden_dim or enforce_input_project:
# #                 self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
# #                 weight_init.c2_xavier_fill(self.input_proj[-1])
# #             else:
# #                 self.input_proj.append(nn.Sequential())
#
# #         self.task_switch = task_switch
# #         self.query_index = {}
#
# #         # output FFNs
# #         self.lang_encoder = lang_encoder
# #         if self.task_switch['mask']:
# #             self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
#
# #         self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
# #         trunc_normal_(self.class_embed, std=.02)
#
# #         if task_switch['spatial']:
# #             self.mask_sptial_embed = nn.ParameterList(
# #                 [nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(3)])
# #             trunc_normal_(self.mask_sptial_embed[0], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[1], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[2], std=.02)
#
# #             self.max_spatial_len = max_spatial_len
#
# #             num_spatial_memories = attn_arch['SPATIAL_MEMORIES']
# #             self.spatial_embed = nn.Embedding(num_spatial_memories, hidden_dim)
# #             self.spatial_featured = nn.Embedding(num_spatial_memories, hidden_dim)
#
# #         # build AttentionDataStruct
# #         attn_arch['NUM_LAYERS'] = self.num_layers
# #         self.attention_data = AttentionDataStruct(attn_arch, task_switch)
#
#
# #     @classmethod
# #     def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
# #         ret = {}
#
# #         ret["lang_encoder"] = lang_encoder
# #         ret["in_channels"] = in_channels
# #         ret["mask_classification"] = mask_classification
#
# #         enc_cfg = cfg['MODEL']['ENCODER']
# #         dec_cfg = cfg['MODEL']['DECODER']
#
# #         ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
# #         ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
# #         ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
# #         ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
#
# #         # Transformer parameters:
# #         ret["nheads"] = dec_cfg['NHEADS']
# #         ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']
#
# #         assert dec_cfg['DEC_LAYERS'] >= 1
# #         ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
# #         ret["pre_norm"] = dec_cfg['PRE_NORM']
# #         ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
# #         ret["mask_dim"] = enc_cfg['MASK_DIM']
# #         ret["task_switch"] = extra['task_switch']
# #         ret["max_spatial_len"] = dec_cfg['MAX_SPATIAL_LEN']
#
# #         # attn data struct
# #         ret["attn_arch"] = cfg['ATTENTION_ARCH']
#
# #         return ret
#
# #     def forward(self, ref_information, query_information, extra={}, task='seg'):
# #         query_multi_scale = query_information
# #         ref_multiscale_feature, ref_mask = ref_information
#
# #         assert len(query_multi_scale) == self.num_feature_levels;
# #         spatial_extra_flag = 'spatial_query_pos_mask' in extra.keys() or task == 'refimg'
# #         grounding_extra_flag = 'grounding_tokens' in extra.keys()
# #         visual_extra_flag = 'visual_query_pos' in extra.keys()
# #         audio_extra_flag = 'audio_tokens' in extra.keys()
# #         spatial_memory_flag = 'prev_mask' in extra.keys()
# #         flags = {"spatial": spatial_extra_flag, "grounding": grounding_extra_flag,
# #                          "memories_spatial": spatial_memory_flag, "visual": visual_extra_flag, "audio": audio_extra_flag}
# #         self.attention_data.reset(flags, task, extra)
#
# #         support_list = []
# #         src = []
# #         out_predict_list = []
# #         src_copy = []
#
# #         bs, c, h, w = ref_mask.tensor.shape
# #         ref_mask_scale = F.interpolate(ref_mask.tensor, (32, 32), mode='nearest')
# #         ref_mask_scale = ref_mask_scale.reshape(bs, c, -1).permute(0, 2, 1)
#
# #         # bs, d, h, w = query_multi_scale[0].shape
# #         # print(query_multi_scale[0].shape)
# #         # print(ref_mask.tensor.shape)
#
# #         # query_feature = query_multi_scale[0].view(bs, d, -1).permute(0, 2, 1)
# #         # ref_feature = ref_multiscale_feature[0].view(bs, d, -1).permute(0, 2, 1)
#
# #         # import math
# #         # query_feature = query_feature
# #         # atten = (query_feature @ ref_feature.transpose(-1, -2)).softmax(dim=-1)
#
# #         # # q_norm = (query_feature / torch.norm(query_feature, dim=-1, keepdim=True))
# #         # # k_norm = (ref_feature / torch.norm(ref_feature, dim=-1, keepdim=True))
# #         # # atten = torch.mul(q_norm, k_norm.transpose(-1, -2))
#
# #         # print(atten.max())
# #         # print(atten.min())
# #         # # atten = query_feature @ ref_feature
# #         # final_mask = atten @ ref_mask_scale
# #         # final_mask = final_mask.reshape(bs, 32, 32, 5).permute(0, 3, 1, 2)
#
# #         #         # for i in range(final_mask[0].shape[0]):
# #         #         #     view_img = final_mask[0][i].cpu().numpy()
# #         #         #     # print(view_img.sum())
# #         #         #     view_img = Image.fromarray(np.uint8(view_img))
# #         #         #     view_img.save(str(i)+'.png')
#
# #         # results = {
# #         #             "predictions_mask": final_mask
# #         # }
# #         # return results
#
# #         for i in range(len(query_multi_scale)):
# #             ref_feature = ref_multiscale_feature[i]
# #             bs, d, _, _ = ref_feature.shape
# #             ref_feature = ref_feature.view(bs, d, -1).permute(0, 2, 1) ### bs, n, d
#
# #             support_sets = ref_feature
# #             support_sets = support_sets.permute(1, 0, 2)  ##### N, B, D
# #             support_list.append(support_sets)
#
# #             query_feature = query_multi_scale[i].view(bs, d, -1).permute(0, 2, 1)
# #             src.append(query_feature.permute(1, 0, 2))
# #             src_copy.append(query_feature.permute(1, 0, 2).clone())
#
#
# #         for i in range(self.num_layers):
# #             level_index = i % self.num_feature_levels
# #             src_mask_features = src[level_index]
# #             spatial_tokens = support_list[level_index]
#
# #             output_pos, _ = self.transformer_cross_attention_layers[i](
# #                 src_mask_features, spatial_tokens,
# #                 memory_mask=None,
# #                 memory_key_padding_mask=None,
# #                 pos=None, query_pos=None
# #             )
#
# #             y = self.transformer_ffn_layers[i](output_pos)
# #             src[level_index] = y
#
#
# #         for i in range(len(src)):
# #             src_mask_features = src[i].permute(1, 0, 2)
# #             spatial_tokens = support_list[i].permute(1, 0, 2)
#
# #             src_mask_features = self.layer_norm(src_mask_features + src_copy[i].permute(1, 0, 2))
#
# #             src_norm = src_mask_features / (torch.norm(src_mask_features, dim=-1, keepdim=True) + 1e-12)
# #             spatial_norm = spatial_tokens / (torch.norm(spatial_tokens, dim=-1, keepdim=True) + 1e-12)
#
# #             avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
# #             avg_atten = avg_atten.softmax(dim=-1)
#
# #             out_predict = avg_atten @ ref_mask_scale
# #             out_predict_list.append(out_predict)
#
# #         results = self.forward_prediction_heads(src, out_predict_list)
# #         return results
#
#
# #     def forward_prediction_heads(self, src, out_predict_list):
# #         num_1, bs, dim = src[0].shape
# #         num_2, _, _ = src[1].shape
# #         num_3, _, _ = src[2].shape
#
# #         feature_1 = src[0].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_1)),
# #                                                     int(numpy.sqrt(num_1)))  ####(32, 32)
# #         feature_2 = src[1].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_2)),
# #                                                     int(numpy.sqrt(num_2)))  ####(32, 32)
# #         feature_3 = src[2].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_3)),
# #                                                     int(numpy.sqrt(num_3)))  ####(32, 32)
#
# #         final_fuse = self.final_fuse(torch.cat((feature_1, feature_2, feature_3), dim=1))
#
# #         out_predict = 1 / 3 * (out_predict_list[0] + out_predict_list[1] + out_predict_list[2])
#
# #         outputs_mask = self.final_predict(torch.cat((final_fuse, out_predict.reshape(bs, 32, 32, 5).permute(0, 3, 1, 2)), dim=1))
#
# #         results = {
# #             "predictions_mask": outputs_mask
# #         }
# #         return results
#
# # @register_decoder
# # def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
# #     return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
# #######################################previous version ####################################################
# # from typing import Optional
# #
# # import imgviz
# # import numpy
# # import torch
# # from torch import nn, Tensor
# # from torch.nn import functional as F
# #
# # from timm.models.layers import trunc_normal_
# # from detectron2.layers import Conv2d
# # import fvcore.nn.weight_init as weight_init
# #
# # from .utils.utils import rand_sample, prepare_features
# # from .utils.attn import MultiheadAttention
# # from .utils.attention_data_struct import AttentionDataStruct
# # from .registry import register_decoder
# # from ...utils import configurable
# # from ...modules import PositionEmbeddingSine
# # from einops import rearrange
# # from PIL import Image
# # import numpy as np
# #
# # def save_colored_mask(mask, save_path):
# #     lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
# #     color_map = imgviz.label_colormap()
# #     lbl_pil.putpalette(color_map.flatten())
# #     lbl_pil.save(save_path)
# # class SelfAttentionLayer(nn.Module):
# #
# #     def __init__(self, d_model, nhead, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt,
# #                      tgt_mask: Optional[Tensor] = None,
# #                      tgt_key_padding_mask: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None):
# #         q = k = self.with_pos_embed(tgt, query_pos)
# #         tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)[0]
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #
# #         return tgt
# #
# #     def forward_pre(self, tgt,
# #                     tgt_mask: Optional[Tensor] = None,
# #                     tgt_key_padding_mask: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):
# #         tgt2 = self.norm(tgt)
# #         q = k = self.with_pos_embed(tgt2, query_pos)
# #         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)[0]
# #         tgt = tgt + self.dropout(tgt2)
# #
# #         return tgt
# #
# #     def forward(self, tgt,
# #                 tgt_mask: Optional[Tensor] = None,
# #                 tgt_key_padding_mask: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, tgt_mask,
# #                                     tgt_key_padding_mask, query_pos)
# #         return self.forward_post(tgt, tgt_mask,
# #                                  tgt_key_padding_mask, query_pos)
# #
# #
# # class CrossAttentionLayer(nn.Module):
# #
# #     def __init__(self, d_model, nhead, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt, memory,
# #                      memory_mask: Optional[Tensor] = None,
# #                      memory_key_padding_mask: Optional[Tensor] = None,
# #                      pos: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None):
# #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
# #                                              key=self.with_pos_embed(memory, pos),
# #                                              value=memory, attn_mask=memory_mask,
# #                                              key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt, avg_attn
# #
# #     def forward_pre(self, tgt, memory,
# #                     memory_mask: Optional[Tensor] = None,
# #                     memory_key_padding_mask: Optional[Tensor] = None,
# #                     pos: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):
# #         tgt2 = self.norm(tgt)
# #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
# #                                              key=self.with_pos_embed(memory, pos),
# #                                              value=memory, attn_mask=memory_mask,
# #                                              key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout(tgt2)
# #
# #         return tgt, avg_attn
# #
# #     def forward(self, tgt, memory,
# #                 memory_mask: Optional[Tensor] = None,
# #                 memory_key_padding_mask: Optional[Tensor] = None,
# #                 pos: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, memory, memory_mask,
# #                                     memory_key_padding_mask, pos, query_pos)
# #         return self.forward_post(tgt, memory, memory_mask,
# #                                  memory_key_padding_mask, pos, query_pos)
# #
# #
# # class FFNLayer(nn.Module):
# #
# #     def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         # Implementation of Feedforward model
# #         self.linear1 = nn.Linear(d_model, dim_feedforward)
# #         self.dropout = nn.Dropout(dropout)
# #         self.linear2 = nn.Linear(dim_feedforward, d_model)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt):
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt
# #
# #     def forward_pre(self, tgt):
# #         tgt2 = self.norm(tgt)
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
# #         tgt = tgt + self.dropout(tgt2)
# #         return tgt
# #
# #     def forward(self, tgt):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt)
# #         return self.forward_post(tgt)
# #
# #
# # def _get_activation_fn(activation):
# #     """Return an activation function given a string"""
# #     if activation == "relu":
# #         return F.relu
# #     if activation == "gelu":
# #         return F.gelu
# #     if activation == "glu":
# #         return F.glu
# #     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
# #
# #
# # class MLP(nn.Module):
# #     """ Very simple multi-layer perceptron (also called FFN)"""
# #
# #     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
# #         super().__init__()
# #         self.num_layers = num_layers
# #         h = [hidden_dim] * (num_layers - 1)
# #         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
# #
# #     def forward(self, x):
# #         for i, layer in enumerate(self.layers):
# #             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
# #         return x
# #
# #
# # class MultiScaleMaskedTransformerDecoder(nn.Module):
# #     _version = 2
# #
# #     @configurable
# #     def __init__(
# #             self,
# #             lang_encoder: nn.Module,
# #             in_channels,
# #             mask_classification=True,
# #             *,
# #             hidden_dim: int,
# #             dim_proj: int,
# #             num_queries: int,
# #             contxt_len: int,
# #             nheads: int,
# #             dim_feedforward: int,
# #             dec_layers: int,
# #             pre_norm: bool,
# #             mask_dim: int,
# #             task_switch: dict,
# #             enforce_input_project: bool,
# #             max_spatial_len: int,
# #             attn_arch: dict,
# #     ):
# #
# #         super().__init__()
# #         assert mask_classification, "Only support mask classification model"
# #         self.mask_classification = mask_classification
# #
# #         N_steps = hidden_dim // 2
# #         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
# #
# #         # define Transformer decoder here
# #         self.num_heads = nheads
# #         self.num_layers = dec_layers
# #         self.contxt_len = contxt_len
# #
# #         self.transformer_self_attention_layers = nn.ModuleList()
# #         self.transformer_cross_attention_layers = nn.ModuleList()
# #         self.transformer_ffn_layers = nn.ModuleList()
# #
# #         self.linear_list = nn.ModuleList()
# #         for _ in range(3):
# #             self.linear_list.append(
# #             nn.Sequential(nn.Conv2d(1024, 512, 1))
# #         )
# #
# #         self.final_predict = nn.Sequential(
# #             nn.BatchNorm2d(512+10),
# #             nn.Conv2d(512+10, 10, 3, 1, 1))
# #
# #         self.final_fuse = nn.Sequential(
# #             nn.Conv2d(512 * 3, 512, 3, 1, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU())
# #
# #         self.extra_cross = CrossAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #
# #         for _ in range(self.num_layers):
# #             self.transformer_self_attention_layers.append(
# #                 SelfAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #             self.transformer_cross_attention_layers.append(
# #                 CrossAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #             self.transformer_ffn_layers.append(
# #                 FFNLayer(
# #                     d_model=hidden_dim,
# #                     dim_feedforward=dim_feedforward,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #         self.decoder_norm = nn.LayerNorm(hidden_dim)
# #         self.num_queries = num_queries
# #         self.query_feat = nn.Embedding(num_queries, hidden_dim)
# #
# #         self.query_embed = nn.Embedding(num_queries, hidden_dim)
# #         self.pn_indicator = nn.Embedding(2, hidden_dim)
# #
# #         self.num_feature_levels = 3
# #         self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
# #         self.input_proj = nn.ModuleList()
# #
# #         for _ in range(self.num_feature_levels):
# #             if in_channels != hidden_dim or enforce_input_project:
# #                 self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
# #                 weight_init.c2_xavier_fill(self.input_proj[-1])
# #             else:
# #                 self.input_proj.append(nn.Sequential())
# #
# #         self.task_switch = task_switch
# #         self.query_index = {}
# #
# #         # output FFNs
# #         self.lang_encoder = lang_encoder
# #         if self.task_switch['mask']:
# #             self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
# #
# #         self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
# #         trunc_normal_(self.class_embed, std=.02)
# #
# #         if task_switch['spatial']:
# #             self.mask_sptial_embed = nn.ParameterList(
# #                 [nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(3)])
# #             trunc_normal_(self.mask_sptial_embed[0], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[1], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[2], std=.02)
# #
# #             self.max_spatial_len = max_spatial_len
# #
# #             num_spatial_memories = attn_arch['SPATIAL_MEMORIES']
# #             self.spatial_embed = nn.Embedding(num_spatial_memories, hidden_dim)
# #             self.spatial_featured = nn.Embedding(num_spatial_memories, hidden_dim)
# #
# #         # build AttentionDataStruct
# #         attn_arch['NUM_LAYERS'] = self.num_layers
# #         self.attention_data = AttentionDataStruct(attn_arch, task_switch)
# #
# #
# #     @classmethod
# #     def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
# #         ret = {}
# #
# #         ret["lang_encoder"] = lang_encoder
# #         ret["in_channels"] = in_channels
# #         ret["mask_classification"] = mask_classification
# #
# #         enc_cfg = cfg['MODEL']['ENCODER']
# #         dec_cfg = cfg['MODEL']['DECODER']
# #
# #         ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
# #         ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
# #         ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
# #         ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
# #
# #         # Transformer parameters:
# #         ret["nheads"] = dec_cfg['NHEADS']
# #         ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']
# #
# #         assert dec_cfg['DEC_LAYERS'] >= 1
# #         ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
# #         ret["pre_norm"] = dec_cfg['PRE_NORM']
# #         ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
# #         ret["mask_dim"] = enc_cfg['MASK_DIM']
# #         ret["task_switch"] = extra['task_switch']
# #         ret["max_spatial_len"] = dec_cfg['MAX_SPATIAL_LEN']
# #
# #         # attn data struct
# #         ret["attn_arch"] = cfg['ATTENTION_ARCH']
# #
# #         return ret
# #
# #     def forward(self, ref_information, query_information, extra={}, task='seg'):
# #         query_multi_scale = query_information
# #         ref_multiscale_feature, ref_mask = ref_information
# #
# #         assert len(query_multi_scale) == self.num_feature_levels;
# #         spatial_extra_flag = 'spatial_query_pos_mask' in extra.keys() or task == 'refimg'
# #         grounding_extra_flag = 'grounding_tokens' in extra.keys()
# #         visual_extra_flag = 'visual_query_pos' in extra.keys()
# #         audio_extra_flag = 'audio_tokens' in extra.keys()
# #         spatial_memory_flag = 'prev_mask' in extra.keys()
# #         flags = {"spatial": spatial_extra_flag, "grounding": grounding_extra_flag,
# #                          "memories_spatial": spatial_memory_flag, "visual": visual_extra_flag, "audio": audio_extra_flag}
# #         self.attention_data.reset(flags, task, extra)
# #
# #         support_list = []
# #         src = []
# #         out_predict_list = []
# #
# #         bs, c, h, w = ref_mask.tensor.shape
# #         ref_mask_scale = F.interpolate(ref_mask.tensor, (32, 32), mode='nearest')
# #         ref_mask_scale = ref_mask_scale.reshape(bs, c, -1).permute(0, 2, 1)
# #
# #         # for i in range(len(query_multi_scale)):
# #         #     ref_feature = ref_multiscale_feature[i]  ### (2, 1, 1024, 32, 32), (2, 11, 1, 32, 32)  (2, 11, 1024, 32, 32)
# #         #     support_sets = self.linear_list[i](ref_feature)
# #         #
# #         #     bs, d, h, w = support_sets.shape
# #         #     support_sets = support_sets.view(bs, d, -1).permute(2, 0, 1)  ##### N, B, D
# #         #     support_list.append(support_sets)
# #         #     src.append(self.linear_list[i](query_multi_scale[i]).view(bs, d, -1).permute(2, 0, 1))
# #         #
# #         # for i in range(self.num_layers):
# #         #     level_index = i % self.num_feature_levels
# #         #     src_mask_features = src[level_index]
# #         #     spatial_tokens = support_list[level_index]
# #         #
# #         #     output_pos, avg_attn = self.transformer_cross_attention_layers[i](
# #         #         src_mask_features, spatial_tokens,
# #         #         memory_mask=None,
# #         #         memory_key_padding_mask=None,
# #         #         pos=None, query_pos=None
# #         #     )
# #         #
# #         #     if i > 5:
# #         #         out_predict = torch.bmm(avg_attn, ref_mask_scale)
# #         #         out_predict_list.append(out_predict)
# #         #
# #         #     y = self.transformer_ffn_layers[i](output_pos)
# #         #     src[level_index] = y + src_mask_features
# #         #
# #         # results = self.forward_prediction_heads(src, out_predict_list)
# #         # return results
# #
# #     def forward_prediction_heads(self, src, out_predict_list):
# #         num_1, bs, dim = src[0].shape
# #         num_2, _, _ = src[1].shape
# #         num_3, _, _ = src[2].shape
# #
# #         feature_1 = src[0].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_1)),
# #                                                     int(numpy.sqrt(num_1)))  ####(32, 32)
# #         feature_2 = src[1].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_2)),
# #                                                     int(numpy.sqrt(num_2)))  ####(32, 32)
# #         feature_3 = src[2].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_3)),
# #                                                     int(numpy.sqrt(num_3)))  ####(32, 32)
# #
# #         final_fuse = self.final_fuse(torch.cat((feature_1, feature_2, feature_3), dim=1))
# #
# #         # out_view = out_predict_list[0].reshape(bs, 32, 32, 10).permute(0, 3, 1, 2)
# #         # out_view = F.interpolate(out_view, (448, 448), align_corners=True, mode='bilinear')  ### b c h w
# #
# #         # # # for i in range(out_view[0].shape[0]):
# #         # print((out_view > 0.5).sum())
# #         # view_img = torch.argmax(out_view.softmax(dim=1), dim=1)
# #         # view_img = view_img[0].detach().cpu().numpy()
# #         # # print(np.unique(view_img))
# #         # #     # print(view_img.sum())
# #         # save_colored_mask(view_img, '1.png')
# #         # # view_img = Image.fromarray(np.uint8(view_img * 255))
# #         # # view_img.save(str(i)+'.png')
# #
# #         out_predict = 1 / 3 * (out_predict_list[0] + out_predict_list[1] + out_predict_list[2])
# #
# #         outputs_mask = self.final_predict(torch.cat((final_fuse, out_predict.reshape(bs, 32, 32, 10).permute(0, 3, 1, 2)), dim=1))
# #
# #         results = {
# #             "predictions_mask": outputs_mask
# #         }
# #         return results
# #
# # @register_decoder
# # def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
# #     return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
#
# ##################################### test 代码 ###################################################################################
# # bs, d, h, w = ref_multiscale_feature[0].shape
# #         ref_feature = ref_multiscale_feature[0].view(bs, d, -1)
# #         query_feature = query_multi_scale[0].view(bs, d, -1).transpose(-1, -2)
# #
# #         atten = (query_feature @ ref_feature).softmax(dim=-1)
# #         final_mask = atten @ ref_mask_scale
# #         final_mask = final_mask.reshape(bs, 32, 32, 10).permute(0, 3, 1, 2)
# #
# #         # for i in range(final_mask[0].shape[0]):
# #         #     view_img = final_mask[0][i].cpu().numpy()
# #         #     # print(view_img.sum())
# #         #     view_img = Image.fromarray(np.uint8(view_img))
# #         #     view_img.save(str(i)+'.png')
# #
# #         results = {
# #             "predictions_mask": final_mask
# #         }
# #         return results
#
# # # q_norm = (query_feature / torch.norm(query_feature, dim=-1, keepdim=True))
# #         # k_norm = (ref_feature / torch.norm(ref_feature, dim=-1, keepdim=True))
# #         # atten = torch.mul(q_norm, k_norm.transpose(-1, -2))
# #
# #         atten = query_feature @ ref_feature
# #         final_mask = atten @ ref_mask_scale
# #         final_mask = final_mask.reshape(bs, 32, 32, 10).permute(0, 3, 1, 2)
# #
# #         # for i in range(final_mask[0].shape[0]):
# #         #     view_img = final_mask[0][i].cpu().numpy()
# #         #     # print(view_img.sum())
# #         #     view_img = Image.fromarray(np.uint8(view_img))
# #         #     view_img.save(str(i)+'.png')
# #
# #         results = {
# #                     "predictions_mask": final_mask
# #         }
# #         return results
#
# #######################################cross attention+dinov2+ref_mask+0/1mask ####################################################
# # from typing import Optional
# # import numpy
# # import torch
# # from torch import nn, Tensor
# # from torch.nn import functional as F
# #
# # from timm.models.layers import trunc_normal_
# # from detectron2.layers import Conv2d
# # import fvcore.nn.weight_init as weight_init
# #
# # from .utils.utils import rand_sample, prepare_features
# # from .utils.attn import MultiheadAttention
# # from .utils.attention_data_struct import AttentionDataStruct
# # from .registry import register_decoder
# # from ...utils import configurable
# # from ...modules import PositionEmbeddingSine
# # from einops import rearrange
# #
# # class SelfAttentionLayer(nn.Module):
# #
# #     def __init__(self, d_model, nhead, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt,
# #                      tgt_mask: Optional[Tensor] = None,
# #                      tgt_key_padding_mask: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None):
# #         q = k = self.with_pos_embed(tgt, query_pos)
# #         tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)[0]
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #
# #         return tgt
# #
# #     def forward_pre(self, tgt,
# #                     tgt_mask: Optional[Tensor] = None,
# #                     tgt_key_padding_mask: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):
# #         tgt2 = self.norm(tgt)
# #         q = k = self.with_pos_embed(tgt2, query_pos)
# #         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)[0]
# #         tgt = tgt + self.dropout(tgt2)
# #
# #         return tgt
# #
# #     def forward(self, tgt,
# #                 tgt_mask: Optional[Tensor] = None,
# #                 tgt_key_padding_mask: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, tgt_mask,
# #                                     tgt_key_padding_mask, query_pos)
# #         return self.forward_post(tgt, tgt_mask,
# #                                  tgt_key_padding_mask, query_pos)
# #
# #
# # class CrossAttentionLayer(nn.Module):
# #
# #     def __init__(self, d_model, nhead, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt, memory,
# #                      memory_mask: Optional[Tensor] = None,
# #                      memory_key_padding_mask: Optional[Tensor] = None,
# #                      pos: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None):
# #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
# #                                              key=self.with_pos_embed(memory, pos),
# #                                              value=memory, attn_mask=memory_mask,
# #                                              key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt, avg_attn
# #
# #     def forward_pre(self, tgt, memory,
# #                     memory_mask: Optional[Tensor] = None,
# #                     memory_key_padding_mask: Optional[Tensor] = None,
# #                     pos: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):
# #         tgt2 = self.norm(tgt)
# #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
# #                                              key=self.with_pos_embed(memory, pos),
# #                                              value=memory, attn_mask=memory_mask,
# #                                              key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout(tgt2)
# #
# #         return tgt, avg_attn
# #
# #     def forward(self, tgt, memory,
# #                 memory_mask: Optional[Tensor] = None,
# #                 memory_key_padding_mask: Optional[Tensor] = None,
# #                 pos: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, memory, memory_mask,
# #                                     memory_key_padding_mask, pos, query_pos)
# #         return self.forward_post(tgt, memory, memory_mask,
# #                                  memory_key_padding_mask, pos, query_pos)
# #
# #
# # class FFNLayer(nn.Module):
# #
# #     def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         # Implementation of Feedforward model
# #         self.linear1 = nn.Linear(d_model, dim_feedforward)
# #         self.dropout = nn.Dropout(dropout)
# #         self.linear2 = nn.Linear(dim_feedforward, d_model)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt):
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt
# #
# #     def forward_pre(self, tgt):
# #         tgt2 = self.norm(tgt)
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
# #         tgt = tgt + self.dropout(tgt2)
# #         return tgt
# #
# #     def forward(self, tgt):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt)
# #         return self.forward_post(tgt)
# #
# #
# # def _get_activation_fn(activation):
# #     """Return an activation function given a string"""
# #     if activation == "relu":
# #         return F.relu
# #     if activation == "gelu":
# #         return F.gelu
# #     if activation == "glu":
# #         return F.glu
# #     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
# #
# #
# # class MLP(nn.Module):
# #     """ Very simple multi-layer perceptron (also called FFN)"""
# #
# #     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
# #         super().__init__()
# #         self.num_layers = num_layers
# #         h = [hidden_dim] * (num_layers - 1)
# #         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
# #
# #     def forward(self, x):
# #         for i, layer in enumerate(self.layers):
# #             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
# #         return x
# #
# #
# # class MultiScaleMaskedTransformerDecoder(nn.Module):
# #     _version = 2
# #
# #     @configurable
# #     def __init__(
# #             self,
# #             lang_encoder: nn.Module,
# #             in_channels,
# #             mask_classification=True,
# #             *,
# #             hidden_dim: int,
# #             dim_proj: int,
# #             num_queries: int,
# #             contxt_len: int,
# #             nheads: int,
# #             dim_feedforward: int,
# #             dec_layers: int,
# #             pre_norm: bool,
# #             mask_dim: int,
# #             task_switch: dict,
# #             enforce_input_project: bool,
# #             max_spatial_len: int,
# #             attn_arch: dict,
# #     ):
# #
# #         super().__init__()
# #         assert mask_classification, "Only support mask classification model"
# #         self.mask_classification = mask_classification
# #
# #         N_steps = hidden_dim // 2
# #         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
# #
# #         # define Transformer decoder here
# #         self.num_heads = nheads
# #         self.num_layers = dec_layers
# #         self.contxt_len = contxt_len
# #
# #         self.transformer_self_attention_layers = nn.ModuleList()
# #         self.transformer_cross_attention_layers = nn.ModuleList()
# #         self.transformer_ffn_layers = nn.ModuleList()
# #
# #         self.final_predict = nn.Conv2d(512, 11, 3, 1, 1)
# #         self.final_fuse = nn.Sequential(
# #             nn.Conv2d(512 * 3, 512, 3, 1, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU())
# #
# #         self.mask_proj = nn.Sequential(
# #             nn.Conv2d(10, 512, 3, 1, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU()
# #         )
# #
# #         self.fr_fusion= nn.ModuleList()
# #         for _ in range(3):
# #             self.fr_fusion.append(nn.Sequential(
# #             nn.Conv2d(1024 + 512, 512, 3, 1, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU()
# #         ))
# #
# #         self.linear_list = nn.ModuleList()
# #         for _ in range(3):
# #             self.linear_list.append(
# #                 nn.Sequential(nn.Conv2d(1024, 512, 1))
# #             )
# #
# #         for _ in range(self.num_layers):
# #             self.transformer_self_attention_layers.append(
# #                 SelfAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #             self.transformer_cross_attention_layers.append(
# #                 CrossAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #             self.transformer_ffn_layers.append(
# #                 FFNLayer(
# #                     d_model=hidden_dim,
# #                     dim_feedforward=dim_feedforward,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #         self.decoder_norm = nn.LayerNorm(hidden_dim)
# #         self.num_queries = num_queries
# #         self.query_feat = nn.Embedding(num_queries, hidden_dim)
# #
# #         self.query_embed = nn.Embedding(num_queries, hidden_dim)
# #         self.pn_indicator = nn.Embedding(2, hidden_dim)
# #
# #         self.num_feature_levels = 3
# #         self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
# #         self.input_proj = nn.ModuleList()
# #
# #         for _ in range(self.num_feature_levels):
# #             if in_channels != hidden_dim or enforce_input_project:
# #                 self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
# #                 weight_init.c2_xavier_fill(self.input_proj[-1])
# #             else:
# #                 self.input_proj.append(nn.Sequential())
# #
# #         self.task_switch = task_switch
# #         self.query_index = {}
# #
# #         # output FFNs
# #         self.lang_encoder = lang_encoder
# #         if self.task_switch['mask']:
# #             self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
# #
# #         self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
# #         trunc_normal_(self.class_embed, std=.02)
# #
# #         if task_switch['spatial']:
# #             self.mask_sptial_embed = nn.ParameterList(
# #                 [nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(3)])
# #             trunc_normal_(self.mask_sptial_embed[0], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[1], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[2], std=.02)
# #
# #             self.max_spatial_len = max_spatial_len
# #
# #             num_spatial_memories = attn_arch['SPATIAL_MEMORIES']
# #             self.spatial_embed = nn.Embedding(num_spatial_memories, hidden_dim)
# #             self.spatial_featured = nn.Embedding(num_spatial_memories, hidden_dim)
# #
# #         # build AttentionDataStruct
# #         attn_arch['NUM_LAYERS'] = self.num_layers
# #         self.attention_data = AttentionDataStruct(attn_arch, task_switch)
# #
# #
# #     @classmethod
# #     def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
# #         ret = {}
# #
# #         ret["lang_encoder"] = lang_encoder
# #         ret["in_channels"] = in_channels
# #         ret["mask_classification"] = mask_classification
# #
# #         enc_cfg = cfg['MODEL']['ENCODER']
# #         dec_cfg = cfg['MODEL']['DECODER']
# #
# #         ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
# #         ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
# #         ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
# #         ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
# #
# #         # Transformer parameters:
# #         ret["nheads"] = dec_cfg['NHEADS']
# #         ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']
# #
# #         assert dec_cfg['DEC_LAYERS'] >= 1
# #         ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
# #         ret["pre_norm"] = dec_cfg['PRE_NORM']
# #         ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
# #         ret["mask_dim"] = enc_cfg['MASK_DIM']
# #         ret["task_switch"] = extra['task_switch']
# #         ret["max_spatial_len"] = dec_cfg['MAX_SPATIAL_LEN']
# #
# #         # attn data struct
# #         ret["attn_arch"] = cfg['ATTENTION_ARCH']
# #
# #         return ret
# #
# #     def forward(self, ref_information, query_information, extra={}, task='seg'):
# #         query_multi_scale = query_information
# #         ref_multiscale_feature, ref_mask = ref_information
# #
# #         assert len(query_multi_scale) == self.num_feature_levels;
# #         spatial_extra_flag = 'spatial_query_pos_mask' in extra.keys() or task == 'refimg'
# #         grounding_extra_flag = 'grounding_tokens' in extra.keys()
# #         visual_extra_flag = 'visual_query_pos' in extra.keys()
# #         audio_extra_flag = 'audio_tokens' in extra.keys()
# #         spatial_memory_flag = 'prev_mask' in extra.keys()
# #         flags = {"spatial": spatial_extra_flag, "grounding": grounding_extra_flag,
# #                          "memories_spatial": spatial_memory_flag, "visual": visual_extra_flag, "audio": audio_extra_flag}
# #         self.attention_data.reset(flags, task, extra)
# #
# #         spatial_list = [32, 32, 32]
# #         ref_mask_list = []
# #         for i in range(len(spatial_list)):
# #             ref_mask_scale = F.interpolate(ref_mask.tensor, (spatial_list[i], spatial_list[i]), mode='nearest')
# #             ref_mask_scale = self.mask_proj(ref_mask_scale)
# #             ref_mask_list.append(ref_mask_scale)
# #
# #         support_list = []
# #         src = []
# #
# #         for i in range(len(query_multi_scale)):
# #             ref_feature = ref_multiscale_feature[i]  ### (2, 1, 1024, 32, 32), (2, 11, 1, 32, 32)  (2, 11, 1024, 32, 32)
# #             support_sets = torch.cat((ref_feature, ref_mask_list[i]), dim=1)
# #             # print(support_sets)
# #             support_sets = self.fr_fusion[i](support_sets)
# #
# #             bs, d, h, w = support_sets.shape
# #             support_sets = support_sets.view(bs, d, -1).permute(2, 0, 1)  ##### N, B, D
# #             support_list.append(support_sets)
# #             src.append(self.linear_list[i](query_multi_scale[i]).view(bs, d, -1).permute(2, 0, 1))
# #
# #         for i in range(self.num_layers):
# #             level_index = i % self.num_feature_levels
# #             src_mask_features = src[level_index]
# #             spatial_tokens = support_list[level_index]
# #
# #             output_pos, avg_attn = self.transformer_cross_attention_layers[i](
# #                 src_mask_features, spatial_tokens,
# #                 memory_mask=None,
# #                 memory_key_padding_mask=None,
# #                 pos=None, query_pos=None
# #             )
# #
# #             y = self.transformer_ffn_layers[i](output_pos)
# #             src[level_index] = y
# #
# #         results = self.forward_prediction_heads(src)
# #         return results
# #
# #     def forward_prediction_heads(self, src):
# #         num_1, bs, dim = src[0].shape
# #         num_2, _, _ = src[1].shape
# #         num_3, _, _ = src[2].shape
# #
# #         feature_1 = src[0].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_1)),
# #                                                     int(numpy.sqrt(num_1)))  ####(32, 32)
# #         feature_2 = src[1].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_2)),
# #                                                     int(numpy.sqrt(num_2)))  ####(32, 32)
# #         feature_3 = src[2].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_3)),
# #                                                     int(numpy.sqrt(num_3)))  ####(32, 32)
# #
# #         final_fuse = self.final_fuse(torch.cat((feature_1, feature_2, feature_3), dim=1))
# #         outputs_mask = self.final_predict(final_fuse)
# #
# #         results = {
# #             "predictions_mask": outputs_mask
# #         }
# #         return results
# #
# # @register_decoder
# # def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
# #     return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
#
#
# #############################################################################################################################################################
# # from typing import Optional
# #
# # import numpy
# # import torch
# # from torch import nn, Tensor
# # from torch.nn import functional as F
# #
# # from timm.models.layers import trunc_normal_
# # from detectron2.layers import Conv2d
# # import fvcore.nn.weight_init as weight_init
# #
# # from .utils.utils import rand_sample, prepare_features
# # from .utils.attn import MultiheadAttention
# # from .utils.attention_data_struct import AttentionDataStruct
# # from .registry import register_decoder
# # from ...utils import configurable
# # from ...modules import PositionEmbeddingSine
# # from einops import rearrange
# #
# # class SelfAttentionLayer(nn.Module):
# #
# #     def __init__(self, d_model, nhead, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt,
# #                      tgt_mask: Optional[Tensor] = None,
# #                      tgt_key_padding_mask: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None):
# #         q = k = self.with_pos_embed(tgt, query_pos)
# #         tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)[0]
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #
# #         return tgt
# #
# #     def forward_pre(self, tgt,
# #                     tgt_mask: Optional[Tensor] = None,
# #                     tgt_key_padding_mask: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):
# #         tgt2 = self.norm(tgt)
# #         q = k = self.with_pos_embed(tgt2, query_pos)
# #         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)[0]
# #         tgt = tgt + self.dropout(tgt2)
# #
# #         return tgt
# #
# #     def forward(self, tgt,
# #                 tgt_mask: Optional[Tensor] = None,
# #                 tgt_key_padding_mask: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, tgt_mask,
# #                                     tgt_key_padding_mask, query_pos)
# #         return self.forward_post(tgt, tgt_mask,
# #                                  tgt_key_padding_mask, query_pos)
# #
# #
# # class CrossAttentionLayer(nn.Module):
# #
# #     def __init__(self, d_model, nhead, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt, memory,
# #                      memory_mask: Optional[Tensor] = None,
# #                      memory_key_padding_mask: Optional[Tensor] = None,
# #                      pos: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None):
# #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
# #                                              key=self.with_pos_embed(memory, pos),
# #                                              value=memory, attn_mask=memory_mask,
# #                                              key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt, avg_attn
# #
# #     def forward_pre(self, tgt, memory,
# #                     memory_mask: Optional[Tensor] = None,
# #                     memory_key_padding_mask: Optional[Tensor] = None,
# #                     pos: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):
# #         tgt2 = self.norm(tgt)
# #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
# #                                              key=self.with_pos_embed(memory, pos),
# #                                              value=memory, attn_mask=memory_mask,
# #                                              key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout(tgt2)
# #
# #         return tgt, avg_attn
# #
# #     def forward(self, tgt, memory,
# #                 memory_mask: Optional[Tensor] = None,
# #                 memory_key_padding_mask: Optional[Tensor] = None,
# #                 pos: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, memory, memory_mask,
# #                                     memory_key_padding_mask, pos, query_pos)
# #         return self.forward_post(tgt, memory, memory_mask,
# #                                  memory_key_padding_mask, pos, query_pos)
# #
# #
# # class FFNLayer(nn.Module):
# #
# #     def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         # Implementation of Feedforward model
# #         self.linear1 = nn.Linear(d_model, dim_feedforward)
# #         self.dropout = nn.Dropout(dropout)
# #         self.linear2 = nn.Linear(dim_feedforward, d_model)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt):
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt
# #
# #     def forward_pre(self, tgt):
# #         tgt2 = self.norm(tgt)
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
# #         tgt = tgt + self.dropout(tgt2)
# #         return tgt
# #
# #     def forward(self, tgt):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt)
# #         return self.forward_post(tgt)
# #
# #
# # def _get_activation_fn(activation):
# #     """Return an activation function given a string"""
# #     if activation == "relu":
# #         return F.relu
# #     if activation == "gelu":
# #         return F.gelu
# #     if activation == "glu":
# #         return F.glu
# #     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
# #
# #
# # class MLP(nn.Module):
# #     """ Very simple multi-layer perceptron (also called FFN)"""
# #
# #     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
# #         super().__init__()
# #         self.num_layers = num_layers
# #         h = [hidden_dim] * (num_layers - 1)
# #         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
# #
# #     def forward(self, x):
# #         for i, layer in enumerate(self.layers):
# #             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
# #         return x
# #
# #
# # class MultiScaleMaskedTransformerDecoder(nn.Module):
# #     _version = 2
# #
# #     @configurable
# #     def __init__(
# #             self,
# #             lang_encoder: nn.Module,
# #             in_channels,
# #             mask_classification=True,
# #             *,
# #             hidden_dim: int,
# #             dim_proj: int,
# #             num_queries: int,
# #             contxt_len: int,
# #             nheads: int,
# #             dim_feedforward: int,
# #             dec_layers: int,
# #             pre_norm: bool,
# #             mask_dim: int,
# #             task_switch: dict,
# #             enforce_input_project: bool,
# #             max_spatial_len: int,
# #             attn_arch: dict,
# #     ):
# #
# #         super().__init__()
# #         assert mask_classification, "Only support mask classification model"
# #         self.mask_classification = mask_classification
# #
# #         N_steps = hidden_dim // 2
# #         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
# #
# #         # define Transformer decoder here
# #         self.num_heads = nheads
# #         self.num_layers = dec_layers
# #         self.contxt_len = contxt_len
# #
# #         self.transformer_self_attention_layers = nn.ModuleList()
# #         self.transformer_cross_attention_layers = nn.ModuleList()
# #         self.transformer_ffn_layers = nn.ModuleList()
# #
# #         self.final_predict = nn.Conv2d(512, 11, 3, 1)  #### pos prompt define as 10, neg prompt as 1
# #
# #         self.final_fuse = nn.Sequential(
# #             nn.Conv2d(512 * 3, 512, 3, 1, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU())
# #
# #         self.fr_fusion= nn.ModuleList()
# #         for _ in range(3):
# #             self.fr_fusion.append(nn.Sequential(
# #             nn.Conv2d(1024 + 11, 512, 3, 1, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU()
# #         ))
# #
# #         self.linear_list = nn.ModuleList()
# #         for _ in range(3):
# #             self.linear_list.append(
# #                 nn.Sequential(nn.Conv2d(1024, 512, 1))
# #             )
# #
# #
# #         for _ in range(self.num_layers):
# #             self.transformer_self_attention_layers.append(
# #                 SelfAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #             self.transformer_cross_attention_layers.append(
# #                 CrossAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #             self.transformer_ffn_layers.append(
# #                 FFNLayer(
# #                     d_model=hidden_dim,
# #                     dim_feedforward=dim_feedforward,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #         self.decoder_norm = nn.LayerNorm(hidden_dim)
# #         self.num_queries = num_queries
# #         self.query_feat = nn.Embedding(num_queries, hidden_dim)
# #
# #         self.query_embed = nn.Embedding(num_queries, hidden_dim)
# #         self.pn_indicator = nn.Embedding(2, hidden_dim)
# #
# #         self.num_feature_levels = 3
# #         self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
# #         self.input_proj = nn.ModuleList()
# #
# #         for _ in range(self.num_feature_levels):
# #             if in_channels != hidden_dim or enforce_input_project:
# #                 self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
# #                 weight_init.c2_xavier_fill(self.input_proj[-1])
# #             else:
# #                 self.input_proj.append(nn.Sequential())
# #
# #         self.task_switch = task_switch
# #         self.query_index = {}
# #
# #         # output FFNs
# #         self.lang_encoder = lang_encoder
# #         if self.task_switch['mask']:
# #             self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
# #
# #         self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
# #         trunc_normal_(self.class_embed, std=.02)
# #
# #         if task_switch['spatial']:
# #             self.mask_sptial_embed = nn.ParameterList(
# #                 [nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(3)])
# #             trunc_normal_(self.mask_sptial_embed[0], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[1], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[2], std=.02)
# #
# #             self.max_spatial_len = max_spatial_len
# #
# #             num_spatial_memories = attn_arch['SPATIAL_MEMORIES']
# #             self.spatial_embed = nn.Embedding(num_spatial_memories, hidden_dim)
# #             self.spatial_featured = nn.Embedding(num_spatial_memories, hidden_dim)
# #
# #         # build AttentionDataStruct
# #         attn_arch['NUM_LAYERS'] = self.num_layers
# #         self.attention_data = AttentionDataStruct(attn_arch, task_switch)
# #
# #
# #     @classmethod
# #     def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
# #         ret = {}
# #
# #         ret["lang_encoder"] = lang_encoder
# #         ret["in_channels"] = in_channels
# #         ret["mask_classification"] = mask_classification
# #
# #         enc_cfg = cfg['MODEL']['ENCODER']
# #         dec_cfg = cfg['MODEL']['DECODER']
# #
# #         ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
# #         ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
# #         ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
# #         ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
# #
# #         # Transformer parameters:
# #         ret["nheads"] = dec_cfg['NHEADS']
# #         ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']
# #
# #         assert dec_cfg['DEC_LAYERS'] >= 1
# #         ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
# #         ret["pre_norm"] = dec_cfg['PRE_NORM']
# #         ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
# #         ret["mask_dim"] = enc_cfg['MASK_DIM']
# #         ret["task_switch"] = extra['task_switch']
# #         ret["max_spatial_len"] = dec_cfg['MAX_SPATIAL_LEN']
# #
# #         # attn data struct
# #         ret["attn_arch"] = cfg['ATTENTION_ARCH']
# #
# #         return ret
# #
# #     def forward(self, ref_information, query_information, extra={}, task='seg'):
# #         query_multi_scale = query_information
# #         ref_multiscale_feature, ref_mask = ref_information
# #
# #         assert len(query_multi_scale) == self.num_feature_levels;
# #         spatial_extra_flag = 'spatial_query_pos_mask' in extra.keys() or task == 'refimg'
# #         grounding_extra_flag = 'grounding_tokens' in extra.keys()
# #         visual_extra_flag = 'visual_query_pos' in extra.keys()
# #         audio_extra_flag = 'audio_tokens' in extra.keys()
# #         spatial_memory_flag = 'prev_mask' in extra.keys()
# #         flags = {"spatial": spatial_extra_flag, "grounding": grounding_extra_flag,
# #                          "memories_spatial": spatial_memory_flag, "visual": visual_extra_flag, "audio": audio_extra_flag}
# #         self.attention_data.reset(flags, task, extra)
# #
# #         spatial_list = [32, 32, 32]
# #         ref_mask_list = []
# #         for i in range(len(spatial_list)):
# #             ref_mask_scale = F.interpolate(ref_mask.tensor, (spatial_list[i], spatial_list[i]), mode='nearest')
# #             ref_mask_list.append(ref_mask_scale)
# #
# #         support_list = []
# #         src = []
# #         # results_list = []
# #
# # #         for i in range(len(query_multi_scale)):
# # #             ref_feature = ref_multiscale_feature[i]  ### （2, 1024, 32, 32)
# # #             ref_feature = self.linear_list[i](ref_feature)  ### (2, 512, 32, 32)
# # #
# # #             ref_mask = ref_mask_list[i]  #### (2, 11, 32, 32)
# # #
# # #             ref_feature = ref_feature.unsqueeze(1).repeat(1, 11, 1, 1, 1)
# # #             ref_mask = ref_mask.unsqueeze(2).repeat(1, 1, 512, 1, 1)
# # #             ref_feature_aug = (ref_feature * ref_mask).permute(1, 0, 2, 3, 4)  ##### (11, 2, 512, 32, 32)
# # #
# # #             n, bs, d, h, w = ref_feature_aug.shape
# # #             src.append(self.linear_list[i](query_multi_scale[i]).view(bs, d, -1).permute(2, 0, 1))
# # #             support_sets = ref_feature_aug.reshape(11, bs, d, -1).permute(0, 3, 1, 2)
# # #             support_list.append(support_sets)
# # #
# # #         for id in range(11):
# # #
# # #             for i in range(self.num_layers):
# # #                 level_index = i % self.num_feature_levels
# # #                 src_mask_features = src[level_index]
# # #                 spatial_tokens = support_list[level_index]
# # #
# # #                 output_pos, avg_attn = self.transformer_cross_attention_layers[i](
# # #                                                 src_mask_features, spatial_tokens[id],
# # #                                                 memory_mask=None,
# # #                                                 memory_key_padding_mask=None,
# # #                                                 pos=None, query_pos=None
# # #                                             )
# # #                 y = self.transformer_ffn_layers[i](output_pos)
# # #                 src[level_index] = y
# # #
# # #             result = self.forward_prediction_heads(src)["predictions_mask"]
# # #             results_list.append(result)
# # #
# # #         results = torch.stack(results_list, dim=0)  ### (11, 2, 1, 512, 512)
# # #         results = results.permute(1, 0, 2, 3, 4).squeeze(2)
# # #         return results
# # #
# # #     def forward_prediction_heads(self, src):
# # #         num_1, bs, dim = src[0].shape
# # #         num_2, _, _ = src[1].shape
# # #         num_3, _, _ = src[2].shape
# # #
# # #         feature_1 = src[0].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_1)), int(numpy.sqrt(num_1)))  ####(32, 32)
# # #         feature_2 = src[1].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_2)), int(numpy.sqrt(num_2)))  ####(32, 32)
# # #         feature_3 = src[2].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_3)), int(numpy.sqrt(num_3)))  ####(32, 32)
# # #
# # #         final_fuse = self.final_fuse(torch.cat((feature_1, feature_2, feature_3), dim=1))
# # #         outputs_mask = self.final_predict(final_fuse)
# # #
# # #         results = {
# # #                     "predictions_mask": outputs_mask
# # #         }
# # #         return results
# # #
# # #
# # # @register_decoder
# # # def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
# # #     return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
# #
# #
# # ################################  原来版本的网络结构  ########################################################################
# #         for i in range(len(query_multi_scale)):
# #             ref_feature = ref_multiscale_feature[i]  ### (2, 1, 1024, 32, 32), (2, 11, 1, 32, 32)  (2, 11, 1024, 32, 32)
# #
# #             support_sets = torch.cat((ref_feature, ref_mask_list[i]), dim=1)
# #             support_sets = self.fr_fusion[i](support_sets)
# #
# #             bs, d, h, w = support_sets.shape
# #             support_sets = support_sets.view(bs, d, -1).permute(2, 0, 1)  ##### N, B, D
# #             support_list.append(support_sets)
# #             src.append(self.linear_list[i](query_multi_scale[i]).view(bs, d, -1).permute(2, 0, 1))
# #
# #         for i in range(self.num_layers):
# #             level_index = i % self.num_feature_levels
# #             src_mask_features = src[level_index]
# #             spatial_tokens = support_list[level_index]
# #
# #             output_pos, avg_attn = self.transformer_cross_attention_layers[i](
# #                             src_mask_features, spatial_tokens,
# #                             memory_mask=None,
# #                             memory_key_padding_mask=None,
# #                             pos=None, query_pos=None
# #                         )
# #
# #             y = self.transformer_ffn_layers[i](output_pos)
# #             src[level_index] = y
# #
# #         results = self.forward_prediction_heads(src)
# #         return results
# #
# #     def forward_prediction_heads(self, src):
# #         num_1, bs, dim = src[0].shape
# #         num_2, _, _ = src[1].shape
# #         num_3, _, _ = src[2].shape
# #
# #         feature_1 = src[0].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_1)), int(numpy.sqrt(num_1)))  ####(32, 32)
# #         feature_2 = src[1].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_2)), int(numpy.sqrt(num_2)))  ####(32, 32)
# #         feature_3 = src[2].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_3)), int(numpy.sqrt(num_3)))  ####(32, 32)
# #
# #         final_fuse = self.final_fuse(torch.cat((feature_1, feature_2, feature_3), dim=1))
# #         outputs_mask = self.final_predict(final_fuse)
# #
# #         results = {
# #                 "predictions_mask": outputs_mask
# #             }
# #         return results
# #
# #
# # @register_decoder
# # def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
# #     return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
#
#
#
# # ####################################### cross attention + direct downsample refmask + dinov2 + comprehensive #############################
# # from typing import Optional
# #
# # import numpy
# # import torch
# # from torch import nn, Tensor
# # from torch.nn import functional as F
# #
# # from timm.models.layers import trunc_normal_
# # from detectron2.layers import Conv2d
# # import fvcore.nn.weight_init as weight_init
# #
# # from .utils.utils import rand_sample, prepare_features
# # from .utils.attn import MultiheadAttention
# # from .utils.attention_data_struct import AttentionDataStruct
# # from .registry import register_decoder
# # from ...utils import configurable
# # from ...modules import PositionEmbeddingSine
# # from einops import rearrange
# #
# #
# # class PatchMerging(nn.Module):
# #     r""" Patch Merging Layer.
# #
# #     Args:
# #         input_resolution (tuple[int]): Resolution of input feature.
# #         dim (int): Number of input channels.
# #         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
# #     """
# #
# #     def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
# #         super().__init__()
# #         self.input_resolution = input_resolution
# #         self.dim = dim
# #         self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
# #         self.norm = norm_layer(4 * dim)
# #         self.linear = nn.Linear(2 * dim, 512)
# #
# #     def forward(self, x):
# #         """
# #         x: B, H*W, C
# #         """
# #         H, W = self.input_resolution
# #         B, L, C = x.shape
# #         assert L == H * W, "input feature has wrong size"
# #         assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
# #
# #         x = x.view(B, H, W, C)
# #
# #         x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
# #         x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
# #         x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
# #         x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
# #         x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
# #         x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
# #
# #         x = self.norm(x)
# #         x = self.reduction(x)
# #         x = self.linear(x)
# #
# #         return x
# #
# #
# # class PatchExpand(nn.Module):
# #     def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
# #         super().__init__()
# #         self.input_resolution = input_resolution
# #         self.dim = dim
# #         self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
# #         self.linear = nn.Linear(dim // dim_scale, 512)
# #         self.norm = norm_layer(dim // dim_scale)
# #
# #     def forward(self, x):
# #         """
# #         x: B, H*W, C
# #         """
# #         H, W = self.input_resolution
# #         x = self.expand(x)
# #         B, L, C = x.shape
# #         assert L == H * W, "input feature has wrong size"
# #
# #         x = x.view(B, H, W, C)
# #         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
# #
# #         x = self.linear(x)
# #         x = x.view(B, -1, C//4)
# #         x= self.norm(x)
# #         return x
# #
# # class SelfAttentionLayer(nn.Module):
# #
# #     def __init__(self, d_model, nhead, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt,
# #                      tgt_mask: Optional[Tensor] = None,
# #                      tgt_key_padding_mask: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None):
# #         q = k = self.with_pos_embed(tgt, query_pos)
# #         tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)[0]
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #
# #         return tgt
# #
# #     def forward_pre(self, tgt,
# #                     tgt_mask: Optional[Tensor] = None,
# #                     tgt_key_padding_mask: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):
# #         tgt2 = self.norm(tgt)
# #         q = k = self.with_pos_embed(tgt2, query_pos)
# #         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)[0]
# #         tgt = tgt + self.dropout(tgt2)
# #
# #         return tgt
# #
# #     def forward(self, tgt,
# #                 tgt_mask: Optional[Tensor] = None,
# #                 tgt_key_padding_mask: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, tgt_mask,
# #                                     tgt_key_padding_mask, query_pos)
# #         return self.forward_post(tgt, tgt_mask,
# #                                  tgt_key_padding_mask, query_pos)
# #
# #
# # class CrossAttentionLayer(nn.Module):
# #
# #     def __init__(self, d_model, nhead, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt, memory,
# #                      memory_mask: Optional[Tensor] = None,
# #                      memory_key_padding_mask: Optional[Tensor] = None,
# #                      pos: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None):
# #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
# #                                              key=self.with_pos_embed(memory, pos),
# #                                              value=memory, attn_mask=memory_mask,
# #                                              key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt, avg_attn
# #
# #     def forward_pre(self, tgt, memory,
# #                     memory_mask: Optional[Tensor] = None,
# #                     memory_key_padding_mask: Optional[Tensor] = None,
# #                     pos: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):
# #         tgt2 = self.norm(tgt)
# #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
# #                                              key=self.with_pos_embed(memory, pos),
# #                                              value=memory, attn_mask=memory_mask,
# #                                              key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout(tgt2)
# #
# #         return tgt, avg_attn
# #
# #     def forward(self, tgt, memory,
# #                 memory_mask: Optional[Tensor] = None,
# #                 memory_key_padding_mask: Optional[Tensor] = None,
# #                 pos: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, memory, memory_mask,
# #                                     memory_key_padding_mask, pos, query_pos)
# #         return self.forward_post(tgt, memory, memory_mask,
# #                                  memory_key_padding_mask, pos, query_pos)
# #
# #
# # class FFNLayer(nn.Module):
# #
# #     def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         # Implementation of Feedforward model
# #         self.linear1 = nn.Linear(d_model, dim_feedforward)
# #         self.dropout = nn.Dropout(dropout)
# #         self.linear2 = nn.Linear(dim_feedforward, d_model)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt):
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt
# #
# #     def forward_pre(self, tgt):
# #         tgt2 = self.norm(tgt)
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
# #         tgt = tgt + self.dropout(tgt2)
# #         return tgt
# #
# #     def forward(self, tgt):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt)
# #         return self.forward_post(tgt)
# #
# #
# # def _get_activation_fn(activation):
# #     """Return an activation function given a string"""
# #     if activation == "relu":
# #         return F.relu
# #     if activation == "gelu":
# #         return F.gelu
# #     if activation == "glu":
# #         return F.glu
# #     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
# #
# #
# # class MLP(nn.Module):
# #     """ Very simple multi-layer perceptron (also called FFN)"""
# #
# #     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
# #         super().__init__()
# #         self.num_layers = num_layers
# #         h = [hidden_dim] * (num_layers - 1)
# #         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
# #
# #     def forward(self, x):
# #         for i, layer in enumerate(self.layers):
# #             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
# #         return x
# #
# #
# # class MultiScaleMaskedTransformerDecoder(nn.Module):
# #     _version = 2
# #
# #     @configurable
# #     def __init__(
# #             self,
# #             lang_encoder: nn.Module,
# #             in_channels,
# #             mask_classification=True,
# #             *,
# #             hidden_dim: int,
# #             dim_proj: int,
# #             num_queries: int,
# #             contxt_len: int,
# #             nheads: int,
# #             dim_feedforward: int,
# #             dec_layers: int,
# #             pre_norm: bool,
# #             mask_dim: int,
# #             task_switch: dict,
# #             enforce_input_project: bool,
# #             max_spatial_len: int,
# #             attn_arch: dict,
# #     ):
# #
# #         super().__init__()
# #         assert mask_classification, "Only support mask classification model"
# #         self.mask_classification = mask_classification
# #
# #         # positional encoding
# #         N_steps = hidden_dim // 2
# #         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
# #
# #         # define Transformer decoder here
# #         self.num_heads = nheads
# #         self.num_layers = dec_layers
# #         self.contxt_len = contxt_len
# #         self.transformer_self_attention_layers = nn.ModuleList()
# #         self.transformer_cross_attention_layers = nn.ModuleList()
# #         self.transformer_ffn_layers = nn.ModuleList()
# #         self.predict = nn.Conv2d(512, 134, 3, 1)
# #
# #         self.patch_embed = PatchMerging((32, 32), dim=1024)
# #         self.patch_expand = PatchExpand((32, 32), dim=1024)
# #         self.linear = nn.Conv2d(1024, 512, 1)
# #
# #         self.conv1 = nn.Conv2d(512, 512, 2, 2)
# #         self.conv2 = nn.ConvTranspose2d(512, 512, 2, 2)
# #         self.conv3 = nn.Conv2d(512, 512, 2, 1)
# #
# #         self.final_fuse = nn.Sequential(
# #             nn.Conv2d(512 * 3, 512, 3, 1, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU())
# #
# #         self.fusion= nn.ModuleList()
# #         for _ in range(3):
# #             self.fusion.append(nn.Sequential(
# #             nn.Conv2d(512 + 3, 512, 3, 1, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU()
# #         ))
# #
# #         for _ in range(self.num_layers):
# #             self.transformer_self_attention_layers.append(
# #                 SelfAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #             self.transformer_cross_attention_layers.append(
# #                 CrossAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #             self.transformer_ffn_layers.append(
# #                 FFNLayer(
# #                     d_model=hidden_dim,
# #                     dim_feedforward=dim_feedforward,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #         self.decoder_norm = nn.LayerNorm(hidden_dim)
# #
# #         self.num_queries = num_queries
# #         # learnable query features
# #         self.query_feat = nn.Embedding(num_queries, hidden_dim)
# #         # learnable query p.e.
# #         self.query_embed = nn.Embedding(num_queries, hidden_dim)
# #         # learnable positive negative indicator
# #         self.pn_indicator = nn.Embedding(2, hidden_dim)
# #
# #         # level embedding (we always use 3 scales)
# #         self.num_feature_levels = 3
# #         self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
# #         self.input_proj = nn.ModuleList()
# #
# #         for _ in range(self.num_feature_levels):
# #             if in_channels != hidden_dim or enforce_input_project:
# #                 self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
# #                 weight_init.c2_xavier_fill(self.input_proj[-1])
# #             else:
# #                 self.input_proj.append(nn.Sequential())
# #
# #         self.task_switch = task_switch
# #         self.query_index = {}
# #
# #         # output FFNs
# #         self.lang_encoder = lang_encoder
# #         if self.task_switch['mask']:
# #             self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
# #
# #         self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
# #         trunc_normal_(self.class_embed, std=.02)
# #
# #         if task_switch['spatial']:
# #             self.mask_sptial_embed = nn.ParameterList(
# #                 [nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(3)])
# #             trunc_normal_(self.mask_sptial_embed[0], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[1], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[2], std=.02)
# #
# #             self.max_spatial_len = max_spatial_len
# #
# #             num_spatial_memories = attn_arch['SPATIAL_MEMORIES']
# #             self.spatial_embed = nn.Embedding(num_spatial_memories, hidden_dim)
# #             self.spatial_featured = nn.Embedding(num_spatial_memories, hidden_dim)
# #
# #         # build AttentionDataStruct
# #         attn_arch['NUM_LAYERS'] = self.num_layers
# #         self.attention_data = AttentionDataStruct(attn_arch, task_switch)
# #
# #
# #     @classmethod
# #     def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
# #         ret = {}
# #
# #         ret["lang_encoder"] = lang_encoder
# #         ret["in_channels"] = in_channels
# #         ret["mask_classification"] = mask_classification
# #
# #         enc_cfg = cfg['MODEL']['ENCODER']
# #         dec_cfg = cfg['MODEL']['DECODER']
# #
# #         ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
# #         ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
# #         ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
# #         ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
# #
# #         # Transformer parameters:
# #         ret["nheads"] = dec_cfg['NHEADS']
# #         ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']
# #
# #         assert dec_cfg['DEC_LAYERS'] >= 1
# #         ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
# #         ret["pre_norm"] = dec_cfg['PRE_NORM']
# #         ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
# #         ret["mask_dim"] = enc_cfg['MASK_DIM']
# #         ret["task_switch"] = extra['task_switch']
# #         ret["max_spatial_len"] = dec_cfg['MAX_SPATIAL_LEN']
# #
# #         # attn data struct
# #         ret["attn_arch"] = cfg['ATTENTION_ARCH']
# #
# #         return ret
# #
# #     def forward(self, ref_information, query_information, extra={}, task='seg'):
# #
# #         query_multi_scale = query_information
# #         ref_multiscale_feature, ref_mask = ref_information
# #
# #         assert len(query_multi_scale) == self.num_feature_levels;
# #         spatial_extra_flag = 'spatial_query_pos_mask' in extra.keys() or task == 'refimg'
# #         grounding_extra_flag = 'grounding_tokens' in extra.keys()
# #         visual_extra_flag = 'visual_query_pos' in extra.keys()
# #         audio_extra_flag = 'audio_tokens' in extra.keys()
# #         spatial_memory_flag = 'prev_mask' in extra.keys()
# #         flags = {"spatial": spatial_extra_flag, "grounding": grounding_extra_flag,
# #                          "memories_spatial": spatial_memory_flag, "visual": visual_extra_flag, "audio": audio_extra_flag}
# #         self.attention_data.reset(flags, task, extra)
# #
# #
# #         # src, pos, size_list = prepare_features(query_multi_scale, self.num_feature_levels, self.pe_layer, self.input_proj,
# #         #                                                 self.level_embed)
# #
# #         spatial_list = [64, 32, 16]
# #         ref_mask_list = []
# #
# #         for i in range(len(spatial_list)):
# #             ref_mask_scale = F.interpolate(ref_mask.tensor, (spatial_list[i], spatial_list[i]), mode='nearest')
# #             ref_mask_list.append(ref_mask_scale)
# #
# #
# #         # _, bs, _ = src[0].shape
# #         support_list = []
# #         src = []
# #
# #         for i in range(len(query_multi_scale)):
# #             bs, d, h, w = ref_multiscale_feature[i].shape
# #             if i == 0:
# #                 ref_feature = ref_multiscale_feature[i].reshape(bs, d, -1).transpose(-1, -2)
# #                 ref_feature = self.patch_expand(ref_feature)
# #
# #                 query_feature = query_multi_scale[i].reshape(bs, d, -1).transpose(-1, -2)
# #                 query_feature = self.patch_expand(query_feature).permute(1, 0, 2)
# #
# #                 ref_feature = ref_feature.reshape(bs, h * 2, w * 2, -1).permute(0, 3, 1, 2)
# #             elif i == 2:
# #                 ref_feature = ref_multiscale_feature[i].reshape(bs, d, -1).transpose(-1, -2)
# #                 ref_feature = self.patch_embed(ref_feature)
# #
# #                 query_feature = query_multi_scale[i].reshape(bs, d, -1).transpose(-1, -2)
# #                 query_feature = self.patch_embed(query_feature).permute(1, 0, 2)
# #
# #                 ref_feature = ref_feature.reshape(bs, h // 2, w // 2, -1).permute(0, 3, 1, 2)
# #
# #             else:
# #                 ref_feature = self.linear(ref_multiscale_feature[i])
# #                 query_feature = self.linear(query_multi_scale[i]).view(bs, -1, h*w).permute(2, 0, 1)
# #
# #
# #             support_sets = torch.cat((ref_feature, ref_mask_list[i]), dim=1)
# #             support_sets = self.fusion[i](support_sets)
# #
# #             bs, d, h, w = support_sets.shape
# #             support_sets = support_sets.view(bs, d, -1).permute(2, 0, 1)
# #             support_list.append(support_sets)
# #             src.append(query_feature)
# #
# #         for i in range(self.num_layers):
# #             level_index = i % self.num_feature_levels
# #             src_mask_features = src[level_index]
# #             spatial_tokens = support_list[level_index]
# #
# #             output_pos, avg_attn = self.transformer_cross_attention_layers[i](
# #                             src_mask_features, spatial_tokens,
# #                             memory_mask=None,
# #                             memory_key_padding_mask=None,
# #                             pos=None, query_pos=None
# #                         )
# #             y = self.transformer_ffn_layers[i](output_pos)
# #             src[level_index] = y
# #
# #         results = self.forward_prediction_heads(src)
# #         return results
# #
# #     def forward_prediction_heads(self, src):
# #         ####### 利用多尺度融合模块进行融合
# #         num_1, bs, dim = src[0].shape
# #         num_2, _, _ = src[1].shape
# #         num_3, _, _ = src[2].shape
# #
# #         feature_1 = src[0].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_1)), int(numpy.sqrt(num_1)))  ####(16, 16)
# #         feature_2 = src[1].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_2)), int(numpy.sqrt(num_2)))  ####(32, 32)
# #         feature_3 = src[2].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_3)), int(numpy.sqrt(num_3)))  ####(64, 64)
# #
# #         feature_1_aug = F.interpolate(feature_1, (64, 64), align_corners=True, mode='bilinear')
# #         feature_2_aug = F.interpolate(feature_2, (64, 64), align_corners=True, mode='bilinear')
# #         feature_3_aug = F.interpolate(feature_3, (64, 64), align_corners=True, mode='bilinear')
# #
# #         final_fuse = self.final_fuse(torch.cat((feature_1_aug, feature_2_aug, feature_3_aug), dim=1))
# #         outputs_mask = self.predict(final_fuse)
# #
# #         results = {
# #                 "predictions_mask": outputs_mask
# #             }
# #         return results
# #
# #
# # @register_decoder
# # def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
# #     return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
#
# ########################################## cross attention + direct downsample ref mask #####################################
# # from typing import Optional
# #
# # import numpy
# # import torch
# # from torch import nn, Tensor
# # from torch.nn import functional as F
# #
# # from timm.models.layers import trunc_normal_
# # from detectron2.layers import Conv2d
# # import fvcore.nn.weight_init as weight_init
# #
# # from .utils.utils import rand_sample, prepare_features
# # from .utils.attn import MultiheadAttention
# # from .utils.attention_data_struct import AttentionDataStruct
# # from .registry import register_decoder
# # from ...utils import configurable
# # from ...modules import PositionEmbeddingSine
# #
# # class MLP_1(nn.Module):
# #
# #     def __init__(self, dim):
# #         super(MLP_1, self).__init__()
# #         self.fc1 = nn.Linear(dim, dim * 4)
# #         self.fc2 = nn.Linear(dim * 4, dim)
# #         self.act = nn.functional.gelu
# #         self.dropout = nn.Dropout(0.1)
# #
# #     def forward(self, x):
# #         x = self.fc1(x)
# #         x = self.act(x)
# #         x = self.dropout(x)
# #         x = self.fc2(x)
# #         x = self.dropout(x)
# #         return x
# #
# #
# # class MultiScaleAtten(nn.Module):
# #     def __init__(self, dim):
# #         super(MultiScaleAtten, self).__init__()
# #         self.qkv_linear = nn.Linear(dim, dim * 3)
# #         self.softmax = nn.Softmax(dim=-1)
# #         self.proj = nn.Linear(dim, dim)
# #         self.num_head = 8
# #         self.scale = (dim // self.num_head) ** 0.5
# #
# #     def forward(self, x):
# #         B, num_blocks, _, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
# #         qkv = self.qkv_linear(x).reshape(B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(4, 0, 1, 2, 5, 3, 6).contiguous()  # (3, B, num_block, num_block, head, N, C)
# #         q, k, v = qkv[0], qkv[1], qkv[2]
# #         atten = q @ k.transpose(-1, -2).contiguous()
# #         atten = self.softmax(atten)
# #         atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
# #         atten_value = self.proj(atten_value)  # (B, num_block, num_block, N, C)
# #         return atten_value
# #
# #
# # class InterTransBlock(nn.Module):
# #     def __init__(self, dim):
# #         super(InterTransBlock, self).__init__()
# #         self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
# #         self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
# #         self.Attention = MultiScaleAtten(dim)
# #         self.FFN = MLP_1(dim)
# #
# #     def forward(self, x):
# #         h = x  # (B, N, H)
# #         x = self.SlayerNorm_1(x)
# #
# #         x = self.Attention(x)  # padding 到right_size
# #         x = h + x
# #
# #         h = x
# #         x = self.SlayerNorm_2(x)
# #
# #         x = self.FFN(x)
# #         x = h + x
# #
# #         return x
# #
# #
# # class SpatialAwareTrans(nn.Module):
# #     def __init__(self, dim=256, num=1):  # (224*64, 112*128, 56*256, 28*256, 14*512) dim = 256
# #         super(SpatialAwareTrans, self).__init__()
# #         self.ini_win_size = 2
# #         self.channels = [512, 512, 512, 512]
# #         self.dim = dim
# #         self.depth = 4
# #         self.fc_module = nn.ModuleList()
# #         self.fc_rever_module = nn.ModuleList()
# #         self.num = num
# #         for i in range(self.depth):
# #             self.fc_module.append(nn.Linear(self.channels[i], self.dim))
# #
# #         for i in range(self.depth):
# #             self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))
# #
# #         self.group_attention = []
# #         for i in range(self.num):
# #             self.group_attention.append(InterTransBlock(dim))
# #         self.group_attention = nn.Sequential(*self.group_attention)
# #         self.split_list = [8 * 8, 4 * 4, 2 * 2, 1 * 1]
# #
# #         ### 窗口大小划分分别为 28:2, 56:4, 112:8
# #
# #     def forward(self, x):
# #         x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
# #         for j, item in enumerate(x):
# #             B, H, W, C = item.shape
# #             win_size = self.ini_win_size ** (self.depth - j - 1)
# #             item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4,
# #                                                                                                 5).contiguous()
# #             item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
# #             x[j] = item
# #         x = tuple(x)
# #         x = torch.cat(x, dim=-2)  # (B, H // win, W // win, N, C)
# #         # Scale fusion
# #         for i in range(self.num):
# #             x = self.group_attention[i](x)  # (B, H // win_size, W // win_size, win_size*win_size, C)
# #
# #         x = torch.split(x, self.split_list, dim=-2)
# #         x = list(x)
# #         # patch reversion
# #         for j, item in enumerate(x):
# #             B, num_blocks, _, N, C = item.shape
# #             win_size = self.ini_win_size ** (self.depth - j - 1)
# #             item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4,
# #                                                                                           5).contiguous().reshape(B,
# #                                                                                                                   num_blocks * win_size,
# #                                                                                                                   num_blocks * win_size,
# #                                                                                                                   C)
# #             item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
# #             x[j] = item
# #         return x
# #
# #
# # class SelfAttentionLayer(nn.Module):
# #
# #     def __init__(self, d_model, nhead, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt,
# #                      tgt_mask: Optional[Tensor] = None,
# #                      tgt_key_padding_mask: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None):
# #         q = k = self.with_pos_embed(tgt, query_pos)
# #         tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)[0]
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #
# #         return tgt
# #
# #     def forward_pre(self, tgt,
# #                     tgt_mask: Optional[Tensor] = None,
# #                     tgt_key_padding_mask: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):
# #         tgt2 = self.norm(tgt)
# #         q = k = self.with_pos_embed(tgt2, query_pos)
# #         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)[0]
# #         tgt = tgt + self.dropout(tgt2)
# #
# #         return tgt
# #
# #     def forward(self, tgt,
# #                 tgt_mask: Optional[Tensor] = None,
# #                 tgt_key_padding_mask: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, tgt_mask,
# #                                     tgt_key_padding_mask, query_pos)
# #         return self.forward_post(tgt, tgt_mask,
# #                                  tgt_key_padding_mask, query_pos)
# #
# #
# # class CrossAttentionLayer(nn.Module):
# #
# #     def __init__(self, d_model, nhead, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt, memory,
# #                      memory_mask: Optional[Tensor] = None,
# #                      memory_key_padding_mask: Optional[Tensor] = None,
# #                      pos: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None):
# #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
# #                                              key=self.with_pos_embed(memory, pos),
# #                                              value=memory, attn_mask=memory_mask,
# #                                              key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt, avg_attn
# #
# #     def forward_pre(self, tgt, memory,
# #                     memory_mask: Optional[Tensor] = None,
# #                     memory_key_padding_mask: Optional[Tensor] = None,
# #                     pos: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):
# #         tgt2 = self.norm(tgt)
# #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
# #                                              key=self.with_pos_embed(memory, pos),
# #                                              value=memory, attn_mask=memory_mask,
# #                                              key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout(tgt2)
# #
# #         return tgt, avg_attn
# #
# #     def forward(self, tgt, memory,
# #                 memory_mask: Optional[Tensor] = None,
# #                 memory_key_padding_mask: Optional[Tensor] = None,
# #                 pos: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, memory, memory_mask,
# #                                     memory_key_padding_mask, pos, query_pos)
# #         return self.forward_post(tgt, memory, memory_mask,
# #                                  memory_key_padding_mask, pos, query_pos)
# #
# #
# # class FFNLayer(nn.Module):
# #
# #     def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         # Implementation of Feedforward model
# #         self.linear1 = nn.Linear(d_model, dim_feedforward)
# #         self.dropout = nn.Dropout(dropout)
# #         self.linear2 = nn.Linear(dim_feedforward, d_model)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt):
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt
# #
# #     def forward_pre(self, tgt):
# #         tgt2 = self.norm(tgt)
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
# #         tgt = tgt + self.dropout(tgt2)
# #         return tgt
# #
# #     def forward(self, tgt):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt)
# #         return self.forward_post(tgt)
# #
# #
# # def _get_activation_fn(activation):
# #     """Return an activation function given a string"""
# #     if activation == "relu":
# #         return F.relu
# #     if activation == "gelu":
# #         return F.gelu
# #     if activation == "glu":
# #         return F.glu
# #     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
# #
# #
# # class MLP(nn.Module):
# #     """ Very simple multi-layer perceptron (also called FFN)"""
# #
# #     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
# #         super().__init__()
# #         self.num_layers = num_layers
# #         h = [hidden_dim] * (num_layers - 1)
# #         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
# #
# #     def forward(self, x):
# #         for i, layer in enumerate(self.layers):
# #             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
# #         return x
# #
# #
# # class MultiScaleMaskedTransformerDecoder(nn.Module):
# #     _version = 2
# #
# #     @configurable
# #     def __init__(
# #             self,
# #             lang_encoder: nn.Module,
# #             in_channels,
# #             mask_classification=True,
# #             *,
# #             hidden_dim: int,
# #             dim_proj: int,
# #             num_queries: int,
# #             contxt_len: int,
# #             nheads: int,
# #             dim_feedforward: int,
# #             dec_layers: int,
# #             pre_norm: bool,
# #             mask_dim: int,
# #             task_switch: dict,
# #             enforce_input_project: bool,
# #             max_spatial_len: int,
# #             attn_arch: dict,
# #     ):
# #
# #         super().__init__()
# #         assert mask_classification, "Only support mask classification model"
# #         self.mask_classification = mask_classification
# #
# #         # positional encoding
# #         N_steps = hidden_dim // 2
# #         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
# #
# #         # define Transformer decoder here
# #         self.num_heads = nheads
# #         self.num_layers = dec_layers
# #         self.contxt_len = contxt_len
# #         self.transformer_self_attention_layers = nn.ModuleList()
# #         self.transformer_cross_attention_layers = nn.ModuleList()
# #         self.transformer_ffn_layers = nn.ModuleList()
# #         self.fusion= nn.ModuleList()
# #
# #         for _ in range(self.num_layers + 1):
# #             self.fusion.append(nn.Sequential(
# #             nn.Conv2d(512 + 3, 512, 3, 1, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU()
# #         ))
# #         self.final_fuse = nn.Sequential(
# #             nn.Conv2d(512 * 2, 512, 3, 1, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU())
# #
# #         self.predict = nn.Conv2d(512, 134, 3, 1)
# #
# #         for _ in range(self.num_layers):
# #             self.transformer_self_attention_layers.append(
# #                 SelfAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #             self.transformer_cross_attention_layers.append(
# #                 CrossAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #             self.transformer_ffn_layers.append(
# #                 FFNLayer(
# #                     d_model=hidden_dim,
# #                     dim_feedforward=dim_feedforward,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #         self.decoder_norm = nn.LayerNorm(hidden_dim)
# #
# #         self.num_queries = num_queries
# #         # learnable query features
# #         self.query_feat = nn.Embedding(num_queries, hidden_dim)
# #         # learnable query p.e.
# #         self.query_embed = nn.Embedding(num_queries, hidden_dim)
# #         # learnable positive negative indicator
# #         self.pn_indicator = nn.Embedding(2, hidden_dim)
# #
# #         # level embedding (we always use 3 scales)
# #         self.num_feature_levels = 3
# #         self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
# #         self.input_proj = nn.ModuleList()
# #
# #         for _ in range(self.num_feature_levels):
# #             if in_channels != hidden_dim or enforce_input_project:
# #                 self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
# #                 weight_init.c2_xavier_fill(self.input_proj[-1])
# #             else:
# #                 self.input_proj.append(nn.Sequential())
# #
# #         self.task_switch = task_switch
# #         self.query_index = {}
# #
# #         # output FFNs
# #         self.lang_encoder = lang_encoder
# #         if self.task_switch['mask']:
# #             self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
# #
# #         self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
# #         trunc_normal_(self.class_embed, std=.02)
# #
# #         if task_switch['spatial']:
# #             self.mask_sptial_embed = nn.ParameterList(
# #                 [nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(3)])
# #             trunc_normal_(self.mask_sptial_embed[0], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[1], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[2], std=.02)
# #
# #             self.max_spatial_len = max_spatial_len
# #
# #             num_spatial_memories = attn_arch['SPATIAL_MEMORIES']
# #             self.spatial_embed = nn.Embedding(num_spatial_memories, hidden_dim)
# #             self.spatial_featured = nn.Embedding(num_spatial_memories, hidden_dim)
# #
# #         # build AttentionDataStruct
# #         attn_arch['NUM_LAYERS'] = self.num_layers
# #         self.attention_data = AttentionDataStruct(attn_arch, task_switch)
# #
# #         self.norm1 = nn.LayerNorm(512)
# #         self.norm2 = nn.LayerNorm(512)
# #         self.win_size = 1
# #         self.m = nn.MaxPool2d(kernel_size=(self.win_size, self.win_size), stride=self.win_size, return_indices=True)
# #         self.pos_embedding = nn.Conv2d(512, 512, self.win_size, self.win_size)
# #         self.inter_trans = SpatialAwareTrans(dim=512)
# #
# #     @classmethod
# #     def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
# #         ret = {}
# #
# #         ret["lang_encoder"] = lang_encoder
# #         ret["in_channels"] = in_channels
# #         ret["mask_classification"] = mask_classification
# #
# #         enc_cfg = cfg['MODEL']['ENCODER']
# #         dec_cfg = cfg['MODEL']['DECODER']
# #
# #         ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
# #         ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
# #         ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
# #         ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
# #
# #         # Transformer parameters:
# #         ret["nheads"] = dec_cfg['NHEADS']
# #         ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']
# #
# #         assert dec_cfg['DEC_LAYERS'] >= 1
# #         ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
# #         ret["pre_norm"] = dec_cfg['PRE_NORM']
# #         ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
# #         ret["mask_dim"] = enc_cfg['MASK_DIM']
# #         ret["task_switch"] = extra['task_switch']
# #         ret["max_spatial_len"] = dec_cfg['MAX_SPATIAL_LEN']
# #
# #         # attn data struct
# #         ret["attn_arch"] = cfg['ATTENTION_ARCH']
# #
# #         return ret
# #
# #     def forward(self, ref_information, query_information, extra={}, task='seg'):
# #         # query_mask_feature, query_multi_scale = query_information
# #         # ref_feature, ref_multiscale_feature, ref_mask_feature, ref_mask_multiscale_feature = ref_information
# #         query_mask_feature, query_multi_scale = query_information
# #         ref_feature, ref_multiscale_feature, ref_mask = ref_information
# #
# #         assert len(query_multi_scale) == self.num_feature_levels;
# #         spatial_extra_flag = 'spatial_query_pos_mask' in extra.keys() or task == 'refimg'
# #         grounding_extra_flag = 'grounding_tokens' in extra.keys()
# #         visual_extra_flag = 'visual_query_pos' in extra.keys()
# #         audio_extra_flag = 'audio_tokens' in extra.keys()
# #         spatial_memory_flag = 'prev_mask' in extra.keys()
# #         flags = {"spatial": spatial_extra_flag, "grounding": grounding_extra_flag,
# #                          "memories_spatial": spatial_memory_flag, "visual": visual_extra_flag, "audio": audio_extra_flag}
# #         self.attention_data.reset(flags, task, extra)
# #         src, pos, size_list = prepare_features(query_multi_scale, self.num_feature_levels, self.pe_layer, self.input_proj,
# #                                                         self.level_embed)
# #
# #         ########  对ref_mask进行多尺度缩放处理  ###########################
# #         spatial_list = [14, 28, 56]
# #         ref_mask_list = []
# #
# #         for i in range(len(spatial_list)):
# #             ref_mask_scale = F.interpolate(ref_mask.tensor, (spatial_list[i], spatial_list[i]), mode='nearest')
# #             ref_mask_list.append(ref_mask_scale)
# #
# #         bs, d, h, w = query_mask_feature.shape
# #         query_mask_feature = query_mask_feature.view(bs, d, -1).permute(2, 0, 1)
# #
# #         _, bs, _ = src[0].shape
# #         support_list = []
# #
# #         for i in range(len(src)):
# #             support_sets = torch.cat((ref_multiscale_feature[i], ref_mask_list[i]), dim=1)
# #             support_sets = self.fusion[i + 1](support_sets)
# #             bs, d, h, w = support_sets.shape
# #             support_sets = support_sets.view(bs, d, -1).permute(2, 0, 1)
# #             support_list.append(support_sets)
# #
# #         for i in range(self.num_layers):
# #             level_index = i % self.num_feature_levels
# #             src_mask_features = src[level_index]
# #             spatial_tokens = support_list[level_index]
# #
# #             output_pos, avg_attn = self.transformer_cross_attention_layers[i](
# #                             src_mask_features, spatial_tokens,
# #                             memory_mask=None,
# #                             memory_key_padding_mask=None,
# #                             pos=None, query_pos=None
# #                         )
# #             y = self.transformer_ffn_layers[i](output_pos)
# #             src[level_index] = y
# #
# #         results = self.forward_prediction_heads(src, query_mask_feature)
# #         return results
# #
# #     def forward_prediction_heads(self, src, mask_features):
# #         ####### 利用多尺度融合模块进行融合
# #         num_1, bs, dim = src[0].shape
# #         num_2, _, _ = src[1].shape
# #         num_3, _, _ = src[2].shape
# #         num_4, _, _ = mask_features.shape
# #
# #         feature_1 = src[0].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_1)), int(numpy.sqrt(num_1)))  ####(14, 14)
# #         feature_2 = src[1].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_2)), int(numpy.sqrt(num_2)))  ####(28, 28)
# #         feature_3 = src[2].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_3)), int(numpy.sqrt(num_3)))  ####(56, 56)
# #         mask_features = mask_features.permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_4)), int(numpy.sqrt(num_4))) ###(112,112)
# #
# #         fuse_feature = self.inter_trans([mask_features, feature_3, feature_2, feature_1])
# #         feature1_aug, feature2_aug, feature3_aug, mask_aug = fuse_feature
# #
# #         mask_aug = F.interpolate(mask_aug, (112, 112), align_corners=True, mode='bilinear')
# #
# #         final_fuse = self.final_fuse(torch.cat((feature1_aug, mask_aug), dim=1))
# #         outputs_mask = self.predict(final_fuse)
# #
# #         results = {
# #                 "predictions_mask": outputs_mask
# #             }
# #         return results
# #
# # @register_decoder
# # def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
# #     return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
# #
#
# #
# # ########################################## cross attention + ref mask encoding #####################################
# # from typing import Optional
# #
# # import numpy
# # import torch
# # from torch import nn, Tensor
# # from torch.nn import functional as F
# #
# # from timm.models.layers import trunc_normal_
# # from detectron2.layers import Conv2d
# # import fvcore.nn.weight_init as weight_init
# #
# # from .utils.utils import rand_sample, prepare_features
# # from .utils.attn import MultiheadAttention
# # from .utils.attention_data_struct import AttentionDataStruct
# # from .registry import register_decoder
# # from ...utils import configurable
# # from ...modules import PositionEmbeddingSine
# #
# # class MLP_1(nn.Module):
# #
# #     def __init__(self, dim):
# #         super(MLP_1, self).__init__()
# #         self.fc1 = nn.Linear(dim, dim * 4)
# #         self.fc2 = nn.Linear(dim * 4, dim)
# #         self.act = nn.functional.gelu
# #         self.dropout = nn.Dropout(0.1)
# #
# #     def forward(self, x):
# #         x = self.fc1(x)
# #         x = self.act(x)
# #         x = self.dropout(x)
# #         x = self.fc2(x)
# #         x = self.dropout(x)
# #         return x
# #
# #
# # class MultiScaleAtten(nn.Module):
# #     def __init__(self, dim):
# #         super(MultiScaleAtten, self).__init__()
# #         self.qkv_linear = nn.Linear(dim, dim * 3)
# #         self.softmax = nn.Softmax(dim=-1)
# #         self.proj = nn.Linear(dim, dim)
# #         self.num_head = 8
# #         self.scale = (dim // self.num_head) ** 0.5
# #
# #     def forward(self, x):
# #         B, num_blocks, _, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
# #         qkv = self.qkv_linear(x).reshape(B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(4, 0, 1, 2, 5, 3, 6).contiguous()  # (3, B, num_block, num_block, head, N, C)
# #         q, k, v = qkv[0], qkv[1], qkv[2]
# #         atten = q @ k.transpose(-1, -2).contiguous()
# #         atten = self.softmax(atten)
# #         atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
# #         atten_value = self.proj(atten_value)  # (B, num_block, num_block, N, C)
# #         return atten_value
# #
# #
# # class InterTransBlock(nn.Module):
# #     def __init__(self, dim):
# #         super(InterTransBlock, self).__init__()
# #         self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
# #         self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
# #         self.Attention = MultiScaleAtten(dim)
# #         self.FFN = MLP_1(dim)
# #
# #     def forward(self, x):
# #         h = x  # (B, N, H)
# #         x = self.SlayerNorm_1(x)
# #
# #         x = self.Attention(x)  # padding 到right_size
# #         x = h + x
# #
# #         h = x
# #         x = self.SlayerNorm_2(x)
# #
# #         x = self.FFN(x)
# #         x = h + x
# #
# #         return x
# #
# #
# # class SpatialAwareTrans(nn.Module):
# #     def __init__(self, dim=256, num=1):  # (224*64, 112*128, 56*256, 28*256, 14*512) dim = 256
# #         super(SpatialAwareTrans, self).__init__()
# #         self.ini_win_size = 2
# #         self.channels = [512, 512, 512, 512]
# #         self.dim = dim
# #         self.depth = 4
# #         self.fc_module = nn.ModuleList()
# #         self.fc_rever_module = nn.ModuleList()
# #         self.num = num
# #         for i in range(self.depth):
# #             self.fc_module.append(nn.Linear(self.channels[i], self.dim))
# #
# #         for i in range(self.depth):
# #             self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))
# #
# #         self.group_attention = []
# #         for i in range(self.num):
# #             self.group_attention.append(InterTransBlock(dim))
# #         self.group_attention = nn.Sequential(*self.group_attention)
# #         self.split_list = [8 * 8, 4 * 4, 2 * 2, 1 * 1]
# #
# #         ### 窗口大小划分分别为 28:2, 56:4, 112:8
# #
# #     def forward(self, x):
# #         # project channel dimension to 256
# #         x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
# #         # Patch Matching
# #         for j, item in enumerate(x):
# #             B, H, W, C = item.shape
# #             win_size = self.ini_win_size ** (self.depth - j - 1)
# #             item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4,
# #                                                                                                 5).contiguous()
# #             item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
# #             x[j] = item
# #         x = tuple(x)
# #         x = torch.cat(x, dim=-2)  # (B, H // win, W // win, N, C)
# #         # Scale fusion
# #         for i in range(self.num):
# #             x = self.group_attention[i](x)  # (B, H // win_size, W // win_size, win_size*win_size, C)
# #
# #         x = torch.split(x, self.split_list, dim=-2)
# #         x = list(x)
# #         # patch reversion
# #         for j, item in enumerate(x):
# #             B, num_blocks, _, N, C = item.shape
# #             win_size = self.ini_win_size ** (self.depth - j - 1)
# #             item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4,
# #                                                                                           5).contiguous().reshape(B,
# #                                                                                                                   num_blocks * win_size,
# #                                                                                                                   num_blocks * win_size,
# #                                                                                                                   C)
# #             item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
# #             x[j] = item
# #         return x
# #
# #
# # class SelfAttentionLayer(nn.Module):
# #
# #     def __init__(self, d_model, nhead, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt,
# #                      tgt_mask: Optional[Tensor] = None,
# #                      tgt_key_padding_mask: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None):
# #         q = k = self.with_pos_embed(tgt, query_pos)
# #         tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)[0]
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #
# #         return tgt
# #
# #     def forward_pre(self, tgt,
# #                     tgt_mask: Optional[Tensor] = None,
# #                     tgt_key_padding_mask: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):
# #         tgt2 = self.norm(tgt)
# #         q = k = self.with_pos_embed(tgt2, query_pos)
# #         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)[0]
# #         tgt = tgt + self.dropout(tgt2)
# #
# #         return tgt
# #
# #     def forward(self, tgt,
# #                 tgt_mask: Optional[Tensor] = None,
# #                 tgt_key_padding_mask: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, tgt_mask,
# #                                     tgt_key_padding_mask, query_pos)
# #         return self.forward_post(tgt, tgt_mask,
# #                                  tgt_key_padding_mask, query_pos)
# #
# #
# # class CrossAttentionLayer(nn.Module):
# #
# #     def __init__(self, d_model, nhead, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt, memory,
# #                      memory_mask: Optional[Tensor] = None,
# #                      memory_key_padding_mask: Optional[Tensor] = None,
# #                      pos: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None):
# #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
# #                                              key=self.with_pos_embed(memory, pos),
# #                                              value=memory, attn_mask=memory_mask,
# #                                              key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt, avg_attn
# #
# #     def forward_pre(self, tgt, memory,
# #                     memory_mask: Optional[Tensor] = None,
# #                     memory_key_padding_mask: Optional[Tensor] = None,
# #                     pos: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):
# #         tgt2 = self.norm(tgt)
# #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
# #                                              key=self.with_pos_embed(memory, pos),
# #                                              value=memory, attn_mask=memory_mask,
# #                                              key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout(tgt2)
# #
# #         return tgt, avg_attn
# #
# #     def forward(self, tgt, memory,
# #                 memory_mask: Optional[Tensor] = None,
# #                 memory_key_padding_mask: Optional[Tensor] = None,
# #                 pos: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, memory, memory_mask,
# #                                     memory_key_padding_mask, pos, query_pos)
# #         return self.forward_post(tgt, memory, memory_mask,
# #                                  memory_key_padding_mask, pos, query_pos)
# #
# #
# # class FFNLayer(nn.Module):
# #
# #     def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         # Implementation of Feedforward model
# #         self.linear1 = nn.Linear(d_model, dim_feedforward)
# #         self.dropout = nn.Dropout(dropout)
# #         self.linear2 = nn.Linear(dim_feedforward, d_model)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt):
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt
# #
# #     def forward_pre(self, tgt):
# #         tgt2 = self.norm(tgt)
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
# #         tgt = tgt + self.dropout(tgt2)
# #         return tgt
# #
# #     def forward(self, tgt):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt)
# #         return self.forward_post(tgt)
# #
# #
# # def _get_activation_fn(activation):
# #     """Return an activation function given a string"""
# #     if activation == "relu":
# #         return F.relu
# #     if activation == "gelu":
# #         return F.gelu
# #     if activation == "glu":
# #         return F.glu
# #     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
# #
# #
# # class MLP(nn.Module):
# #     """ Very simple multi-layer perceptron (also called FFN)"""
# #
# #     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
# #         super().__init__()
# #         self.num_layers = num_layers
# #         h = [hidden_dim] * (num_layers - 1)
# #         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
# #
# #     def forward(self, x):
# #         for i, layer in enumerate(self.layers):
# #             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
# #         return x
# #
# #
# # class MultiScaleMaskedTransformerDecoder(nn.Module):
# #     _version = 2
# #
# #     @configurable
# #     def __init__(
# #             self,
# #             lang_encoder: nn.Module,
# #             in_channels,
# #             mask_classification=True,
# #             *,
# #             hidden_dim: int,
# #             dim_proj: int,
# #             num_queries: int,
# #             contxt_len: int,
# #             nheads: int,
# #             dim_feedforward: int,
# #             dec_layers: int,
# #             pre_norm: bool,
# #             mask_dim: int,
# #             task_switch: dict,
# #             enforce_input_project: bool,
# #             max_spatial_len: int,
# #             attn_arch: dict,
# #     ):
# #
# #         super().__init__()
# #         assert mask_classification, "Only support mask classification model"
# #         self.mask_classification = mask_classification
# #
# #         # positional encoding
# #         N_steps = hidden_dim // 2
# #         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
# #
# #         # define Transformer decoder here
# #         self.num_heads = nheads
# #         self.num_layers = dec_layers
# #         self.contxt_len = contxt_len
# #         self.transformer_self_attention_layers = nn.ModuleList()
# #         self.transformer_cross_attention_layers = nn.ModuleList()
# #         self.transformer_ffn_layers = nn.ModuleList()
# #         self.fusion= nn.ModuleList()
# #
# #         for _ in range(self.num_layers + 1):
# #             self.fusion.append(nn.Sequential(
# #             nn.Conv2d(512 * 2, 512, 3, 1, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU()
# #         ))
# #         self.final_fuse = nn.Sequential(
# #             nn.Conv2d(512 * 2, 512, 3, 1, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU())
# #
# #         self.predict = nn.Conv2d(512, 134, 3, 1)
# #
# #         for _ in range(self.num_layers):
# #             self.transformer_self_attention_layers.append(
# #                 SelfAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #             self.transformer_cross_attention_layers.append(
# #                 CrossAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #             self.transformer_ffn_layers.append(
# #                 FFNLayer(
# #                     d_model=hidden_dim,
# #                     dim_feedforward=dim_feedforward,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #         self.decoder_norm = nn.LayerNorm(hidden_dim)
# #
# #         self.num_queries = num_queries
# #         # learnable query features
# #         self.query_feat = nn.Embedding(num_queries, hidden_dim)
# #         # learnable query p.e.
# #         self.query_embed = nn.Embedding(num_queries, hidden_dim)
# #         # learnable positive negative indicator
# #         self.pn_indicator = nn.Embedding(2, hidden_dim)
# #
# #         # level embedding (we always use 3 scales)
# #         self.num_feature_levels = 3
# #         self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
# #         self.input_proj = nn.ModuleList()
# #
# #         for _ in range(self.num_feature_levels):
# #             if in_channels != hidden_dim or enforce_input_project:
# #                 self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
# #                 weight_init.c2_xavier_fill(self.input_proj[-1])
# #             else:
# #                 self.input_proj.append(nn.Sequential())
# #
# #         self.task_switch = task_switch
# #         self.query_index = {}
# #
# #         # output FFNs
# #         self.lang_encoder = lang_encoder
# #         if self.task_switch['mask']:
# #             self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
# #
# #         self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
# #         trunc_normal_(self.class_embed, std=.02)
# #
# #         if task_switch['spatial']:
# #             self.mask_sptial_embed = nn.ParameterList(
# #                 [nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(3)])
# #             trunc_normal_(self.mask_sptial_embed[0], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[1], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[2], std=.02)
# #
# #             self.max_spatial_len = max_spatial_len
# #
# #             num_spatial_memories = attn_arch['SPATIAL_MEMORIES']
# #             self.spatial_embed = nn.Embedding(num_spatial_memories, hidden_dim)
# #             self.spatial_featured = nn.Embedding(num_spatial_memories, hidden_dim)
# #
# #         # build AttentionDataStruct
# #         attn_arch['NUM_LAYERS'] = self.num_layers
# #         self.attention_data = AttentionDataStruct(attn_arch, task_switch)
# #
# #         self.norm1 = nn.LayerNorm(512)
# #         self.norm2 = nn.LayerNorm(512)
# #         self.win_size = 1
# #         self.m = nn.MaxPool2d(kernel_size=(self.win_size, self.win_size), stride=self.win_size, return_indices=True)
# #         self.pos_embedding = nn.Conv2d(512, 512, self.win_size, self.win_size)
# #
# #         # self.extra_cross_atten = CrossAttentionLayer(d_model=hidden_dim,
# #         #             nhead=nheads,
# #         #             dropout=0.0,
# #         #             normalize_before=pre_norm
# #         #         )
# #         # self.extra_ffn = FFNLayer(
# #         #                 d_model=hidden_dim,
# #         #                 dim_feedforward=dim_feedforward,
# #         #                 dropout=0.0,
# #         #                 normalize_before=pre_norm,
# #         #             )
# #
# #         self.inter_trans = SpatialAwareTrans(dim=512)
# #
# #     @classmethod
# #     def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
# #         ret = {}
# #
# #         ret["lang_encoder"] = lang_encoder
# #         ret["in_channels"] = in_channels
# #         ret["mask_classification"] = mask_classification
# #
# #         enc_cfg = cfg['MODEL']['ENCODER']
# #         dec_cfg = cfg['MODEL']['DECODER']
# #
# #         ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
# #         ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
# #         ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
# #         ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
# #
# #         # Transformer parameters:
# #         ret["nheads"] = dec_cfg['NHEADS']
# #         ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']
# #
# #         assert dec_cfg['DEC_LAYERS'] >= 1
# #         ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
# #         ret["pre_norm"] = dec_cfg['PRE_NORM']
# #         ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
# #         ret["mask_dim"] = enc_cfg['MASK_DIM']
# #         ret["task_switch"] = extra['task_switch']
# #         ret["max_spatial_len"] = dec_cfg['MAX_SPATIAL_LEN']
# #
# #         # attn data struct
# #         ret["attn_arch"] = cfg['ATTENTION_ARCH']
# #
# #         return ret
# #
# #     def forward(self, ref_information, query_information, extra={}, task='seg'):
# #         query_mask_feature, query_multi_scale = query_information
# #         ref_feature, ref_multiscale_feature, ref_mask_feature, ref_mask_multiscale_feature = ref_information
# #
# #         assert len(query_multi_scale) == self.num_feature_levels;
# #         spatial_extra_flag = 'spatial_query_pos_mask' in extra.keys() or task == 'refimg'
# #         grounding_extra_flag = 'grounding_tokens' in extra.keys()
# #         visual_extra_flag = 'visual_query_pos' in extra.keys()
# #         audio_extra_flag = 'audio_tokens' in extra.keys()
# #         spatial_memory_flag = 'prev_mask' in extra.keys()
# #         flags = {"spatial": spatial_extra_flag, "grounding": grounding_extra_flag,
# #                          "memories_spatial": spatial_memory_flag, "visual": visual_extra_flag, "audio": audio_extra_flag}
# #         self.attention_data.reset(flags, task, extra)
# #         src, pos, size_list = prepare_features(query_multi_scale, self.num_feature_levels, self.pe_layer, self.input_proj,
# #                                                         self.level_embed)
# #
# #         ref_features = torch.cat((ref_feature, ref_mask_feature), dim=1)
# #         spatial_query_pos = self.fusion[0](ref_features)  #### b, c, h, w
# #
# #         ###### cross attention之前进行数据的处理 #######
# #         bs, d, h, w = query_mask_feature.shape
# #         query_mask_feature = query_mask_feature.view(bs, d, -1).permute(2, 0, 1)
# #         # spatial_query_pos = spatial_query_pos.view(bs, d, -1).permute(2, 0, 1)
# #
# #         # query_mask_feature, _ = self.extra_cross_atten(
# #         #     query_mask_feature, spatial_query_pos,
# #         #     memory_mask=None,
# #         #     memory_key_padding_mask=None,
# #         #     pos=None, query_pos=None
# #         # )
# #         # query_mask_feature = self.extra_ffn(query_mask_feature)
# #
# #         _, bs, _ = src[0].shape
# #         support_list = []
# #
# #         for i in range(len(src)):
# #             support_sets = torch.cat((ref_multiscale_feature[i], ref_mask_multiscale_feature[i]), dim=1)
# #             support_sets = self.fusion[i + 1](support_sets)
# #             bs, d, h, w = support_sets.shape
# #             support_sets = support_sets.view(bs, d, -1).permute(2, 0, 1)
# #             support_list.append(support_sets)
# #
# #         for i in range(self.num_layers):
# #             level_index = i % self.num_feature_levels
# #             src_mask_features = src[level_index]
# #             spatial_tokens = support_list[level_index]
# #
# #             output_pos, avg_attn = self.transformer_cross_attention_layers[i](
# #                             src_mask_features, spatial_tokens,
# #                             memory_mask=None,
# #                             memory_key_padding_mask=None,
# #                             pos=None, query_pos=None
# #                         )
# #             y = self.transformer_ffn_layers[i](output_pos)
# #             src[level_index] = y
# #
# #         results = self.forward_prediction_heads(src, query_mask_feature)
# #         return results
# #
# #     def forward_prediction_heads(self, src, mask_features):
# #         ####### 利用多尺度融合模块进行融合
# #         num_1, bs, dim = src[0].shape
# #         num_2, _, _ = src[1].shape
# #         num_3, _, _ = src[2].shape
# #         num_4, _, _ = mask_features.shape
# #
# #         feature_1 = src[0].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_1)), int(numpy.sqrt(num_1)))  ####(14, 14)
# #         feature_2 = src[1].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_2)), int(numpy.sqrt(num_2)))  ####(28, 28)
# #         feature_3 = src[2].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_3)), int(numpy.sqrt(num_3)))  ####(56, 56)
# #         mask_features = mask_features.permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_4)), int(numpy.sqrt(num_4))) ###(112,112)
# #
# #         fuse_feature = self.inter_trans([mask_features, feature_3, feature_2, feature_1])
# #         feature1_aug, feature2_aug, feature3_aug, mask_aug = fuse_feature
# #
# #         # feature1_aug = F.interpolate(feature2_aug, (112, 112), align_corners=True, mode='bilinear')
# #         # feature2_aug = F.interpolate(feature2_aug, (56, 56), align_corners=True, mode='bilinear')
# #         # feature3_aug = F.interpolate(feature3_aug, (56, 56), align_corners=True, mode='bilinear')
# #         mask_aug = F.interpolate(mask_aug, (112, 112), align_corners=True, mode='bilinear')
# #
# #         final_fuse = self.final_fuse(torch.cat((feature1_aug, mask_aug), dim=1))
# #         outputs_mask = self.predict(final_fuse)
# #
# #         results = {
# #                 "predictions_mask": outputs_mask
# #             }
# #         return results
# #
# # @register_decoder
# # def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
# #     return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
#
# ########################################## cross block + direct downsample ref mask  #####################################
# # #
# # from dataclasses import dataclass
# # from typing import Any, Dict, List, Literal, Optional, Tuple, Union
# # import einops as E
# # import torch
# #
# # from ..nn import CrossConv2d
# # from ..nn import reset_conv2d_parameters
# # from ..nn import Vmap, vmap
# # from ..validation import (Kwargs, as_2tuple, size2t,
# #                          validate_arguments_init)
# # from torch import nn
# # from typing import Optional
# #
# # import numpy
# # import torch
# # from torch import nn, Tensor
# # from torch.nn import functional as F
# #
# # from timm.models.layers import trunc_normal_
# # from detectron2.layers import Conv2d
# # import fvcore.nn.weight_init as weight_init
# #
# # from .utils.utils import rand_sample, prepare_features
# # from .utils.attn import MultiheadAttention
# # from .utils.attention_data_struct import AttentionDataStruct
# # from .registry import register_decoder
# # from ...utils import configurable
# # from ...modules import PositionEmbeddingSine
# # import einops as E
# #
# #
# # def get_nonlinearity(nonlinearity: Optional[str]) -> nn.Module:
# #     if nonlinearity is None:
# #         return nn.Identity()
# #     if nonlinearity == "Softmax":
# #         # For Softmax, we need to specify the channel dimension
# #         return nn.Softmax(dim=1)
# #     if hasattr(nn, nonlinearity):
# #         return getattr(nn, nonlinearity)()
# #     raise ValueError(f"nonlinearity {nonlinearity} not found")
# #
# #
# # @validate_arguments_init
# # @dataclass(eq=False, repr=False)
# # class ConvOp(nn.Sequential):
# #
# #     in_channels: int
# #     out_channels: int
# #     kernel_size: size2t = 3
# #     nonlinearity: Optional[str] = "LeakyReLU"
# #     init_distribution: Optional[str] = "kaiming_normal"
# #     init_bias: Union[None, float, int] = 0.0
# #
# #     def __post_init__(self):
# #         super().__init__()
# #         self.conv = nn.Conv2d(
# #             self.in_channels,
# #             self.out_channels,
# #             kernel_size=self.kernel_size,
# #             padding=self.kernel_size // 2,
# #             padding_mode="zeros",
# #             bias=True,
# #         )
# #
# #         if self.nonlinearity is not None:
# #             self.nonlin = get_nonlinearity(self.nonlinearity)
# #
# #         reset_conv2d_parameters(
# #             self, self.init_distribution, self.init_bias, self.nonlinearity
# #         )
# #
# #
# # @validate_arguments_init
# # @dataclass(eq=False, repr=False)
# # class CrossOp(nn.Module):
# #
# #     in_channels: size2t
# #     out_channels: int
# #     kernel_size: size2t = 3
# #     nonlinearity: Optional[str] = "LeakyReLU"
# #     init_distribution: Optional[str] = "kaiming_normal"
# #     init_bias: Union[None, float, int] = 0.0
# #
# #     def __post_init__(self):
# #         super().__init__()
# #
# #         self.cross_conv = CrossConv2d(
# #             in_channels=as_2tuple(self.in_channels),
# #             out_channels=self.out_channels,
# #             kernel_size=self.kernel_size,
# #             padding=self.kernel_size // 2,
# #         )
# #
# #         if self.nonlinearity is not None:
# #             self.nonlin = get_nonlinearity(self.nonlinearity)
# #
# #         reset_conv2d_parameters(
# #             self, self.init_distribution, self.init_bias, self.nonlinearity
# #         )
# #
# #     def forward(self, target, support):
# #         interaction = self.cross_conv(target, support).squeeze(dim=1)
# #
# #         if self.nonlinearity is not None:
# #             interaction = vmap(self.nonlin, interaction)
# #
# #         new_target = interaction.mean(dim=1, keepdims=True)
# #
# #         return new_target, interaction
# #
# #
# # @validate_arguments_init
# # @dataclass(eq=False, repr=False)
# # class CrossBlock(nn.Module):
# #     in_channels: size2t
# #     cross_features: int
# #     conv_features: Optional[int] = None
# #     cross_kws: Optional[Dict[str, Any]] = None
# #     conv_kws: Optional[Dict[str, Any]] = None
# #
# #     def __post_init__(self):
# #         super().__init__()
# #
# #         conv_features = self.conv_features or self.cross_features
# #         cross_kws = self.cross_kws or {}
# #         conv_kws = self.conv_kws or {}
# #
# #         self.cross = CrossOp(self.in_channels, self.cross_features, **cross_kws)
# #         self.target = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))
# #         self.support = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))
# #
# #     def forward(self, target, support):
# #         target, support = self.cross(target, support)
# #         target = self.target(target)
# #         support = self.support(support)
# #         return target, support
# #
# #
# # class MLP_1(nn.Module):
# #
# #     def __init__(self, dim):
# #         super(MLP_1, self).__init__()
# #         self.fc1 = nn.Linear(dim, dim * 4)
# #         self.fc2 = nn.Linear(dim * 4, dim)
# #         self.act = nn.functional.gelu
# #         self.dropout = nn.Dropout(0.1)
# #
# #     def forward(self, x):
# #         x = self.fc1(x)
# #         x = self.act(x)
# #         x = self.dropout(x)
# #         x = self.fc2(x)
# #         x = self.dropout(x)
# #         return x
# #
# #
# # class MultiScaleAtten(nn.Module):
# #     def __init__(self, dim):
# #         super(MultiScaleAtten, self).__init__()
# #         self.qkv_linear = nn.Linear(dim, dim * 3)
# #         self.softmax = nn.Softmax(dim=-1)
# #         self.proj = nn.Linear(dim, dim)
# #         self.num_head = 8
# #         self.scale = (dim // self.num_head) ** 0.5
# #
# #     def forward(self, x):
# #         B, num_blocks, _, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
# #         qkv = self.qkv_linear(x).reshape(B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(4, 0, 1, 2, 5, 3, 6).contiguous()  # (3, B, num_block, num_block, head, N, C)
# #         q, k, v = qkv[0], qkv[1], qkv[2]
# #         atten = q @ k.transpose(-1, -2).contiguous()
# #         atten = self.softmax(atten)
# #         atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
# #         atten_value = self.proj(atten_value)  # (B, num_block, num_block, N, C)
# #         return atten_value
# #
# #
# # class InterTransBlock(nn.Module):
# #     def __init__(self, dim):
# #         super(InterTransBlock, self).__init__()
# #         self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
# #         self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
# #         self.Attention = MultiScaleAtten(dim)
# #         self.FFN = MLP_1(dim)
# #
# #     def forward(self, x):
# #         h = x  # (B, N, H)
# #         x = self.SlayerNorm_1(x)
# #
# #         x = self.Attention(x)  # padding 到right_size
# #         x = h + x
# #
# #         h = x
# #         x = self.SlayerNorm_2(x)
# #
# #         x = self.FFN(x)
# #         x = h + x
# #
# #         return x
# #
# #
# # class SpatialAwareTrans(nn.Module):
# #     def __init__(self, dim=256, num=1):  # (224*64, 112*128, 56*256, 28*256, 14*512) dim = 256
# #         super(SpatialAwareTrans, self).__init__()
# #         self.ini_win_size = 2
# #         self.channels = [512, 512, 512, 512]
# #         self.dim = dim
# #         self.depth = 4
# #         self.fc_module = nn.ModuleList()
# #         self.fc_rever_module = nn.ModuleList()
# #         self.num = num
# #         for i in range(self.depth):
# #             self.fc_module.append(nn.Linear(self.channels[i], self.dim))
# #
# #         for i in range(self.depth):
# #             self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))
# #
# #         self.group_attention = []
# #         for i in range(self.num):
# #             self.group_attention.append(InterTransBlock(dim))
# #         self.group_attention = nn.Sequential(*self.group_attention)
# #         self.split_list = [8 * 8, 4 * 4, 2 * 2, 1 * 1]
# #
# #         ### 窗口大小划分分别为 28:2, 56:4, 112:8
# #
# #     def forward(self, x):
# #         # project channel dimension to 256
# #         x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
# #         # Patch Matching
# #         for j, item in enumerate(x):
# #             B, H, W, C = item.shape
# #             win_size = self.ini_win_size ** (self.depth - j - 1)
# #             item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4,
# #                                                                                                 5).contiguous()
# #             item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
# #             x[j] = item
# #         x = tuple(x)
# #         x = torch.cat(x, dim=-2)  # (B, H // win, W // win, N, C)
# #         # Scale fusion
# #         for i in range(self.num):
# #             x = self.group_attention[i](x)  # (B, H // win_size, W // win_size, win_size*win_size, C)
# #
# #         x = torch.split(x, self.split_list, dim=-2)
# #         x = list(x)
# #         # patch reversion
# #         for j, item in enumerate(x):
# #             B, num_blocks, _, N, C = item.shape
# #             win_size = self.ini_win_size ** (self.depth - j - 1)
# #             item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4,
# #                                                                                           5).contiguous().reshape(B,
# #                                                                                                                   num_blocks * win_size,
# #                                                                                                                   num_blocks * win_size,
# #                                                                                                                   C)
# #             item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
# #             x[j] = item
# #         return x
# #
# #
# # class SelfAttentionLayer(nn.Module):
# #
# #     def __init__(self, d_model, nhead, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt,
# #                      tgt_mask: Optional[Tensor] = None,
# #                      tgt_key_padding_mask: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None):
# #         q = k = self.with_pos_embed(tgt, query_pos)
# #         tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)[0]
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #
# #         return tgt
# #
# #     def forward_pre(self, tgt,
# #                     tgt_mask: Optional[Tensor] = None,
# #                     tgt_key_padding_mask: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):
# #         tgt2 = self.norm(tgt)
# #         q = k = self.with_pos_embed(tgt2, query_pos)
# #         tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
# #                               key_padding_mask=tgt_key_padding_mask)[0]
# #         tgt = tgt + self.dropout(tgt2)
# #
# #         return tgt
# #
# #     def forward(self, tgt,
# #                 tgt_mask: Optional[Tensor] = None,
# #                 tgt_key_padding_mask: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, tgt_mask,
# #                                     tgt_key_padding_mask, query_pos)
# #         return self.forward_post(tgt, tgt_mask,
# #                                  tgt_key_padding_mask, query_pos)
# #
# #
# # class CrossAttentionLayer(nn.Module):
# #
# #     def __init__(self, d_model, nhead, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt, memory,
# #                      memory_mask: Optional[Tensor] = None,
# #                      memory_key_padding_mask: Optional[Tensor] = None,
# #                      pos: Optional[Tensor] = None,
# #                      query_pos: Optional[Tensor] = None):
# #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
# #                                              key=self.with_pos_embed(memory, pos),
# #                                              value=memory, attn_mask=memory_mask,
# #                                              key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt, avg_attn
# #
# #     def forward_pre(self, tgt, memory,
# #                     memory_mask: Optional[Tensor] = None,
# #                     memory_key_padding_mask: Optional[Tensor] = None,
# #                     pos: Optional[Tensor] = None,
# #                     query_pos: Optional[Tensor] = None):
# #         tgt2 = self.norm(tgt)
# #         tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
# #                                              key=self.with_pos_embed(memory, pos),
# #                                              value=memory, attn_mask=memory_mask,
# #                                              key_padding_mask=memory_key_padding_mask)
# #         tgt = tgt + self.dropout(tgt2)
# #
# #         return tgt, avg_attn
# #
# #     def forward(self, tgt, memory,
# #                 memory_mask: Optional[Tensor] = None,
# #                 memory_key_padding_mask: Optional[Tensor] = None,
# #                 pos: Optional[Tensor] = None,
# #                 query_pos: Optional[Tensor] = None):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt, memory, memory_mask,
# #                                     memory_key_padding_mask, pos, query_pos)
# #         return self.forward_post(tgt, memory, memory_mask,
# #                                  memory_key_padding_mask, pos, query_pos)
# #
# #
# # class FFNLayer(nn.Module):
# #
# #     def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
# #                  activation="relu", normalize_before=False):
# #         super().__init__()
# #         # Implementation of Feedforward model
# #         self.linear1 = nn.Linear(d_model, dim_feedforward)
# #         self.dropout = nn.Dropout(dropout)
# #         self.linear2 = nn.Linear(dim_feedforward, d_model)
# #
# #         self.norm = nn.LayerNorm(d_model)
# #
# #         self.activation = _get_activation_fn(activation)
# #         self.normalize_before = normalize_before
# #
# #         self._reset_parameters()
# #
# #     def _reset_parameters(self):
# #         for p in self.parameters():
# #             if p.dim() > 1:
# #                 nn.init.xavier_uniform_(p)
# #
# #     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
# #         return tensor if pos is None else tensor + pos
# #
# #     def forward_post(self, tgt):
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
# #         tgt = tgt + self.dropout(tgt2)
# #         tgt = self.norm(tgt)
# #         return tgt
# #
# #     def forward_pre(self, tgt):
# #         tgt2 = self.norm(tgt)
# #         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
# #         tgt = tgt + self.dropout(tgt2)
# #         return tgt
# #
# #     def forward(self, tgt):
# #         if self.normalize_before:
# #             return self.forward_pre(tgt)
# #         return self.forward_post(tgt)
# #
# #
# # def _get_activation_fn(activation):
# #     """Return an activation function given a string"""
# #     if activation == "relu":
# #         return F.relu
# #     if activation == "gelu":
# #         return F.gelu
# #     if activation == "glu":
# #         return F.glu
# #     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
# #
# #
# # class MLP(nn.Module):
# #     """ Very simple multi-layer perceptron (also called FFN)"""
# #
# #     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
# #         super().__init__()
# #         self.num_layers = num_layers
# #         h = [hidden_dim] * (num_layers - 1)
# #         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
# #
# #     def forward(self, x):
# #         for i, layer in enumerate(self.layers):
# #             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
# #         return x
# #
# #
# # class MultiScaleMaskedTransformerDecoder(nn.Module):
# #     _version = 2
# #
# #     @configurable
# #     def __init__(
# #             self,
# #             lang_encoder: nn.Module,
# #             in_channels,
# #             mask_classification=True,
# #             *,
# #             hidden_dim: int,
# #             dim_proj: int,
# #             num_queries: int,
# #             contxt_len: int,
# #             nheads: int,
# #             dim_feedforward: int,
# #             dec_layers: int,
# #             pre_norm: bool,
# #             mask_dim: int,
# #             task_switch: dict,
# #             enforce_input_project: bool,
# #             max_spatial_len: int,
# #             attn_arch: dict,
# #     ):
# #
# #         super().__init__()
# #         assert mask_classification, "Only support mask classification model"
# #         self.mask_classification = mask_classification
# #
# #         # positional encoding
# #         N_steps = hidden_dim // 2
# #         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
# #
# #         # define Transformer decoder here
# #         self.num_heads = nheads
# #         self.num_layers = dec_layers
# #         self.contxt_len = contxt_len
# #         self.transformer_self_attention_layers = nn.ModuleList()
# #         self.transformer_cross_attention_layers = nn.ModuleList()
# #         self.transformer_ffn_layers = nn.ModuleList()
# #         self.fusion= nn.ModuleList()
# #
# #         block_kws = dict(cross_kws=dict(nonlinearity=None))
# #         self.cross_blocks = nn.ModuleList()
# #
# #         for _ in range(self.num_layers + 1):
# #             self.cross_blocks.append(CrossBlock([512, 515], 512, 512, **block_kws))
# #
# #         self.final_fuse = nn.Sequential(
# #             nn.Conv2d(512 * 2, 512, 3, 1, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU())
# #
# #         self.predict = nn.Conv2d(512, 134, 3, 1)
# #
# #         for _ in range(self.num_layers):
# #             self.transformer_self_attention_layers.append(
# #                 SelfAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #             self.transformer_cross_attention_layers.append(
# #                 CrossAttentionLayer(
# #                     d_model=hidden_dim,
# #                     nhead=nheads,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #             self.transformer_ffn_layers.append(
# #                 FFNLayer(
# #                     d_model=hidden_dim,
# #                     dim_feedforward=dim_feedforward,
# #                     dropout=0.0,
# #                     normalize_before=pre_norm,
# #                 )
# #             )
# #
# #         self.decoder_norm = nn.LayerNorm(hidden_dim)
# #
# #         self.num_queries = num_queries
# #         # learnable query features
# #         self.query_feat = nn.Embedding(num_queries, hidden_dim)
# #         # learnable query p.e.
# #         self.query_embed = nn.Embedding(num_queries, hidden_dim)
# #         # learnable positive negative indicator
# #         self.pn_indicator = nn.Embedding(2, hidden_dim)
# #
# #         # level embedding (we always use 3 scales)
# #         self.num_feature_levels = 3
# #         self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
# #         self.input_proj = nn.ModuleList()
# #
# #         for _ in range(self.num_feature_levels):
# #             if in_channels != hidden_dim or enforce_input_project:
# #                 self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
# #                 weight_init.c2_xavier_fill(self.input_proj[-1])
# #             else:
# #                 self.input_proj.append(nn.Sequential())
# #
# #         self.task_switch = task_switch
# #         self.query_index = {}
# #
# #         # output FFNs
# #         self.lang_encoder = lang_encoder
# #         if self.task_switch['mask']:
# #             self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
# #
# #         self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
# #         trunc_normal_(self.class_embed, std=.02)
# #
# #         if task_switch['spatial']:
# #             self.mask_sptial_embed = nn.ParameterList(
# #                 [nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(3)])
# #             trunc_normal_(self.mask_sptial_embed[0], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[1], std=.02)
# #             trunc_normal_(self.mask_sptial_embed[2], std=.02)
# #
# #             self.max_spatial_len = max_spatial_len
# #
# #             num_spatial_memories = attn_arch['SPATIAL_MEMORIES']
# #             self.spatial_embed = nn.Embedding(num_spatial_memories, hidden_dim)
# #             self.spatial_featured = nn.Embedding(num_spatial_memories, hidden_dim)
# #
# #         # build AttentionDataStruct
# #         attn_arch['NUM_LAYERS'] = self.num_layers
# #         self.attention_data = AttentionDataStruct(attn_arch, task_switch)
# #
# #         self.norm1 = nn.LayerNorm(512)
# #         self.norm2 = nn.LayerNorm(512)
# #         self.win_size = 1
# #         self.m = nn.MaxPool2d(kernel_size=(self.win_size, self.win_size), stride=self.win_size, return_indices=True)
# #         self.pos_embedding = nn.Conv2d(512, 512, self.win_size, self.win_size)
# #         self.inter_trans = SpatialAwareTrans(dim=512)
# #
# #     @classmethod
# #     def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
# #         ret = {}
# #
# #         ret["lang_encoder"] = lang_encoder
# #         ret["in_channels"] = in_channels
# #         ret["mask_classification"] = mask_classification
# #
# #         enc_cfg = cfg['MODEL']['ENCODER']
# #         dec_cfg = cfg['MODEL']['DECODER']
# #
# #         ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
# #         ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
# #         ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
# #         ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
# #
# #         # Transformer parameters:
# #         ret["nheads"] = dec_cfg['NHEADS']
# #         ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']
# #
# #         assert dec_cfg['DEC_LAYERS'] >= 1
# #         ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
# #         ret["pre_norm"] = dec_cfg['PRE_NORM']
# #         ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
# #         ret["mask_dim"] = enc_cfg['MASK_DIM']
# #         ret["task_switch"] = extra['task_switch']
# #         ret["max_spatial_len"] = dec_cfg['MAX_SPATIAL_LEN']
# #
# #         # attn data struct
# #         ret["attn_arch"] = cfg['ATTENTION_ARCH']
# #
# #         return ret
# #
# #     def forward(self, ref_information, query_information, extra={}, task='seg'):
# #         query_mask_feature, query_multi_scale = query_information
# #         ref_feature, ref_multiscale_feature, ref_mask = ref_information
# #         assert len(query_multi_scale) == self.num_feature_levels;
# #         spatial_extra_flag = 'spatial_query_pos_mask' in extra.keys() or task == 'refimg'
# #         grounding_extra_flag = 'grounding_tokens' in extra.keys()
# #         visual_extra_flag = 'visual_query_pos' in extra.keys()
# #         audio_extra_flag = 'audio_tokens' in extra.keys()
# #         spatial_memory_flag = 'prev_mask' in extra.keys()
# #         flags = {"spatial": spatial_extra_flag, "grounding": grounding_extra_flag,
# #                          "memories_spatial": spatial_memory_flag, "visual": visual_extra_flag, "audio": audio_extra_flag}
# #         self.attention_data.reset(flags, task, extra)
# #
# #         src, pos, size_list = prepare_features(query_multi_scale, self.num_feature_levels, self.pe_layer, self.input_proj,
# #                                                         self.level_embed)
# #
# #         ########  对ref_mask进行多尺度缩放处理  ###########################
# #         spatial_list = [56, 28, 14]
# #         ref_mask_list = []
# #         src_list = []
# #         for i in range(len(spatial_list)):
# #             ref_mask_scale = F.interpolate(ref_mask.tensor, (spatial_list[i], spatial_list[i]), align_corners=True, mode='bilinear')
# #             ref_mask_list.append(ref_mask_scale)
# #
# #         # bs, d, h, w = query_mask_feature.shape
# #         # query_mask_feature = query_mask_feature.view(bs, d, -1).permute(2, 0, 1)
# #
# #         for i in range(len(src)):
# #             support_sets = torch.cat((ref_multiscale_feature[len(src)-i-1], ref_mask_list[i]), dim=1)
# #             support_sets = support_sets.unsqueeze(1)
# #             query_sets = query_multi_scale[len(src)-i-1].unsqueeze(1)
# #
# #             target, support = self.cross_blocks[i](query_sets, support_sets)
# #             src_list.append(target.squeeze(1))
# #
# #         results = self.forward_prediction_heads(src_list, query_mask_feature)
# #         return results
# #
# #     def forward_prediction_heads(self, src, mask_features):
# #         # print(1)
# #         ####### 利用多尺度融合模块进行融合
# #         # num_1, bs, dim = src[0].shape
# #         # num_2, _, _ = src[1].shape
# #         # num_3, _, _ = src[2].shape
# #         # num_4, _, _ = mask_features.shape
# #         #
# #         # feature_1 = src[0].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_1)), int(numpy.sqrt(num_1)))  ####(14, 14)
# #         # feature_2 = src[1].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_2)), int(numpy.sqrt(num_2)))  ####(28, 28)
# #         # feature_3 = src[2].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_3)), int(numpy.sqrt(num_3)))  ####(56, 56)
# #         feature_1 = src[0]
# #         feature_2 = src[1]
# #         feature_3 = src[2]
# #         # mask_features = mask_features.permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_4)), int(numpy.sqrt(num_4))) ###(112,112)
# #
# #         fuse_feature = self.inter_trans([mask_features, feature_1, feature_2, feature_3])
# #         feature1_aug, feature2_aug, feature3_aug, mask_aug = fuse_feature
# #         mask_aug = F.interpolate(mask_aug, (112, 112), align_corners=True, mode='bilinear')
# #
# #         final_fuse = self.final_fuse(torch.cat((feature1_aug, mask_aug), dim=1))
# #         outputs_mask = self.predict(final_fuse)
# #
# #         results = {
# #                 "predictions_mask": outputs_mask
# #             }
# #         return results
# #
# # @register_decoder
# # def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
# #     return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)