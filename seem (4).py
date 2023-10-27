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

        # if idx == 6 or idx == 7 or idx == 8:
            # nn.init.constant_(self.fc_out.bias.data, 0)  # nn.init.constant_()表示将偏差定义为常量0
            # nn.init.constant_(self.fc_out.weight.data, 0)  # nn.init.constant_()表示将偏差定义为常量0


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
        self.win_size_list = [8, 8, 4]

    def forward(self, x):
        x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
        # Patch Matching
        for j, item in enumerate(x):
            B, H, W, C = item.shape
            # win_size = self.ini_win_size ** (self.depth - j)
            win_size = self.win_size_list[j]
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
            # win_size = self.ini_win_size ** (self.depth - j - 1)
            win_size = self.win_size_list[j]
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
            nn.BatchNorm2d(512 + 10),
            nn.Conv2d(512 + 10, 10, 3, 1, 1))

        self.final_fuse = nn.Sequential(
            nn.Conv2d(1556, 512, 3, 1, padding=1),
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
        out_predict_3 = F.interpolate(out_predict_list[2].reshape(bs, h3, w3, 10).permute(0, 3, 1, 2), (f_1_aug.shape[-1], f_1_aug.shape[-2]), mode='bilinear',
                                      align_corners=True)

        final_fuse = self.final_fuse(torch.cat((f_1_aug, f_3_aug), dim=1))

        out_predict = 1 / 2 * (out_predict_list[0].reshape(bs, h1, w1, 10).permute(0, 3, 1, 2) + out_predict_3)

        outputs_mask = self.final_predict(
            torch.cat((final_fuse, out_predict), dim=1))

        results = {
            "predictions_mask": outputs_mask
        }
        return results


@register_decoder
def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
    return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)