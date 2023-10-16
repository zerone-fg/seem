from typing import Optional

import imgviz
import numpy
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from timm.models.layers import trunc_normal_
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init

from .utils.utils import rand_sample, prepare_features
from .utils.attn import MultiheadAttention
from .utils.attention_data_struct import AttentionDataStruct
from .registry import register_decoder
from ...utils import configurable
from ...modules import PositionEmbeddingSine
from einops import rearrange
from PIL import Image
import numpy as np

def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    color_map = imgviz.label_colormap()
    lbl_pil.putpalette(color_map.flatten())
    lbl_pil.save(save_path)
class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                             key=self.with_pos_embed(memory, pos),
                                             value=memory, attn_mask=memory_mask,
                                             key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt, avg_attn

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2, avg_attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                             key=self.with_pos_embed(memory, pos),
                                             value=memory, attn_mask=memory_mask,
                                             key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)

        return tgt, avg_attn

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
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
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.contxt_len = contxt_len

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.linear_list = nn.ModuleList()
        self.src_linear_list = nn.ModuleList()
        for _ in range(3):
            self.linear_list.append(
            nn.Sequential(nn.Linear(1024, 512))
        )
            
        for _ in range(3):
            self.src_linear_list.append(
            nn.Sequential(nn.Linear(1024, 512))
        )
        

        self.final_predict = nn.Sequential(
            nn.BatchNorm2d(1024+5),
            nn.Conv2d(1024+5, 5, 3, 1, 1))

        self.final_fuse = nn.Sequential(
            nn.Conv2d(1024 * 3, 1024, 3, 1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU())

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.pn_indicator = nn.Embedding(2, hidden_dim)

        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.task_switch = task_switch
        self.query_index = {}

        # output FFNs
        self.lang_encoder = lang_encoder
        if self.task_switch['mask']:
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed, std=.02)

        if task_switch['spatial']:
            self.mask_sptial_embed = nn.ParameterList(
                [nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for x in range(3)])
            trunc_normal_(self.mask_sptial_embed[0], std=.02)
            trunc_normal_(self.mask_sptial_embed[1], std=.02)
            trunc_normal_(self.mask_sptial_embed[2], std=.02)

            self.max_spatial_len = max_spatial_len

            num_spatial_memories = attn_arch['SPATIAL_MEMORIES']
            self.spatial_embed = nn.Embedding(num_spatial_memories, hidden_dim)
            self.spatial_featured = nn.Embedding(num_spatial_memories, hidden_dim)

        # build AttentionDataStruct
        attn_arch['NUM_LAYERS'] = self.num_layers
        self.attention_data = AttentionDataStruct(attn_arch, task_switch)


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

        # Transformer parameters:
        ret["nheads"] = dec_cfg['NHEADS']
        ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']

        assert dec_cfg['DEC_LAYERS'] >= 1
        ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
        ret["pre_norm"] = dec_cfg['PRE_NORM']
        ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
        ret["mask_dim"] = enc_cfg['MASK_DIM']
        ret["task_switch"] = extra['task_switch']
        ret["max_spatial_len"] = dec_cfg['MAX_SPATIAL_LEN']

        # attn data struct
        ret["attn_arch"] = cfg['ATTENTION_ARCH']

        return ret

    def forward(self, ref_information, query_information, extra={}, task='seg'):
        query_multi_scale = query_information
        ref_multiscale_feature, ref_mask = ref_information

        assert len(query_multi_scale) == self.num_feature_levels;
        spatial_extra_flag = 'spatial_query_pos_mask' in extra.keys() or task == 'refimg'
        grounding_extra_flag = 'grounding_tokens' in extra.keys()
        visual_extra_flag = 'visual_query_pos' in extra.keys()
        audio_extra_flag = 'audio_tokens' in extra.keys()
        spatial_memory_flag = 'prev_mask' in extra.keys()
        flags = {"spatial": spatial_extra_flag, "grounding": grounding_extra_flag,
                         "memories_spatial": spatial_memory_flag, "visual": visual_extra_flag, "audio": audio_extra_flag}
        self.attention_data.reset(flags, task, extra)

        support_list = []
        src = []
        out_predict_list = []

        bs, c, h, w = ref_mask.tensor.shape
        ref_mask_scale = F.interpolate(ref_mask.tensor, (32, 32), mode='nearest')
        ref_mask_scale = ref_mask_scale.reshape(bs, c, -1).permute(0, 2, 1)

        for i in range(len(query_multi_scale)):
            ref_feature = ref_multiscale_feature[i]
            bs, d, _, _ = ref_feature.shape
            ref_feature = ref_feature.view(bs, d, -1).permute(0, 2, 1) ### bs, n, d
    
            # support_sets = self.linear_list[i](ref_feature)
            support_sets = ref_feature
            support_sets = support_sets.permute(1, 0, 2)  ##### N, B, D
            support_list.append(support_sets)
            
            query_feature = query_multi_scale[i].view(bs, d, -1).permute(0, 2, 1)
            src.append(query_feature.permute(1, 0, 2))
            # src.append(self.src_linear_list[i](query_feature).permute(1, 0, 2))
            # ref_norm = ref_feature / torch.norm(ref_feature, dim=-1, keepdim=True)
            # query_norm = query_feature / torch.norm(query_feature, dim=-1, keepdim=True)

            # avg_atten = (ref_norm @ query_norm.transpose(-1, -2)).softmax(dim=-1)
            # print(avg_atten.max())
        
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            src_mask_features = src[level_index]
            spatial_tokens = support_list[level_index]
        
            output_pos, _ = self.transformer_cross_attention_layers[i](
                src_mask_features, spatial_tokens,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None, query_pos=None
            )
            
            # output_pos = self.layer_norm(output_pos)
            y = self.transformer_ffn_layers[i](output_pos)
            src[level_index] = y

        
        for i in range(len(src)):
            src_mask_features = src[i].permute(1, 0, 2)
            spatial_tokens = support_list[i].permute(1, 0, 2)
            
            src_norm = src_mask_features / (torch.norm(src_mask_features, dim=-1, keepdim=True) + 1e-10)
            spatial_norm = spatial_tokens / (torch.norm(spatial_tokens, dim=-1, keepdim=True) + 1e-10)
            
            avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
            avg_atten = avg_atten.softmax(dim=-1)

            out_predict = avg_atten @ ref_mask_scale
            out_predict_list.append(out_predict)

        results = self.forward_prediction_heads(src, out_predict_list)
        return results


    def forward_prediction_heads(self, src, out_predict_list):
        num_1, bs, dim = src[0].shape
        num_2, _, _ = src[1].shape
        num_3, _, _ = src[2].shape

        feature_1 = src[0].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_1)),
                                                    int(numpy.sqrt(num_1)))  ####(32, 32)
        feature_2 = src[1].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_2)),
                                                    int(numpy.sqrt(num_2)))  ####(32, 32)
        feature_3 = src[2].permute(1, 2, 0).reshape(bs, dim, int(numpy.sqrt(num_3)),
                                                    int(numpy.sqrt(num_3)))  ####(32, 32)

        final_fuse = self.final_fuse(torch.cat((feature_1, feature_2, feature_3), dim=1))

        out_predict = 1 / 3 * (out_predict_list[0] + out_predict_list[1] + out_predict_list[2])

        outputs_mask = self.final_predict(torch.cat((final_fuse, out_predict.reshape(bs, 32, 32, 5).permute(0, 3, 1, 2)), dim=1))

        results = {
            "predictions_mask": outputs_mask
        }
        return results
    
@register_decoder
def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
    return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)

