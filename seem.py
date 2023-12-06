################################################ seem + no cross attention + uppernet + cos change #############
# from typing import Optional
# import sys
# sys.path.append("/data1/paintercoco/xdecoder/body/decoder/")
# import imgviz
# import numpy
# import torch
# from torch import nn, Tensor
# from torch.nn import functional as F
# from .registry import register_decoder
# from ...utils import configurable
# from ...modules import PositionEmbeddingSine
# from PIL import Image
# import numpy as np
# from cluster import CTM
# import matplotlib.pyplot as plt
# from uppernet import FPNHEAD
#
#
# def save_colored_mask(mask, save_path):
#     lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
#     color_map = imgviz.label_colormap()
#     lbl_pil.putpalette(color_map.flatten())
#     lbl_pil.save(save_path)
#
# class FFNLayer(nn.Module):
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
# class MLP_1(nn.Module):
#     def __init__(self, dim):
#         super(MLP_1, self).__init__()
#         self.fc1 = nn.Linear(dim, dim*4)
#         self.fc2 = nn.Linear(dim*4, dim)
#         self.act = nn.functional.gelu
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x
#
# class InterTransBlock(nn.Module):
#     def __init__(self, dim):
#         super(InterTransBlock, self).__init__()
#         self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
#         self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
#         self.Attention = MultiScaleAtten(dim)
#         self.FFN = MLP_1(dim)
#
#     def forward(self, x):
#         h = x  # (B, N, H)
#         x = self.SlayerNorm_1(x)
#
#         x = self.Attention(x)  # padding 到right_size
#         x = h + x
#
#         h = x
#         x = self.SlayerNorm_2(x)
#
#         x = self.FFN(x)
#         x = h + x
#
#         return x
#
#
# class MultiScaleAtten(nn.Module):
#     def __init__(self, dim):
#         super(MultiScaleAtten, self).__init__()
#         self.qkv_linear = nn.Linear(dim, dim * 3)
#         self.softmax = nn.Softmax(dim=-1)
#         self.proj = nn.Linear(dim, dim)
#         self.num_head = 8
#         self.scale = (dim // self.num_head) ** 0.5
#         self.initialize()
#
#     def forward(self, x):
#         B, num_blocks, _, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
#         qkv = self.qkv_linear(x).reshape(B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(4, 0, 1, 2, 5, 3, 6).contiguous()  # (3, B, num_block, num_block, head, N, C)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         atten = q @ k.transpose(-1, -2).contiguous()
#         atten = self.softmax(atten)
#         atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
#         atten_value = self.proj(atten_value)  # (B, num_block, num_block, N, C)
#         return atten_value
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)
#
#
# class SpatialAwareTrans(nn.Module):
#     def __init__(self, dim=256, num=1):
#         super(SpatialAwareTrans, self).__init__()
#         self.ini_win_size = 2
#         self.channels = [512 + 10, 512 + 10, 1024 + 10]
#         self.dim = dim
#         self.depth = 3
#         self.fc_module = nn.ModuleList()
#         self.fc_rever_module = nn.ModuleList()
#         self.num = num
#         for i in range(self.depth):
#             self.fc_module.append(nn.Linear(self.channels[i], self.dim))
#
#         for i in range(self.depth):
#             self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))
#
#         self.group_attention = []
#         for i in range(self.num):
#             self.group_attention.append(InterTransBlock(dim))
#         self.group_attention = nn.Sequential(*self.group_attention)
#         self.split_list = [8 * 8, 8 * 8, 4 * 4]
#         self.win_size_list = [8, 8, 4]
#         self.initialize()
#
#     def forward(self, x):
#         x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
#         for j, item in enumerate(x):
#             B, H, W, C = item.shape
#             win_size = self.win_size_list[j]
#             item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4,
#                                                                                                 5).contiguous()
#             item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
#             x[j] = item
#         x = tuple(x)
#         x = torch.cat(x, dim=-2)
#         for i in range(self.num):
#             x = self.group_attention[i](x)  # (B, H // win_size, W // win_size, win_size*win_size, C)
#
#         x = torch.split(x, self.split_list, dim=-2)
#         x = list(x)
#         for j, item in enumerate(x):
#             B, num_blocks, _, N, C = item.shape
#             win_size = self.win_size_list[j]
#             item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B,
#                                                                                                                   num_blocks * win_size,
#                                                                                                                   num_blocks * win_size,
#                                                                                                                   C)
#             item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
#             x[j] = item
#         return x
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)
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
#         self.num_heads = nheads
#         self.num_layers = 3
#         self.contxt_len = contxt_len
#
#         self.num_feature_levels = 3
#
#         self.transformer_cross_attention_layers = nn.ModuleList()
#         self.transformer_ffn_layers = nn.ModuleList()
#
#         self.final_predict = nn.Sequential(
#             nn.BatchNorm2d(512 + 10),
#             nn.Conv2d(512 + 10, 10, 3, 1, 1))
#
#         self.final_fuse = nn.Sequential(
#             nn.Conv2d(1556, 512, 3, 1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#
#         self.hidden_dim_list = [512, 512, 1024]
#         for idx in range(self.num_layers):
#             self.transformer_cross_attention_layers.append(
#                 CrossAttention(
#                     embed_size=self.hidden_dim_list[idx],
#                     heads=nheads,
#                     idx=idx
#                 )
#             )
#
#             self.transformer_ffn_layers.append(
#                 FFNLayer(
#                     d_model=self.hidden_dim_list[idx],
#                     dim_feedforward=self.hidden_dim_list[idx] * 2,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )
#
#         self.channel_reduction = nn.ModuleList()
#         self.channel_reduction.append(
#             nn.Sequential(
#                 nn.Conv2d(1024, 512, 3, 1),
#                 nn.BatchNorm2d(512)
#             ))
#         self.channel_reduction.append(
#             nn.Sequential(
#                 nn.Conv2d(1024, 512, 3, 1),
#                 nn.BatchNorm2d(512)
#             ))
#
#         self.spatial_list = [64, 64, 32]
#         self.channel_list = [512, 512, 1024]
#
#         self.skip_connection = nn.ModuleList()
#         self.skip_connection.append(
#             nn.Sequential(
#                 nn.Conv2d(self.channel_list[0] + self.channel_list[1], self.channel_list[1], 3, 1, 1),
#                 nn.BatchNorm2d(self.channel_list[1])
#             ))
#         self.skip_connection.append(
#             nn.Sequential(
#                 nn.Conv2d(self.channel_list[1] + self.channel_list[2], self.channel_list[2], 3, 1, 1),
#                 nn.BatchNorm2d(self.channel_list[2])
#             )
#         )
#
#         self.inter_scale = SpatialAwareTrans(256, 1)
#
#         self.output_upscaling = nn.Sequential(
#             nn.ConvTranspose2d((512 + 10), (512 + 10) // 4, kernel_size=2, stride=2),
#             nn.BatchNorm2d((512 + 10) // 4),
#             nn.LeakyReLU(inplace=True),
#             nn.ConvTranspose2d((512 + 10) // 4, (512 + 10) // 8, kernel_size=2, stride=2),
#             nn.BatchNorm2d((512 + 10) // 8),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d((512 + 10) // 8, 10, 3, 1, 1)
#         )
#         self.CTM_module_list = nn.ModuleList()
#
#         self.CTM_module_list.append(CTM(sample_ratio=64, dim_out=512, k=5))
#         self.CTM_module_list.append(CTM(sample_ratio=32, dim_out=512, k=3))
#         self.CTM_module_list.append(CTM(sample_ratio=16, dim_out=1024, k=3))
#
#         self.upper = FPNHEAD(1024, 512)
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
#         ret["attn_arch"] = cfg['ATTENTION_ARCH']
#
#         return ret
#
#     def forward(self, ref_information, query_information, mode):
#         query_multi_scale = query_information
#         ref_multiscale_feature, ref_mask = ref_information
#         out_predict_list = []
#
#         bs_sp, c, h, w = ref_mask.tensor.shape
#         ref_mask_list = []
#         for i in range(self.num_feature_levels):
#             ref_mask_si = F.interpolate(ref_mask.tensor, (self.spatial_list[i], self.spatial_list[i]), mode='nearest')
#             ref_mask_list.append(ref_mask_si.reshape(bs_sp, c, -1).permute(0, 2, 1))
#
#         query_stage_list = []
#         ref_stage_list = []
#         src_copy = []
#
#         for i in range(self.num_feature_levels):
#             if i != 2:
#                 query_multi_si = F.interpolate(self.channel_reduction[i](query_multi_scale[i]),
#                                                (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
#                                                mode='bilinear')
#             else:
#                 query_multi_si = query_multi_scale[i]
#
#             query_stage_list.append(query_multi_si)
#             src_copy.append(query_multi_si.clone())
#
#         for i in range(self.num_feature_levels):
#             if i != 2:
#                 ref_multi_si = F.interpolate(self.channel_reduction[i](ref_multiscale_feature[i]),
#                                              (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
#                                              mode='bilinear')
#             else:
#                 ref_multi_si = ref_multiscale_feature[i]
#             ref_stage_list.append(ref_multi_si)
#
#         spatial_cens = []
#         spatial_params = []
#
#         for level_index in range(self.num_feature_levels):
#             if level_index != 0:
#                 pre_feature = F.interpolate(pre_feature, (query_stage_list[level_index].shape[-1], query_stage_list[level_index].shape[-2]), align_corners=True,
#                                              mode='bilinear')
#                 query_stage_list[level_index] = torch.cat((query_stage_list[level_index], pre_feature), dim=1)
#                 query_stage_list[level_index] = self.skip_connection[level_index - 1](query_stage_list[level_index])
#
#             src_mask_features = query_stage_list[level_index]
#             bs_src, d, _, _ = src_mask_features.shape
#
#             if mode != 'test':
#                 spatial_tokens = ref_stage_list[level_index]
#                 bs_sp, d, _, _ = spatial_tokens.shape
#                 spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)
#
#             else:
#                 spatial_tokens = ref_stage_list[level_index]
#                 bs_sp, d, _, _ = spatial_tokens.shape
#                 spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)  #### spatial_tokens: (bs, N, d)
#                 ref_mask = ref_mask_list[level_index].reshape(bs_sp, -1, 10)
#
#                 token_dict = {'x': spatial_tokens,
#                               'token_num': spatial_tokens.size(1),
#                               'idx_token': torch.arange(spatial_tokens.size(1))[None, :].repeat(
#                                   spatial_tokens.size(0), 1),
#                               'agg_weight': spatial_tokens.new_ones(spatial_tokens.size(0), spatial_tokens.size(1), 1),
#                               'mask': None,
#                               'ref_mask': ref_mask}
#
#                 token_dict_down, _ = self.CTM_module_list[level_index](token_dict)
#                 spatial_tokens = token_dict_down['x']
#                 temp_mask = token_dict_down['ref_mask']
#
#             spatial_tokens = spatial_tokens.reshape(bs_src, -1, d)
#
#             spatial_cens.append(spatial_tokens)
#             spatial_params.append(temp_mask)
#             pre_feature = query_stage_list[level_index]
#
#         if mode != 'test':
#             for i in range(len(query_stage_list)):
#                 src_mask_features = query_stage_list[i]
#                 spatial_tokens = ref_stage_list[i]
#
#                 bs_src, d, _, _ = src_mask_features.shape
#                 bs_sp, d, _, _ = spatial_tokens.shape
#
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#                 spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)
#
#                 src_norm = src_mask_features
#                 spatial_norm = spatial_tokens
#
#                 avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
#                 avg_atten = avg_atten.softmax(dim=-1)
#
#                 out_predict = avg_atten @ ref_mask_list[i]
#                 out_predict_list.append(out_predict)
#
#         else:
#             for i in range(len(query_stage_list)):
#                 src_mask_features = query_stage_list[i]
#                 spatial_tokens = spatial_cens[i]
#
#                 bs_src, d, _, _ = src_mask_features.shape
#                 _, nums, _ = spatial_tokens.shape
#
#                 ref_mask = spatial_params[i].reshape(1, -1, 10)
#
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#
#                 src_norm = src_mask_features
#                 spatial_norm = spatial_tokens
#
#                 avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
#                 avg_atten = avg_atten.softmax(dim=-1)
#
#                 out_predict = avg_atten @ ref_mask
#                 out_predict_list.append(out_predict)
#
#             # out = out_predict_list[0].reshape(bs_src, 64, 64, 10).permute(0, 3, 1, 2)
#             # plt.figure(figsize=(10, 10))
#             # out = out.softmax(dim=1)
#             #
#             # for index in range(10):
#             #     ax = plt.subplot(5, 5, index+1,)
#             #     plt.imshow(out[0][index].cpu().detach().numpy())
#             #     plt.savefig("feature.jpg",dpi=300)
#
#         results = self.forward_prediction_heads(src_copy, out_predict_list, mode)
#         return results
#
#     def forward_prediction_heads(self, src, out_predict_list, mode='train'):
#         bs, dim1, h1, w1 = src[0].shape
#         bs, dim2, h2, w2 = src[1].shape
#         bs, dim3, h3, w3 = src[2].shape
#
#         final_fuse = self.upper([src[0], src[1], src[2]])
#
#         out_predict_3 = F.interpolate(out_predict_list[2].reshape(bs, h3, w3, 10).permute(0, 3, 1, 2), (h2, w2), mode='bilinear',
#                                       align_corners=True)
#         out_predict = 1 / 2 * (out_predict_list[0].reshape(bs, h1, w1, 10).permute(0, 3, 1, 2) + out_predict_3)
#         outputs_mask = self.output_upscaling(
#             torch.cat((final_fuse, out_predict), dim=1)
#         )
#         #
#         # out_save = outputs_mask ### (b, 10, 512, 64, 64)
#         # out_save = out_save.softmax(dim=1)
#         # plt.figure(figsize=(10, 10))
#         # for index in range(10):
#         #     ax = plt.subplot(5, 5, index + 1,)
#         #     plt.imshow(out_save[0][index].cpu().detach().numpy())
#         #     plt.savefig("feature_1.jpg", dpi=300)
#
#         results = {
#             "predictions_mask": outputs_mask
#         }
#
#         return results
#
#
# @register_decoder
# def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
#     return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)

##################################################### seem + no cross attention + sam + cos change  ##############
# from typing import Optional
# import sys
# sys.path.append("/data1/paintercoco/xdecoder/body/decoder/")
# import imgviz
# import numpy
# import torch
# from torch import nn, Tensor
# from torch.nn import functional as F
# from .registry import register_decoder
# from ...utils import configurable
# from ...modules import PositionEmbeddingSine
# from PIL import Image
# import numpy as np
# from cluster import CTM
# import matplotlib.pyplot as plt
# from modeling import MaskDecoder, TwoWayTransformer
# from typing import Any, Dict, List, Tuple
# from modeling.prompt_encoder import PositionEmbeddingRandom
#
#
# def save_colored_mask(mask, save_path):
#     lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
#     color_map = imgviz.label_colormap()
#     lbl_pil.putpalette(color_map.flatten())
#     lbl_pil.save(save_path)
#
#
# class LayerNorm2d(nn.Module):
#     def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(num_channels))
#         self.bias = nn.Parameter(torch.zeros(num_channels))
#         self.eps = eps
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         x = self.weight[:, None, None] * x + self.bias[:, None, None]
#         return x
#
#
# class FFNLayer(nn.Module):
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
# class MLP_1(nn.Module):
#     def __init__(self, dim):
#         super(MLP_1, self).__init__()
#         self.fc1 = nn.Linear(dim, dim*4)
#         self.fc2 = nn.Linear(dim*4, dim)
#         self.act = nn.functional.gelu
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x
#
# class InterTransBlock(nn.Module):
#     def __init__(self, dim):
#         super(InterTransBlock, self).__init__()
#         self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
#         self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
#         self.Attention = MultiScaleAtten(dim)
#         self.FFN = MLP_1(dim)
#
#     def forward(self, x):
#         h = x  # (B, N, H)
#         x = self.SlayerNorm_1(x)
#
#         x = self.Attention(x)  # padding 到right_size
#         x = h + x
#
#         h = x
#         x = self.SlayerNorm_2(x)
#
#         x = self.FFN(x)
#         x = h + x
#
#         return x
#
#
# class MultiScaleAtten(nn.Module):
#     def __init__(self, dim):
#         super(MultiScaleAtten, self).__init__()
#         self.qkv_linear = nn.Linear(dim, dim * 3)
#         self.softmax = nn.Softmax(dim=-1)
#         self.proj = nn.Linear(dim, dim)
#         self.num_head = 8
#         self.scale = (dim // self.num_head) ** 0.5
#         self.initialize()
#
#     def forward(self, x):
#         B, num_blocks, _, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
#         qkv = self.qkv_linear(x).reshape(B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(4, 0, 1, 2, 5, 3, 6).contiguous()  # (3, B, num_block, num_block, head, N, C)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         atten = q @ k.transpose(-1, -2).contiguous()
#         atten = self.softmax(atten)
#         atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
#         atten_value = self.proj(atten_value)  # (B, num_block, num_block, N, C)
#         return atten_value
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)
#
#
# class SpatialAwareTrans(nn.Module):
#     def __init__(self, dim=256, num=1):
#         super(SpatialAwareTrans, self).__init__()
#         self.ini_win_size = 2
#         self.channels = [512 + 10, 512 + 10, 1024 + 10]
#         self.dim = dim
#         self.depth = 3
#         self.fc_module = nn.ModuleList()
#         self.fc_rever_module = nn.ModuleList()
#         self.num = num
#         for i in range(self.depth):
#             self.fc_module.append(nn.Linear(self.channels[i], self.dim))
#
#         for i in range(self.depth):
#             self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))
#
#         self.group_attention = []
#         for i in range(self.num):
#             self.group_attention.append(InterTransBlock(dim))
#         self.group_attention = nn.Sequential(*self.group_attention)
#         self.split_list = [8 * 8, 8 * 8, 4 * 4]
#         self.win_size_list = [8, 8, 4]
#         self.initialize()
#
#     def forward(self, x):
#         x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
#         for j, item in enumerate(x):
#             B, H, W, C = item.shape
#             win_size = self.win_size_list[j]
#             item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4,
#                                                                                                 5).contiguous()
#             item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
#             x[j] = item
#         x = tuple(x)
#         x = torch.cat(x, dim=-2)
#         for i in range(self.num):
#             x = self.group_attention[i](x)  # (B, H // win_size, W // win_size, win_size*win_size, C)
#
#         x = torch.split(x, self.split_list, dim=-2)
#         x = list(x)
#         for j, item in enumerate(x):
#             B, num_blocks, _, N, C = item.shape
#             win_size = self.win_size_list[j]
#             item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B,
#                                                                                                                   num_blocks * win_size,
#                                                                                                                   num_blocks * win_size,
#                                                                                                                   C)
#             item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
#             x[j] = item
#         return x
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)
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
#         self.num_heads = nheads
#         self.num_layers = 6
#         self.contxt_len = contxt_len
#
#         self.num_feature_levels = 3
#
#         self.transformer_cross_attention_layers = nn.ModuleList()
#         self.transformer_ffn_layers = nn.ModuleList()
#
#         self.final_fuse = nn.Sequential(
#             nn.Conv2d(1556, 512, 3, 1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#
#         self.hidden_dim_list = [512, 512, 512, 512, 1024, 1024]
#         for idx in range(self.num_layers):
#             self.transformer_cross_attention_layers.append(
#                 CrossAttention(
#                     embed_size=self.hidden_dim_list[idx],
#                     heads=nheads,
#                     idx=idx
#                 )
#             )
#
#             self.transformer_ffn_layers.append(
#                 FFNLayer(
#                     d_model=self.hidden_dim_list[idx],
#                     dim_feedforward=self.hidden_dim_list[idx] * 2,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )
#
#         self.channel_reduction = nn.ModuleList()
#         self.channel_reduction.append(
#             nn.Sequential(
#                 nn.Conv2d(1024, 512, 3, 1),
#                 nn.BatchNorm2d(512)
#             ))
#         self.channel_reduction.append(
#             nn.Sequential(
#                 nn.Conv2d(1024, 512, 3, 1),
#                 nn.BatchNorm2d(512)
#             ))
#
#         self.spatial_list = [64, 64, 32]
#         self.channel_list = [512, 512, 1024]
#
#         self.skip_connection = nn.ModuleList()
#         self.skip_connection.append(
#             nn.Sequential(
#                 nn.Conv2d(self.channel_list[0] + self.channel_list[1], self.channel_list[1], 3, 1, 1),
#                 nn.BatchNorm2d(self.channel_list[1])
#             ))
#         self.skip_connection.append(
#             nn.Sequential(
#                 nn.Conv2d(self.channel_list[1] + self.channel_list[2], self.channel_list[2], 3, 1, 1),
#                 nn.BatchNorm2d(self.channel_list[2])
#             )
#         )
#
#         self.inter_scale = SpatialAwareTrans(256, 1)
#
#         self.CTM_module_list = nn.ModuleList()
#         self.CTM_module_list.append(CTM(sample_ratio=64, dim_out=512, k=5))
#         self.CTM_module_list.append(CTM(sample_ratio=32, dim_out=512, k=3))
#         self.CTM_module_list.append(CTM(sample_ratio=16, dim_out=1024, k=3))
#
#         self.sam_decoder = MaskDecoder(
#             num_multimask_outputs=10,
#             transformer=TwoWayTransformer(
#                 depth=2,
#                 embedding_dim=512,
#                 mlp_dim=2048,
#                 num_heads=8,
#             ),
#             transformer_dim=512,
#             iou_head_depth=3,
#             iou_head_hidden_dim=512,
#         )
#
#         self.Convs = nn.Sequential(
#             nn.Conv2d(10, 512 // 4, kernel_size=3, stride=1, padding=1),
#             LayerNorm2d(512 // 4),
#             nn.GELU(),
#             nn.Conv2d(512 // 4, 512 // 2, kernel_size=3, stride=1, padding=1),
#             LayerNorm2d(512 // 2),
#             nn.GELU(),
#             nn.Conv2d(512 // 2, 512, kernel_size=1),
#         )
#         self.pe_layer = PositionEmbeddingRandom(256)
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
#         ret["attn_arch"] = cfg['ATTENTION_ARCH']
#
#         return ret
#
#     def forward(self, ref_information, query_information, mode):
#         query_multi_scale = query_information
#         ref_multiscale_feature, ref_mask = ref_information
#         out_predict_list = []
#
#         bs_sp, c, h, w = ref_mask.tensor.shape
#         ref_mask_list = []
#         for i in range(self.num_feature_levels):
#             ref_mask_si = F.interpolate(ref_mask.tensor, (self.spatial_list[i], self.spatial_list[i]), mode='nearest')
#             ref_mask_list.append(ref_mask_si.reshape(bs_sp, c, -1).permute(0, 2, 1))
#
#         query_stage_list = []
#         ref_stage_list = []
#         src_copy = []
#
#         for i in range(self.num_feature_levels):
#             if i != 2:
#                 query_multi_si = F.interpolate(self.channel_reduction[i](query_multi_scale[i]),
#                                                (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
#                                                mode='bilinear')
#             else:
#                 query_multi_si = query_multi_scale[i]
#
#             query_stage_list.append(query_multi_si)
#             src_copy.append(query_multi_si.clone())
#
#         for i in range(self.num_feature_levels):
#             if i != 2:
#                 ref_multi_si = F.interpolate(self.channel_reduction[i](ref_multiscale_feature[i]),
#                                              (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
#                                              mode='bilinear')
#             else:
#                 ref_multi_si = ref_multiscale_feature[i]
#
#             ref_stage_list.append(ref_multi_si)
#
#
#         spatial_cens = []
#         spatial_params = []
#
#         for level_index in range(self.num_feature_levels):
#             if level_index != 0:
#                 pre_feature = F.interpolate(pre_feature, (query_stage_list[level_index].shape[-1], query_stage_list[level_index].shape[-2]), align_corners=True,
#                                              mode='bilinear')
#                 query_stage_list[level_index] = torch.cat((query_stage_list[level_index], pre_feature), dim=1)
#                 query_stage_list[level_index] = self.skip_connection[level_index - 1](query_stage_list[level_index])
#
#             src_mask_features = query_stage_list[level_index]
#             bs_src, d, _, _ = src_mask_features.shape
#
#             if mode != 'test':
#                 spatial_tokens = ref_stage_list[level_index]
#                 bs_sp, d, _, _ = spatial_tokens.shape
#                 spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)
#
#             else:
#                 spatial_tokens = ref_stage_list[level_index]
#                 bs_sp, d, _, _ = spatial_tokens.shape
#                 spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)  #### spatial_tokens: (bs, N, d)
#                 ref_mask = ref_mask_list[level_index].reshape(bs_sp, -1, 10)
#
#                 token_dict = {'x': spatial_tokens,
#                               'token_num': spatial_tokens.size(1),
#                               'idx_token': torch.arange(spatial_tokens.size(1))[None, :].repeat(
#                                   spatial_tokens.size(0), 1),
#                               'agg_weight': spatial_tokens.new_ones(spatial_tokens.size(0), spatial_tokens.size(1), 1),
#                               'mask': None,
#                               'ref_mask': ref_mask}
#
#                 token_dict_down, _ = self.CTM_module_list[level_index](token_dict)
#                 spatial_tokens = token_dict_down['x']
#                 temp_mask = token_dict_down['ref_mask']
#
#             spatial_tokens = spatial_tokens.reshape(bs_src, -1, d)
#
#             spatial_cens.append(spatial_tokens)
#             spatial_params.append(temp_mask)
#             pre_feature = query_stage_list[level_index]
#
#         if mode != 'test':
#             for i in range(len(query_stage_list)):
#                 src_mask_features = query_stage_list[i]
#                 spatial_tokens = ref_stage_list[i]
#
#                 bs_src, d, _, _ = src_mask_features.shape
#                 bs_sp, d, _, _ = spatial_tokens.shape
#
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#                 spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)
#
#                 src_norm = src_mask_features
#                 spatial_norm = spatial_tokens
#
#                 avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
#                 avg_atten = avg_atten.softmax(dim=-1)
#
#                 out_predict = avg_atten @ ref_mask_list[i]
#                 out_predict_list.append(out_predict)
#
#         else:
#             for i in range(len(query_stage_list)):
#                 src_mask_features = query_stage_list[i]
#                 spatial_tokens = spatial_cens[i]
#
#                 bs_src, d, _, _ = src_mask_features.shape
#                 _, nums, _ = spatial_tokens.shape
#
#                 ref_mask = spatial_params[i].reshape(1, -1, 10)
#
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#
#                 src_norm = src_mask_features
#                 spatial_norm = spatial_tokens
#
#                 avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
#                 avg_atten = avg_atten.softmax(dim=-1)
#
#                 out_predict = avg_atten @ ref_mask
#                 out_predict_list.append(out_predict)
#
#         out = out_predict_list[0].reshape(bs_src, 64, 64, 10).permute(0, 3, 1, 2)
#         plt.figure(figsize=(10, 10))
#         out = out.softmax(dim=1)
#
#         # for index in range(10):
#         #     ax = plt.subplot(5, 5, index+1,)
#         #     plt.imshow(out[0][index].cpu().detach().numpy())
#         #     plt.savefig("feature.jpg",dpi=300)
#
#         results = self.forward_prediction_heads(src_copy, out_predict_list, mode)
#         return results
#
#     def forward_prediction_heads(self, src, out_predict_list, mode='train'):
#         bs, dim1, h1, w1 = src[0].shape
#         bs, dim2, h2, w2 = src[1].shape
#         bs, dim3, h3, w3 = src[2].shape
#
#         feature_1 = torch.cat((src[0], out_predict_list[0].reshape(bs, -1, h1, w1)), dim=1)
#         feature_2 = torch.cat((src[1], out_predict_list[1].reshape(bs, -1, h2, w2)), dim=1)
#         feature_3 = torch.cat((src[2], out_predict_list[2].reshape(bs, -1, h3, w3)), dim=1)
#
#         f_1_aug, f_2_aug, f_3_aug = self.inter_scale((feature_1, feature_2, feature_3))
#         f_3_aug = F.interpolate(f_3_aug, (f_1_aug.shape[-1], f_1_aug.shape[-2]), mode='bilinear', align_corners=True)
#         out_predict_3 = F.interpolate(out_predict_list[2].reshape(bs, h3, w3, 10).permute(0, 3, 1, 2), (h2, w2), mode='bilinear',
#                                       align_corners=True)
#         final_fuse = self.final_fuse(torch.cat((f_1_aug, f_3_aug), dim=1))
#         out_predict = 1 / 2 * (out_predict_list[0].reshape(bs, h1, w1, 10).permute(0, 3, 1, 2) + out_predict_3)
#
#         outputs_feature = self.Convs(
#             out_predict
#         )
#
#         curr_embedding = outputs_feature + final_fuse
#
#         low_res_masks, iou_predictions = self.sam_decoder(
#             image_embeddings=curr_embedding,
#             multimask_output=True,
#             image_pe=self.pe_layer.forward((64, 64)).unsqueeze(0),
#             sparse_prompt_embeddings=None,
#             dense_prompt_embeddings=None
#         )
#         masks = postprocess_masks(
#             low_res_masks,
#             input_size=(448, 448),
#             original_size=(448, 448),
#         )
#
#         results = {
#             "predictions_mask": masks
#         }
#         return results
#
#
# @register_decoder
# def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
#     return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
#
#
# def postprocess_masks(
#     masks: torch.Tensor,
#     input_size: Tuple[int, ...],
#     original_size: Tuple[int, ...],
# ) -> torch.Tensor:
#     """
#     Remove padding and upscale masks to the original image size.
#
#     Arguments:
#       masks (torch.Tensor): Batched masks from the mask_decoder,
#         in BxCxHxW format.
#       input_size (tuple(int, int)): The size of the image input to the
#         model, in (H, W) format. Used to remove padding.
#       original_size (tuple(int, int)): The original size of the image
#         before resizing for input to the model, in (H, W) format.
#
#     Returns:
#       (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
#         is given by original_size.
#     """
#     masks = F.interpolate(
#         masks,
#         input_size,
#         mode="bilinear",
#         align_corners=False,
#     )
#     masks = masks[..., : input_size[0], : input_size[1]]
#     masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
#     return masks

# ########################################################## seem + no cross atten ######################################
from typing import Optional
import sys
sys.path.append("/data1/paintercoco/xdecoder/body/decoder/")
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
from cluster import CTM
import matplotlib.pyplot as plt
from uppernet import FPNHEAD


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
        self.initialize()

    def forward(self, x):
        B, num_blocks, _, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
        qkv = self.qkv_linear(x).reshape(B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(4, 0, 1, 2, 5, 3, 6).contiguous()  # (3, B, num_block, num_block, head, N, C)
        q, k, v = qkv[0], qkv[1], qkv[2]
        atten = q @ k.transpose(-1, -2).contiguous()
        atten = self.softmax(atten)
        atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
        atten_value = self.proj(atten_value)  # (B, num_block, num_block, N, C)
        return atten_value

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)


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
        self.initialize()

    def forward(self, x):
        x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
        for j, item in enumerate(x):
            B, H, W, C = item.shape
            win_size = self.win_size_list[j]
            item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4,
                                                                                                5).contiguous()
            item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
            x[j] = item
        x = tuple(x)
        x = torch.cat(x, dim=-2)
        for i in range(self.num):
            x = self.group_attention[i](x)  # (B, H // win_size, W // win_size, win_size*win_size, C)

        x = torch.split(x, self.split_list, dim=-2)
        x = list(x)
        for j, item in enumerate(x):
            B, num_blocks, _, N, C = item.shape
            win_size = self.win_size_list[j]
            item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B,
                                                                                                                  num_blocks * win_size,
                                                                                                                  num_blocks * win_size,
                                                                                                                  C)
            item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
            x[j] = item
        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)


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
        self.num_layers = 3
        self.contxt_len = contxt_len

        self.num_feature_levels = 3

        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.final_predict = nn.Sequential(
            nn.BatchNorm2d(512 + 10),
            nn.Conv2d(512 + 10, 10, 3, 1, 1))

        self.final_fuse = nn.Sequential(
            nn.Conv2d(1536, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.hidden_dim_list = [512, 512, 1024]
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

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d((512 + 10), (512 + 10) // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d((512 + 10) // 4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d((512 + 10) // 4, (512 + 10) // 8, kernel_size=2, stride=2),
            nn.BatchNorm2d((512 + 10) // 8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d((512 + 10) // 8, 10, 3, 1, 1)
        )
        self.CTM_module_list = nn.ModuleList()

        self.CTM_module_list.append(CTM(sample_ratio=64, dim_out=512, k=5))
        self.CTM_module_list.append(CTM(sample_ratio=32, dim_out=512, k=3))
        self.CTM_module_list.append(CTM(sample_ratio=16, dim_out=1024, k=3))

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

    def forward(self, ref_information, query_information, mode):
        query_multi_scale = query_information
        ref_multiscale_feature, ref_mask = ref_information
        out_predict_list = []

        bs_sp, c, h, w = ref_mask.tensor.shape
        ref_mask_list = []
        for i in range(self.num_feature_levels):
            ref_mask_si = F.interpolate(ref_mask.tensor, (self.spatial_list[i], self.spatial_list[i]), mode='nearest')
            ref_mask_list.append(ref_mask_si.reshape(bs_sp, c, -1).permute(0, 2, 1))

        query_stage_list = []
        ref_stage_list = []
        src_copy = []

        for i in range(self.num_feature_levels):
            if i != 2:
                query_multi_si = F.interpolate(self.channel_reduction[i](query_multi_scale[i]),
                                               (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
                                               mode='bilinear')
            else:
                query_multi_si = query_multi_scale[i]

            query_stage_list.append(query_multi_si)
            src_copy.append(query_multi_si.clone())

        for i in range(self.num_feature_levels):
            if i != 2:
                ref_multi_si = F.interpolate(self.channel_reduction[i](ref_multiscale_feature[i]),
                                             (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
                                             mode='bilinear')
            else:
                ref_multi_si = ref_multiscale_feature[i]
            ref_stage_list.append(ref_multi_si)

        spatial_cens = []
        spatial_params = []

        for level_index in range(self.num_feature_levels):
            if level_index != 0:
                pre_feature = F.interpolate(pre_feature, (query_stage_list[level_index].shape[-1], query_stage_list[level_index].shape[-2]), align_corners=True,
                                             mode='bilinear')
                query_stage_list[level_index] = torch.cat((query_stage_list[level_index], pre_feature), dim=1)
                query_stage_list[level_index] = self.skip_connection[level_index - 1](query_stage_list[level_index])

            src_mask_features = query_stage_list[level_index]
            bs_src, d, _, _ = src_mask_features.shape

            if mode != 'test':
                spatial_tokens = ref_stage_list[level_index]
                bs_sp, d, _, _ = spatial_tokens.shape
                spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)

            else:
                spatial_tokens = ref_stage_list[level_index]
                bs_sp, d, _, _ = spatial_tokens.shape
                spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)  #### spatial_tokens: (bs, N, d)
                ref_mask = ref_mask_list[level_index].reshape(bs_sp, -1, 10)

                token_dict = {'x': spatial_tokens,
                              'token_num': spatial_tokens.size(1),
                              'idx_token': torch.arange(spatial_tokens.size(1))[None, :].repeat(
                                  spatial_tokens.size(0), 1),
                              'agg_weight': spatial_tokens.new_ones(spatial_tokens.size(0), spatial_tokens.size(1), 1),
                              'mask': None,
                              'ref_mask': ref_mask}

                token_dict_down, _ = self.CTM_module_list[level_index](token_dict)
                spatial_tokens = token_dict_down['x']
                temp_mask = token_dict_down['ref_mask']

            spatial_tokens = spatial_tokens.reshape(bs_src, -1, d)

            spatial_cens.append(spatial_tokens)
            # spatial_params.append(temp_mask)
            pre_feature = query_stage_list[level_index]

        if mode != 'test':
            for i in range(len(query_stage_list)):
                src_mask_features = query_stage_list[i]
                spatial_tokens = ref_stage_list[i]

                bs_src, d, _, _ = src_mask_features.shape
                bs_sp, d, _, _ = spatial_tokens.shape

                src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
                spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)

                src_norm = src_mask_features
                spatial_norm = spatial_tokens

                avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
                avg_atten = avg_atten.softmax(dim=-1)

                out_predict = avg_atten @ ref_mask_list[i]
                out_predict_list.append(out_predict)

        else:
            for i in range(len(query_stage_list)):
                src_mask_features = query_stage_list[i]
                spatial_tokens = spatial_cens[i]

                bs_src, d, _, _ = src_mask_features.shape
                _, nums, _ = spatial_tokens.shape

                ref_mask = spatial_params[i].reshape(1, -1, 10)

                src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)

                src_norm = src_mask_features
                spatial_norm = spatial_tokens

                avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
                avg_atten = avg_atten.softmax(dim=-1)

                out_predict = avg_atten @ ref_mask
                out_predict_list.append(out_predict)

        results = self.forward_prediction_heads(src_copy, out_predict_list, mode)
        return results

    def forward_prediction_heads(self, src, out_predict_list, mode='train'):
        bs, dim1, h1, w1 = src[0].shape
        bs, dim2, h2, w2 = src[1].shape
        bs, dim3, h3, w3 = src[2].shape

        f_1_aug = src[0]
        f_3_aug = src[2]

        f_3_aug = F.interpolate(f_3_aug, (f_1_aug.shape[-1], f_1_aug.shape[-2]), mode='bilinear', align_corners=True)
        out_predict_3 = F.interpolate(out_predict_list[2].reshape(bs, h3, w3, 10).permute(0, 3, 1, 2), (h2, w2), mode='bilinear',
                                      align_corners=True)

        final_fuse = self.final_fuse(torch.cat((f_1_aug, f_3_aug), dim=1))
        out_predict = 1 / 2 * (out_predict_list[0].reshape(bs, h1, w1, 10).permute(0, 3, 1, 2) + out_predict_3)
        outputs_mask = self.output_upscaling(
            torch.cat((final_fuse, out_predict), dim=1)
        )

        results = {
            "predictions_mask": outputs_mask
        }
        return results


@register_decoder
def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
    return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)

########################################################## seem + no final_fuse ######################################
# from typing import Optional
# import sys
# sys.path.append("/data1/paintercoco/xdecoder/body/decoder/")
# import imgviz
# import numpy
# import torch
# from torch import nn, Tensor
# from torch.nn import functional as F
# from .registry import register_decoder
# from ...utils import configurable
# from ...modules import PositionEmbeddingSine
# from PIL import Image
# import numpy as np
# from cluster import CTM
# import matplotlib.pyplot as plt
#
#
# def save_colored_mask(mask, save_path):
#     lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
#     color_map = imgviz.label_colormap()
#     lbl_pil.putpalette(color_map.flatten())
#     lbl_pil.save(save_path)
#
# class FFNLayer(nn.Module):
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
# class MLP_1(nn.Module):
#     def __init__(self, dim):
#         super(MLP_1, self).__init__()
#         self.fc1 = nn.Linear(dim, dim*4)
#         self.fc2 = nn.Linear(dim*4, dim)
#         self.act = nn.functional.gelu
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x
#
# class InterTransBlock(nn.Module):
#     def __init__(self, dim):
#         super(InterTransBlock, self).__init__()
#         self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
#         self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
#         self.Attention = MultiScaleAtten(dim)
#         self.FFN = MLP_1(dim)
#
#     def forward(self, x):
#         h = x  # (B, N, H)
#         x = self.SlayerNorm_1(x)
#
#         x = self.Attention(x)  # padding 到right_size
#         x = h + x
#
#         h = x
#         x = self.SlayerNorm_2(x)
#
#         x = self.FFN(x)
#         x = h + x
#
#         return x
#
#
# class MultiScaleAtten(nn.Module):
#     def __init__(self, dim):
#         super(MultiScaleAtten, self).__init__()
#         self.qkv_linear = nn.Linear(dim, dim * 3)
#         self.softmax = nn.Softmax(dim=-1)
#         self.proj = nn.Linear(dim, dim)
#         self.num_head = 8
#         self.scale = (dim // self.num_head) ** 0.5
#         self.initialize()
#
#     def forward(self, x):
#         B, num_blocks, _, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
#         qkv = self.qkv_linear(x).reshape(B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(4, 0, 1, 2, 5, 3, 6).contiguous()  # (3, B, num_block, num_block, head, N, C)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         atten = q @ k.transpose(-1, -2).contiguous()
#         atten = self.softmax(atten)
#         atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
#         atten_value = self.proj(atten_value)  # (B, num_block, num_block, N, C)
#         return atten_value
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)
#
#
# class SpatialAwareTrans(nn.Module):
#     def __init__(self, dim=256, num=1):
#         super(SpatialAwareTrans, self).__init__()
#         self.ini_win_size = 2
#         self.channels = [512 + 10, 512 + 10, 1024 + 10]
#         self.dim = dim
#         self.depth = 3
#         self.fc_module = nn.ModuleList()
#         self.fc_rever_module = nn.ModuleList()
#         self.num = num
#         for i in range(self.depth):
#             self.fc_module.append(nn.Linear(self.channels[i], self.dim))
#
#         for i in range(self.depth):
#             self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))
#
#         self.group_attention = []
#         for i in range(self.num):
#             self.group_attention.append(InterTransBlock(dim))
#         self.group_attention = nn.Sequential(*self.group_attention)
#         self.split_list = [8 * 8, 8 * 8, 4 * 4]
#         self.win_size_list = [8, 8, 4]
#         self.initialize()
#
#     def forward(self, x):
#         x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
#         for j, item in enumerate(x):
#             B, H, W, C = item.shape
#             win_size = self.win_size_list[j]
#             item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4,
#                                                                                                 5).contiguous()
#             item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
#             x[j] = item
#         x = tuple(x)
#         x = torch.cat(x, dim=-2)
#         for i in range(self.num):
#             x = self.group_attention[i](x)  # (B, H // win_size, W // win_size, win_size*win_size, C)
#
#         x = torch.split(x, self.split_list, dim=-2)
#         x = list(x)
#         for j, item in enumerate(x):
#             B, num_blocks, _, N, C = item.shape
#             win_size = self.win_size_list[j]
#             item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B,
#                                                                                                                   num_blocks * win_size,
#                                                                                                                   num_blocks * win_size,
#                                                                                                                   C)
#             item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
#             x[j] = item
#         return x
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)
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
#         self.num_heads = nheads
#         self.num_layers = 3
#         self.contxt_len = contxt_len
#
#         self.num_feature_levels = 3
#
#         self.transformer_cross_attention_layers = nn.ModuleList()
#         self.transformer_ffn_layers = nn.ModuleList()
#
#         self.final_predict = nn.Sequential(
#             nn.BatchNorm2d(512 + 10),
#             nn.Conv2d(512 + 10, 10, 3, 1, 1))
#
#         self.final_fuse = nn.Sequential(
#             nn.Conv2d(1536, 512, 3, 1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#
#         self.hidden_dim_list = [512, 512, 1024]
#         for idx in range(self.num_layers):
#             self.transformer_cross_attention_layers.append(
#                 CrossAttention(
#                     embed_size=self.hidden_dim_list[idx],
#                     heads=nheads,
#                     idx=idx
#                 )
#             )
#
#             self.transformer_ffn_layers.append(
#                 FFNLayer(
#                     d_model=self.hidden_dim_list[idx],
#                     dim_feedforward=self.hidden_dim_list[idx] * 2,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )
#
#         self.channel_reduction = nn.ModuleList()
#         self.channel_reduction.append(
#             nn.Sequential(
#                 nn.Conv2d(1024, 512, 3, 1),
#                 nn.BatchNorm2d(512)
#             ))
#         self.channel_reduction.append(
#             nn.Sequential(
#                 nn.Conv2d(1024, 512, 3, 1),
#                 nn.BatchNorm2d(512)
#             ))
#
#         self.spatial_list = [64, 64, 32]
#         self.channel_list = [512, 512, 1024]
#
#         self.skip_connection = nn.ModuleList()
#         self.skip_connection.append(
#             nn.Sequential(
#                 nn.Conv2d(self.channel_list[0] + self.channel_list[1], self.channel_list[1], 3, 1, 1),
#                 nn.BatchNorm2d(self.channel_list[1])
#             ))
#         self.skip_connection.append(
#             nn.Sequential(
#                 nn.Conv2d(self.channel_list[1] + self.channel_list[2], self.channel_list[2], 3, 1, 1),
#                 nn.BatchNorm2d(self.channel_list[2])
#             )
#         )
#
#         self.inter_scale = SpatialAwareTrans(256, 1)
#
#         self.output_upscaling = nn.Sequential(
#             nn.ConvTranspose2d((512 + 10), (512 + 10) // 4, kernel_size=2, stride=2),
#             nn.BatchNorm2d((512 + 10) // 4),
#             nn.LeakyReLU(inplace=True),
#             nn.ConvTranspose2d((512 + 10) // 4, (512 + 10) // 8, kernel_size=2, stride=2),
#             nn.BatchNorm2d((512 + 10) // 8),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d((512 + 10) // 8, 10, 3, 1, 1)
#         )
#         self.CTM_module_list = nn.ModuleList()
#
#         self.CTM_module_list.append(CTM(sample_ratio=64, dim_out=512, k=5))
#         self.CTM_module_list.append(CTM(sample_ratio=32, dim_out=512, k=3))
#         self.CTM_module_list.append(CTM(sample_ratio=16, dim_out=1024, k=3))
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
#         ret["attn_arch"] = cfg['ATTENTION_ARCH']
#
#         return ret
#
#     def forward(self, ref_information, query_information, mode):
#         query_multi_scale = query_information
#         ref_multiscale_feature, ref_mask = ref_information
#         out_predict_list = []
#
#         bs_sp, c, h, w = ref_mask.tensor.shape
#         ref_mask_list = []
#         for i in range(self.num_feature_levels):
#             ref_mask_si = F.interpolate(ref_mask.tensor, (self.spatial_list[i], self.spatial_list[i]), mode='nearest')
#             ref_mask_list.append(ref_mask_si.reshape(bs_sp, c, -1).permute(0, 2, 1))
#
#         query_stage_list = []
#         ref_stage_list = []
#         src_copy = []
#
#         for i in range(self.num_feature_levels):
#             if i != 2:
#                 query_multi_si = F.interpolate(self.channel_reduction[i](query_multi_scale[i]),
#                                                (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
#                                                mode='bilinear')
#             else:
#                 query_multi_si = query_multi_scale[i]
#
#             query_stage_list.append(query_multi_si)
#             src_copy.append(query_multi_si.clone())
#
#         for i in range(self.num_feature_levels):
#             if i != 2:
#                 ref_multi_si = F.interpolate(self.channel_reduction[i](ref_multiscale_feature[i]),
#                                              (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
#                                              mode='bilinear')
#             else:
#                 ref_multi_si = ref_multiscale_feature[i]
#             ref_stage_list.append(ref_multi_si)
#
#         spatial_cens = []
#         spatial_params = []
#
#         for level_index in range(self.num_feature_levels):
#             if level_index != 0:
#                 pre_feature = F.interpolate(pre_feature, (query_stage_list[level_index].shape[-1], query_stage_list[level_index].shape[-2]), align_corners=True,
#                                              mode='bilinear')
#                 query_stage_list[level_index] = torch.cat((query_stage_list[level_index], pre_feature), dim=1)
#                 query_stage_list[level_index] = self.skip_connection[level_index - 1](query_stage_list[level_index])
#
#             for j in range(1):
#                 src_mask_features = query_stage_list[level_index]
#                 bs_src, d, _, _ = src_mask_features.shape
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#
#                 if mode != 'test':
#                     spatial_tokens = ref_stage_list[level_index]
#                     bs_sp, d, _, _ = spatial_tokens.shape
#                     spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)
#                 else:
#                     if j == 0:
#                         spatial_tokens = ref_stage_list[level_index]
#                         bs_sp, d, _, _ = spatial_tokens.shape
#                         spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)  #### spatial_tokens: (bs, N, d)
#                         ref_mask = ref_mask_list[level_index].reshape(bs_sp, -1, 10)
#
#                         token_dict = {'x': spatial_tokens,
#                                       'token_num': spatial_tokens.size(1),
#                                       'idx_token': torch.arange(spatial_tokens.size(1))[None, :].repeat(
#                                           spatial_tokens.size(0), 1),
#                                       'agg_weight': spatial_tokens.new_ones(spatial_tokens.size(0), spatial_tokens.size(1), 1),
#                                       'mask': None,
#                                       'ref_mask': ref_mask}
#
#                         token_dict_down, _ = self.CTM_module_list[level_index](token_dict)
#                         spatial_tokens = token_dict_down['x']
#                         temp_mask = token_dict_down['ref_mask']
#
#                     spatial_tokens = spatial_tokens.reshape(bs_src, -1, d)
#
#                 output_pos = self.transformer_cross_attention_layers[level_index + j](
#                     spatial_tokens, spatial_tokens, src_mask_features,
#                     mask=None
#                 )
#                 y = self.transformer_ffn_layers[level_index + j](output_pos.permute(1, 0, 2)).permute(1, 0, 2)  ### b, n, d
#                 query_stage_list[level_index] = y.reshape(bs_src, self.spatial_list[level_index],
#                                                        self.spatial_list[level_index], d).permute(0, 3, 1, 2)
#             spatial_cens.append(spatial_tokens)
#             spatial_params.append(temp_mask)
#             pre_feature = query_stage_list[level_index]
#
#         if mode != 'test':
#             for i in range(len(query_stage_list)):
#                 src_mask_features = query_stage_list[i]
#                 spatial_tokens = ref_stage_list[i]
#
#                 bs_src, d, _, _ = src_mask_features.shape
#                 bs_sp, d, _, _ = spatial_tokens.shape
#
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#                 spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)
#
#                 src_norm = src_mask_features / (torch.norm(src_mask_features, dim=-1, keepdim=True) + 1e-12)
#                 spatial_norm = spatial_tokens / (torch.norm(spatial_tokens, dim=-1, keepdim=True) + 1e-12)
#
#                 avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
#                 avg_atten = avg_atten.softmax(dim=-1)
#
#                 out_predict = avg_atten @ ref_mask_list[i]
#                 out_predict_list.append(out_predict)
#
#         else:
#             for i in range(len(query_stage_list)):
#                 src_mask_features = query_stage_list[i]
#                 spatial_tokens = spatial_cens[i]
#
#                 bs_src, d, _, _ = src_mask_features.shape
#                 _, nums, _ = spatial_tokens.shape   ##### (1, 1000, d)
#
#                 ref_mask = spatial_params[i].reshape(1, -1, 10)
#
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#                 src_norm = src_mask_features / (torch.norm(src_mask_features, dim=-1, keepdim=True) + 1e-12)
#                 spatial_norm = spatial_tokens / (torch.norm(spatial_tokens, dim=-1, keepdim=True) + 1e-12)
#
#                 avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
#                 avg_atten = avg_atten.softmax(dim=-1)
#
#                 out_predict = avg_atten @ ref_mask
#                 out_predict_list.append(out_predict)
#
#         results = self.forward_prediction_heads(src_copy, out_predict_list, mode)
#         return results
#
#     def forward_prediction_heads(self, src, out_predict_list, mode='train'):
#         bs, dim1, h1, w1 = src[0].shape
#         bs, dim2, h2, w2 = src[1].shape
#         bs, dim3, h3, w3 = src[2].shape
#
#         f_1_aug = src[0]
#         f_3_aug = src[2]
#         f_3_aug = F.interpolate(f_3_aug, (f_1_aug.shape[-1], f_1_aug.shape[-2]), mode='bilinear', align_corners=True)
#         out_predict_3 = F.interpolate(out_predict_list[2].reshape(bs, h3, w3, 10).permute(0, 3, 1, 2), (h2, w2), mode='bilinear',
#                                       align_corners=True)
#
#         final_fuse = self.final_fuse(torch.cat((f_1_aug, f_3_aug), dim=1))
#         out_predict = 1 / 2 * (out_predict_list[0].reshape(bs, h1, w1, 10).permute(0, 3, 1, 2) + out_predict_3)
#         outputs_mask = self.output_upscaling(
#             torch.cat((final_fuse, out_predict), dim=1)
#         )
#         results = {
#             "predictions_mask": outputs_mask
#         }
#         return results
#
#
# @register_decoder
# def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
#     return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)

# # ######################################################### seem + sam #########################################
# from typing import Optional
# import sys
# sys.path.append("/data1/paintercoco/xdecoder/body/decoder/")
# import imgviz
# import numpy
# import torch
# from torch import nn, Tensor
# from torch.nn import functional as F
# from .registry import register_decoder
# from ...utils import configurable
# from ...modules import PositionEmbeddingSine
# from PIL import Image
# import numpy as np
# from cluster import CTM
# import matplotlib.pyplot as plt
# from modeling import MaskDecoder, TwoWayTransformer
# from typing import Any, Dict, List, Tuple
# from modeling.prompt_encoder import PositionEmbeddingRandom
#
# class LayerNorm2d(nn.Module):
#     def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(num_channels))
#         self.bias = nn.Parameter(torch.zeros(num_channels))
#         self.eps = eps
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         x = self.weight[:, None, None] * x + self.bias[:, None, None]
#         return x
#
# def save_colored_mask(mask, save_path):
#     lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
#     color_map = imgviz.label_colormap()
#     lbl_pil.putpalette(color_map.flatten())
#     lbl_pil.save(save_path)
#
# class FFNLayer(nn.Module):
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
# class MLP_1(nn.Module):
#     def __init__(self, dim):
#         super(MLP_1, self).__init__()
#         self.fc1 = nn.Linear(dim, dim*4)
#         self.fc2 = nn.Linear(dim*4, dim)
#         self.act = nn.functional.gelu
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x
#
# class InterTransBlock(nn.Module):
#     def __init__(self, dim):
#         super(InterTransBlock, self).__init__()
#         self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
#         self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
#         self.Attention = MultiScaleAtten(dim)
#         self.FFN = MLP_1(dim)
#
#     def forward(self, x):
#         h = x  # (B, N, H)
#         x = self.SlayerNorm_1(x)
#
#         x = self.Attention(x)  # padding 到right_size
#         x = h + x
#
#         h = x
#         x = self.SlayerNorm_2(x)
#
#         x = self.FFN(x)
#         x = h + x
#
#         return x
#
#
# class MultiScaleAtten(nn.Module):
#     def __init__(self, dim):
#         super(MultiScaleAtten, self).__init__()
#         self.qkv_linear = nn.Linear(dim, dim * 3)
#         self.softmax = nn.Softmax(dim=-1)
#         self.proj = nn.Linear(dim, dim)
#         self.num_head = 8
#         self.scale = (dim // self.num_head) ** 0.5
#         self.initialize()
#
#     def forward(self, x):
#         B, num_blocks, _, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
#         qkv = self.qkv_linear(x).reshape(B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(4, 0, 1, 2, 5, 3, 6).contiguous()  # (3, B, num_block, num_block, head, N, C)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         atten = q @ k.transpose(-1, -2).contiguous()
#         atten = self.softmax(atten)
#         atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
#         atten_value = self.proj(atten_value)  # (B, num_block, num_block, N, C)
#         return atten_value
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)
#
#
# class SpatialAwareTrans(nn.Module):
#     def __init__(self, dim=256, num=1):
#         super(SpatialAwareTrans, self).__init__()
#         self.ini_win_size = 2
#         self.channels = [512 + 10, 512 + 10, 1024 + 10]
#         self.dim = dim
#         self.depth = 3
#         self.fc_module = nn.ModuleList()
#         self.fc_rever_module = nn.ModuleList()
#         self.num = num
#         for i in range(self.depth):
#             self.fc_module.append(nn.Linear(self.channels[i], self.dim))
#
#         for i in range(self.depth):
#             self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))
#
#         self.group_attention = []
#         for i in range(self.num):
#             self.group_attention.append(InterTransBlock(dim))
#         self.group_attention = nn.Sequential(*self.group_attention)
#         self.split_list = [8 * 8, 8 * 8, 4 * 4]
#         self.win_size_list = [8, 8, 4]
#         self.initialize()
#
#     def forward(self, x):
#         x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
#         for j, item in enumerate(x):
#             B, H, W, C = item.shape
#             win_size = self.win_size_list[j]
#             item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4,
#                                                                                                 5).contiguous()
#             item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
#             x[j] = item
#         x = tuple(x)
#         x = torch.cat(x, dim=-2)
#         for i in range(self.num):
#             x = self.group_attention[i](x)  # (B, H // win_size, W // win_size, win_size*win_size, C)
#
#         x = torch.split(x, self.split_list, dim=-2)
#         x = list(x)
#         for j, item in enumerate(x):
#             B, num_blocks, _, N, C = item.shape
#             win_size = self.win_size_list[j]
#             item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B,
#                                                                                                                   num_blocks * win_size,
#                                                                                                                   num_blocks * win_size,
#                                                                                                                   C)
#             item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
#             x[j] = item
#         return x
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)
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
#         self.num_heads = nheads
#         self.num_layers = 6
#         self.contxt_len = contxt_len
#
#         self.num_feature_levels = 3
#
#         self.transformer_cross_attention_layers = nn.ModuleList()
#         self.transformer_ffn_layers = nn.ModuleList()
#
#         self.final_fuse = nn.Sequential(
#             nn.Conv2d(1536, 512, 3, 1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#
#         self.hidden_dim_list = [512, 512, 512, 512, 1024, 1024]
#         for idx in range(self.num_layers):
#             self.transformer_cross_attention_layers.append(
#                 CrossAttention(
#                     embed_size=self.hidden_dim_list[idx],
#                     heads=nheads,
#                     idx=idx
#                 )
#             )
#
#             self.transformer_ffn_layers.append(
#                 FFNLayer(
#                     d_model=self.hidden_dim_list[idx],
#                     dim_feedforward=self.hidden_dim_list[idx] * 2,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )
#
#         self.channel_reduction = nn.ModuleList()
#         self.channel_reduction.append(
#             nn.Sequential(
#                 nn.Conv2d(1024, 512, 3, 1),
#                 nn.BatchNorm2d(512)
#             ))
#         self.channel_reduction.append(
#             nn.Sequential(
#                 nn.Conv2d(1024, 512, 3, 1),
#                 nn.BatchNorm2d(512)
#             ))
#
#         self.spatial_list = [64, 64, 32]
#         self.channel_list = [512, 512, 1024]
#
#         self.skip_connection = nn.ModuleList()
#         self.skip_connection.append(
#             nn.Sequential(
#                 nn.Conv2d(self.channel_list[0] + self.channel_list[1], self.channel_list[1], 3, 1, 1),
#                 nn.BatchNorm2d(self.channel_list[1])
#             ))
#         self.skip_connection.append(
#             nn.Sequential(
#                 nn.Conv2d(self.channel_list[1] + self.channel_list[2], self.channel_list[2], 3, 1, 1),
#                 nn.BatchNorm2d(self.channel_list[2])
#             )
#         )
#
#         self.inter_scale = SpatialAwareTrans(256, 1)
#
#         self.CTM_module_list = nn.ModuleList()
#         self.CTM_module_list.append(CTM(sample_ratio=64, dim_out=512, k=5))
#         self.CTM_module_list.append(CTM(sample_ratio=32, dim_out=512, k=3))
#         self.CTM_module_list.append(CTM(sample_ratio=16, dim_out=1024, k=3))
#
#         self.sam_decoder = MaskDecoder(
#             num_multimask_outputs=10,
#             transformer=TwoWayTransformer(
#                 depth=2,
#                 embedding_dim=512,
#                 mlp_dim=2048,
#                 num_heads=8,
#             ),
#             transformer_dim=512,
#             iou_head_depth=3,
#             iou_head_hidden_dim=512,
#         )
#
#         self.Convs = nn.Sequential(
#             nn.Conv2d(10, 512 // 4, kernel_size=3, stride=1, padding=1),
#             LayerNorm2d(512 // 4),
#             nn.GELU(),
#             nn.Conv2d(512 // 4, 512 // 2, kernel_size=3, stride=1, padding=1),
#             LayerNorm2d(512 // 2),
#             nn.GELU(),
#             nn.Conv2d(512 // 2, 512, kernel_size=1),
#         )
#
#         self.pe_layer = PositionEmbeddingRandom(256)
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
#         ret["attn_arch"] = cfg['ATTENTION_ARCH']
#
#         return ret
#
#     def forward(self, ref_information, query_information, mode):
#         query_multi_scale = query_information
#         ref_multiscale_feature, ref_mask = ref_information
#         out_predict_list = []
#
#         bs_sp, c, h, w = ref_mask.tensor.shape
#         ref_mask_list = []
#         for i in range(self.num_feature_levels):
#             ref_mask_si = F.interpolate(ref_mask.tensor, (self.spatial_list[i], self.spatial_list[i]), mode='nearest')
#             ref_mask_list.append(ref_mask_si.reshape(bs_sp, c, -1).permute(0, 2, 1))
#
#         query_stage_list = []
#         ref_stage_list = []
#         src_copy = []
#
#         for i in range(self.num_feature_levels):
#             if i != 2:
#                 query_multi_si = F.interpolate(self.channel_reduction[i](query_multi_scale[i]),
#                                                (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
#                                                mode='bilinear')
#             else:
#                 query_multi_si = query_multi_scale[i]
#
#             query_stage_list.append(query_multi_si)
#             src_copy.append(query_multi_si.clone())
#
#         for i in range(self.num_feature_levels):
#             if i != 2:
#                 ref_multi_si = F.interpolate(self.channel_reduction[i](ref_multiscale_feature[i]),
#                                              (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
#                                              mode='bilinear')
#             else:
#                 ref_multi_si = ref_multiscale_feature[i]
#
#             ref_stage_list.append(ref_multi_si)
#
#         spatial_cens = []
#         spatial_params = []
#
#         for level_index in range(self.num_feature_levels):
#             if level_index != 0:
#                 pre_feature = F.interpolate(pre_feature, (query_stage_list[level_index].shape[-1], query_stage_list[level_index].shape[-2]), align_corners=True,
#                                              mode='bilinear')
#                 query_stage_list[level_index] = torch.cat((query_stage_list[level_index], pre_feature), dim=1)
#                 query_stage_list[level_index] = self.skip_connection[level_index - 1](query_stage_list[level_index])
#
#             for j in range(2):
#                 src_mask_features = query_stage_list[level_index]
#                 bs_src, d, _, _ = src_mask_features.shape
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#
#                 if mode != 'test':
#                     spatial_tokens = ref_stage_list[level_index]
#                     bs_sp, d, _, _ = spatial_tokens.shape
#                     spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)
#                 else:
#                     if j == 0:
#                         spatial_tokens = ref_stage_list[level_index]
#                         bs_sp, d, _, _ = spatial_tokens.shape
#                         spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)  #### spatial_tokens: (bs, N, d)
#                         ref_mask = ref_mask_list[level_index].reshape(bs_sp, -1, 10)
#
#                         token_dict = {'x': spatial_tokens,
#                                       'token_num': spatial_tokens.size(1),
#                                       'idx_token': torch.arange(spatial_tokens.size(1))[None, :].repeat(
#                                           spatial_tokens.size(0), 1),
#                                       'agg_weight': spatial_tokens.new_ones(spatial_tokens.size(0), spatial_tokens.size(1), 1),
#                                       'mask': None,
#                                       'ref_mask': ref_mask}
#
#                         token_dict_down, _ = self.CTM_module_list[level_index](token_dict)
#                         spatial_tokens = token_dict_down['x']
#                         temp_mask = token_dict_down['ref_mask']
#
#                     spatial_tokens = spatial_tokens.reshape(bs_src, -1, d)
#
#                 output_pos = self.transformer_cross_attention_layers[level_index * 2 + j](
#                     spatial_tokens, spatial_tokens, src_mask_features,
#                     mask=None
#                 )
#                 y = self.transformer_ffn_layers[level_index * 2 + j](output_pos.permute(1, 0, 2)).permute(1, 0, 2)  ### b, n, d
#                 query_stage_list[level_index] = y.reshape(bs_src, self.spatial_list[level_index],
#                                                        self.spatial_list[level_index], d).permute(0, 3, 1, 2)
#
#             spatial_cens.append(spatial_tokens)
#             # spatial_params.append(temp_mask)
#             pre_feature = query_stage_list[level_index]
#
#         if mode != 'test':
#             for i in range(len(query_stage_list)):
#                 src_mask_features = query_stage_list[i]
#                 spatial_tokens = ref_stage_list[i]
#
#                 bs_src, d, _, _ = src_mask_features.shape
#                 bs_sp, d, _, _ = spatial_tokens.shape
#
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#                 spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)
#
#                 src_norm = src_mask_features
#                 spatial_norm = spatial_tokens
#
#                 avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
#                 avg_atten = avg_atten.softmax(dim=-1)
#
#                 out_predict = avg_atten @ ref_mask_list[i]
#                 out_predict_list.append(out_predict)
#
#         else:
#             for i in range(len(query_stage_list)):
#                 src_mask_features = query_stage_list[i]
#                 spatial_tokens = spatial_cens[i]
#
#                 bs_src, d, _, _ = src_mask_features.shape
#                 _, nums, _ = spatial_tokens.shape   ##### (1, 1000, d)
#
#                 ref_mask = spatial_params[i].reshape(1, -1, 10)
#
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#
#                 src_norm = src_mask_features
#                 spatial_norm = spatial_tokens
#
#                 avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
#                 avg_atten = avg_atten.softmax(dim=-1)
#
#                 out_predict = avg_atten @ ref_mask
#                 out_predict_list.append(out_predict)
#
#         results = self.forward_prediction_heads(src_copy, out_predict_list, mode)
#         return results
#
#     def forward_prediction_heads(self, src, out_predict_list, mode='train'):
#         bs, dim1, h1, w1 = src[0].shape
#         bs, dim2, h2, w2 = src[1].shape
#         bs, dim3, h3, w3 = src[2].shape
#
#         f_1_aug = src[0]
#         f_3_aug = src[2]
#
#         f_3_aug = F.interpolate(f_3_aug, (f_1_aug.shape[-1], f_1_aug.shape[-2]), mode='bilinear', align_corners=True)
#         out_predict_3 = F.interpolate(out_predict_list[2].reshape(bs, h3, w3, 10).permute(0, 3, 1, 2), (h2, w2), mode='bilinear',
#                                       align_corners=True)
#
#         final_fuse = self.final_fuse(torch.cat((f_1_aug, f_3_aug), dim=1))
#         out_predict = 1 / 2 * (out_predict_list[0].reshape(bs, h1, w1, 10).permute(0, 3, 1, 2) + out_predict_3)
#
#         outputs_feature = self.Convs(
#             out_predict
#         )
#
#         curr_embedding = outputs_feature + final_fuse
#
#         low_res_masks, iou_predictions = self.sam_decoder(
#             image_embeddings=curr_embedding,
#             multimask_output=True,
#             image_pe=self.pe_layer.forward((64, 64)).unsqueeze(0),
#             sparse_prompt_embeddings=None,
#             dense_prompt_embeddings=None
#         )
#         masks = postprocess_masks(
#             low_res_masks,
#             input_size=(448, 448),
#             original_size=(448, 448),
#         )
#
#         results = {
#             "predictions_mask": masks
#         }
#         return results
#
#
# @register_decoder
# def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
#     return MultiScaleMaskedTransformerDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
#
#
# def postprocess_masks(
#     masks: torch.Tensor,
#     input_size: Tuple[int, ...],
#     original_size: Tuple[int, ...],
# ) -> torch.Tensor:
#     """
#     Remove padding and upscale masks to the original image size.
#
#     Arguments:
#       masks (torch.Tensor): Batched masks from the mask_decoder,
#         in BxCxHxW format.
#       input_size (tuple(int, int)): The size of the image input to the
#         model, in (H, W) format. Used to remove padding.
#       original_size (tuple(int, int)): The original size of the image
#         before resizing for input to the model, in (H, W) format.
#
#     Returns:
#       (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
#         is given by original_size.
#     """
#     masks = F.interpolate(
#         masks,
#         input_size,
#         mode="bilinear",
#         align_corners=False,
#     )
#     masks = masks[..., : input_size[0], : input_size[1]]
#     masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
#     return masks
#
# # ######################################################### seem + spatial-DPC #########################################
# from typing import Optional
# import sys
# sys.path.append("/data1/paintercoco/xdecoder/body/decoder/")
# import imgviz
# import numpy
# import torch
# from torch import nn, Tensor
# from torch.nn import functional as F
# from .registry import register_decoder
# from ...utils import configurable
# from ...modules import PositionEmbeddingSine
# from PIL import Image
# import numpy as np
# from cluster import CTM
# import matplotlib.pyplot as plt
#
#
# def save_colored_mask(mask, save_path):
#     lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
#     color_map = imgviz.label_colormap()
#     lbl_pil.putpalette(color_map.flatten())
#     lbl_pil.save(save_path)
#
# class FFNLayer(nn.Module):
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
# class MLP_1(nn.Module):
#     def __init__(self, dim):
#         super(MLP_1, self).__init__()
#         self.fc1 = nn.Linear(dim, dim*4)
#         self.fc2 = nn.Linear(dim*4, dim)
#         self.act = nn.functional.gelu
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x
#
# class InterTransBlock(nn.Module):
#     def __init__(self, dim):
#         super(InterTransBlock, self).__init__()
#         self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
#         self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
#         self.Attention = MultiScaleAtten(dim)
#         self.FFN = MLP_1(dim)
#
#     def forward(self, x):
#         h = x  # (B, N, H)
#         x = self.SlayerNorm_1(x)
#
#         x = self.Attention(x)  # padding 到right_size
#         x = h + x
#
#         h = x
#         x = self.SlayerNorm_2(x)
#
#         x = self.FFN(x)
#         x = h + x
#
#         return x
#
#
# class MultiScaleAtten(nn.Module):
#     def __init__(self, dim):
#         super(MultiScaleAtten, self).__init__()
#         self.qkv_linear = nn.Linear(dim, dim * 3)
#         self.softmax = nn.Softmax(dim=-1)
#         self.proj = nn.Linear(dim, dim)
#         self.num_head = 8
#         self.scale = (dim // self.num_head) ** 0.5
#         self.initialize()
#
#     def forward(self, x):
#         B, num_blocks, _, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
#         qkv = self.qkv_linear(x).reshape(B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(4, 0, 1, 2, 5, 3, 6).contiguous()  # (3, B, num_block, num_block, head, N, C)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         atten = q @ k.transpose(-1, -2).contiguous()
#         atten = self.softmax(atten)
#         atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
#         atten_value = self.proj(atten_value)  # (B, num_block, num_block, N, C)
#         return atten_value
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)
#
#
# class SpatialAwareTrans(nn.Module):
#     def __init__(self, dim=256, num=1):
#         super(SpatialAwareTrans, self).__init__()
#         self.ini_win_size = 2
#         self.channels = [512 + 10, 512 + 10, 1024 + 10]
#         self.dim = dim
#         self.depth = 3
#         self.fc_module = nn.ModuleList()
#         self.fc_rever_module = nn.ModuleList()
#         self.num = num
#         for i in range(self.depth):
#             self.fc_module.append(nn.Linear(self.channels[i], self.dim))
#
#         for i in range(self.depth):
#             self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))
#
#         self.group_attention = []
#         for i in range(self.num):
#             self.group_attention.append(InterTransBlock(dim))
#         self.group_attention = nn.Sequential(*self.group_attention)
#         self.split_list = [8 * 8, 8 * 8, 4 * 4]
#         self.win_size_list = [8, 8, 4]
#         self.initialize()
#
#     def forward(self, x):
#         x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
#         for j, item in enumerate(x):
#             B, H, W, C = item.shape
#             win_size = self.win_size_list[j]
#             item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4,
#                                                                                                 5).contiguous()
#             item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
#             x[j] = item
#         x = tuple(x)
#         x = torch.cat(x, dim=-2)
#         for i in range(self.num):
#             x = self.group_attention[i](x)  # (B, H // win_size, W // win_size, win_size*win_size, C)
#
#         x = torch.split(x, self.split_list, dim=-2)
#         x = list(x)
#         for j, item in enumerate(x):
#             B, num_blocks, _, N, C = item.shape
#             win_size = self.win_size_list[j]
#             item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B,
#                                                                                                                   num_blocks * win_size,
#                                                                                                                   num_blocks * win_size,
#                                                                                                                   C)
#             item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
#             x[j] = item
#         return x
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)
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
#             nn.BatchNorm2d(512 + 10),
#             nn.Conv2d(512 + 10, 10, 3, 1, 1))
#
#         self.final_fuse = nn.Sequential(
#             nn.Conv2d(1556, 512, 3, 1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#
#         self.hidden_dim_list = [512, 512, 512, 512, 1024, 1024, 512, 512, 1024]
#         for idx in range(self.num_layers):
#             self.transformer_cross_attention_layers.append(
#                 CrossAttention(
#                     embed_size=self.hidden_dim_list[idx],
#                     heads=nheads,
#                     idx=idx
#                 )
#             )
#
#             self.transformer_ffn_layers.append(
#                 FFNLayer(
#                     d_model=self.hidden_dim_list[idx],
#                     dim_feedforward=self.hidden_dim_list[idx] * 2,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )
#
#         self.channel_reduction = nn.ModuleList()
#         self.channel_reduction.append(
#             nn.Sequential(
#                 nn.Conv2d(1024, 512, 3, 1),
#                 nn.BatchNorm2d(512)
#             ))
#         self.channel_reduction.append(
#             nn.Sequential(
#                 nn.Conv2d(1024, 512, 3, 1),
#                 nn.BatchNorm2d(512)
#             ))
#
#         self.spatial_list = [64, 64, 32]
#         self.channel_list = [512, 512, 1024]
#
#         self.skip_connection = nn.ModuleList()
#         self.skip_connection.append(
#             nn.Sequential(
#                 nn.Conv2d(self.channel_list[0] + self.channel_list[1], self.channel_list[1], 3, 1, 1),
#                 nn.BatchNorm2d(self.channel_list[1])
#             ))
#         self.skip_connection.append(
#             nn.Sequential(
#                 nn.Conv2d(self.channel_list[1] + self.channel_list[2], self.channel_list[2], 3, 1, 1),
#                 nn.BatchNorm2d(self.channel_list[2])
#             )
#         )
#
#         self.inter_scale = SpatialAwareTrans(256, 1)
#
#         self.output_upscaling = nn.Sequential(
#             nn.ConvTranspose2d((512 + 10), (512 + 10) // 4, kernel_size=2, stride=2),
#             nn.BatchNorm2d((512 + 10) // 4),
#             nn.LeakyReLU(inplace=True),
#             nn.ConvTranspose2d((512 + 10) // 4, (512 + 10) // 8, kernel_size=2, stride=2),
#             nn.BatchNorm2d((512 + 10) // 8),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d((512 + 10) // 8, 10, 3, 1, 1)
#         )
#         self.CTM_module_list = nn.ModuleList()
#
#         self.CTM_module_list.append(CTM(sample_ratio=64, dim_out=512, k=5))
#         self.CTM_module_list.append(CTM(sample_ratio=32, dim_out=512, k=3))
#         self.CTM_module_list.append(CTM(sample_ratio=16, dim_out=1024, k=3))
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
#         ret["attn_arch"] = cfg['ATTENTION_ARCH']
#
#         return ret
#
#     def forward(self, ref_information, query_information, mode):
#         query_multi_scale = query_information
#         ref_multiscale_feature, ref_mask = ref_information
#         out_predict_list = []
#
#         bs_sp, c, h, w = ref_mask.tensor.shape
#         ref_mask_list = []
#         for i in range(self.num_feature_levels):
#             ref_mask_si = F.interpolate(ref_mask.tensor, (self.spatial_list[i], self.spatial_list[i]), mode='nearest')
#             ref_mask_list.append(ref_mask_si.reshape(bs_sp, c, -1).permute(0, 2, 1))
#
#         query_stage_list = []
#         ref_stage_list = []
#         src_copy = []
#
#         for i in range(self.num_feature_levels):
#             if i != 2:
#                 query_multi_si = F.interpolate(self.channel_reduction[i](query_multi_scale[i]),
#                                                (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
#                                                mode='bilinear')
#             else:
#                 query_multi_si = query_multi_scale[i]
#
#             query_stage_list.append(query_multi_si)
#             src_copy.append(query_multi_si.clone())
#
#         for i in range(self.num_feature_levels):
#             if i != 2:
#                 ref_multi_si = F.interpolate(self.channel_reduction[i](ref_multiscale_feature[i]),
#                                              (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
#                                              mode='bilinear')
#             else:
#                 ref_multi_si = ref_multiscale_feature[i]
#             ref_stage_list.append(ref_multi_si)
#
#         # bs, d, h, w = query_multi_scale[0].shape
#         # print(query_multi_scale[0].shape)
#         # print(ref_mask.tensor.shape)
#         #
#         # query_feature = query_multi_scale[0].view(bs, d, -1).permute(0, 2, 1)
#         # ref_feature = ref_multiscale_feature[0].view(bs, d, -1).permute(0, 2, 1)
#         #
#         # import math
#         # query_feature = query_feature
#         # atten = (query_feature @ ref_feature.transpose(-1, -2)).softmax(dim=-1)
#         #
#         # # q_norm = (query_feature / torch.norm(query_feature, dim=-1, keepdim=True))
#         # # k_norm = (ref_feature / torch.norm(ref_feature, dim=-1, keepdim=True))
#         # # atten = torch.mul(q_norm, k_norm.transpose(-1, -2))
#         #
#         # print(atten.max())
#         # print(atten.min())
#         # # atten = query_feature @ ref_feature
#         # final_mask = atten @ ref_mask_list[2]
#         # final_mask = final_mask.reshape(bs, 32, 32, 10).permute(0, 3, 1, 2)
#         #
#         # plt.figure(figsize=(10, 10))
#         # out = final_mask.softmax(dim=1)
#         #
#         # for index in range(10):
#         #     ax = plt.subplot(5, 5, index+1,)
#         #     plt.imshow(out[0][index].cpu().detach().numpy())
#         #     plt.savefig("dino.jpg", dpi=300)
#         #
#         #
#         #         # for i in range(final_mask[0].shape[0]):
#         #         #     view_img = final_mask[0][i].cpu().numpy()
#         #         #     # print(view_img.sum())
#         #         #     view_img = Image.fromarray(np.uint8(view_img))
#         #         #     view_img.save(str(i)+'.png')
#         #
#         # results = {
#         #             "predictions_mask": final_mask
#         # }
#         # return results
#
#
#         spatial_cens = []
#         spatial_params = []
#
#         for level_index in range(self.num_feature_levels):
#             if level_index != 0:
#                 pre_feature = F.interpolate(pre_feature, (query_stage_list[level_index].shape[-1], query_stage_list[level_index].shape[-2]), align_corners=True,
#                                              mode='bilinear')
#                 query_stage_list[level_index] = torch.cat((query_stage_list[level_index], pre_feature), dim=1)
#                 query_stage_list[level_index] = self.skip_connection[level_index - 1](query_stage_list[level_index])
#
#             for j in range(2):
#                 src_mask_features = query_stage_list[level_index]
#                 bs_src, d, _, _ = src_mask_features.shape
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#
#                 if mode != 'test':
#                     spatial_tokens = ref_stage_list[level_index]
#                     bs_sp, d, _, _ = spatial_tokens.shape
#                     spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)
#                 else:
#                     if j == 0:
#                         spatial_tokens = ref_stage_list[level_index]
#                         bs_sp, d, _, _ = spatial_tokens.shape
#                         spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)  #### spatial_tokens: (bs, N, d)
#                         ref_mask = ref_mask_list[level_index].reshape(bs_sp, -1, 10)
#
#                         token_dict = {'x': spatial_tokens,
#                                       'token_num': spatial_tokens.size(1),
#                                       'idx_token': torch.arange(spatial_tokens.size(1))[None, :].repeat(
#                                           spatial_tokens.size(0), 1),
#                                       'agg_weight': spatial_tokens.new_ones(spatial_tokens.size(0), spatial_tokens.size(1), 1),
#                                       'mask': None,
#                                       'ref_mask': ref_mask}
#
#                         token_dict_down, _ = self.CTM_module_list[level_index](token_dict)
#                         spatial_tokens = token_dict_down['x']
#                         temp_mask = token_dict_down['ref_mask']
#
#                     spatial_tokens = spatial_tokens.reshape(bs_src, -1, d)
#
#                 output_pos = self.transformer_cross_attention_layers[level_index * 2 + j](
#                     spatial_tokens, spatial_tokens, src_mask_features,
#                     mask=None
#                 )
#                 y = self.transformer_ffn_layers[level_index * 2 + j](output_pos.permute(1, 0, 2)).permute(1, 0, 2)  ### b, n, d
#                 query_stage_list[level_index] = y.reshape(bs_src, self.spatial_list[level_index],
#                                                        self.spatial_list[level_index], d).permute(0, 3, 1, 2)
#             spatial_cens.append(spatial_tokens)
#             spatial_params.append(temp_mask)
#             pre_feature = query_stage_list[level_index]
#
#         if mode != 'test':
#             for i in range(len(query_stage_list)):
#                 src_mask_features = query_stage_list[i]
#                 spatial_tokens = ref_stage_list[i]
#
#                 bs_src, d, _, _ = src_mask_features.shape
#                 bs_sp, d, _, _ = spatial_tokens.shape
#
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#                 spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)
#
#                 src_norm = src_mask_features / (torch.norm(src_mask_features, dim=-1, keepdim=True) + 1e-12)
#                 spatial_norm = spatial_tokens / (torch.norm(spatial_tokens, dim=-1, keepdim=True) + 1e-12)
#
#                 avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
#                 avg_atten = avg_atten.softmax(dim=-1)
#
#                 out_predict = avg_atten @ ref_mask_list[i]
#                 out_predict_list.append(out_predict)
#
#         else:
#             for i in range(len(query_stage_list)):
#                 src_mask_features = query_stage_list[i]
#                 spatial_tokens = spatial_cens[i]
#
#                 bs_src, d, _, _ = src_mask_features.shape
#                 _, nums, _ = spatial_tokens.shape   ##### (1, 1000, d)
#
#                 ref_mask = spatial_params[i].reshape(1, -1, 10)
#
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#                 src_norm = src_mask_features / (torch.norm(src_mask_features, dim=-1, keepdim=True) + 1e-12)
#                 spatial_norm = spatial_tokens / (torch.norm(spatial_tokens, dim=-1, keepdim=True) + 1e-12)
#
#                 avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
#                 avg_atten = avg_atten.softmax(dim=-1)
#
#                 out_predict = avg_atten @ ref_mask
#                 out_predict_list.append(out_predict)
#         #
#         # out = out_predict_list[0].reshape(bs_src, 64, 64, 10).permute(0, 3, 1, 2)
#         # plt.figure(figsize=(10, 10))
#         # out = out.softmax(dim=1)
#         #
#         # for index in range(10):
#         #     ax = plt.subplot(5, 5, index+1,)
#         #     plt.imshow(out[0][index].cpu().detach().numpy())
#         #     plt.savefig("feature.jpg",dpi=300)
#
#         results = self.forward_prediction_heads(src_copy, out_predict_list, mode)
#         return results
#
#     def forward_prediction_heads(self, src, out_predict_list, mode='train'):
#         bs, dim1, h1, w1 = src[0].shape
#         bs, dim2, h2, w2 = src[1].shape
#         bs, dim3, h3, w3 = src[2].shape
#
#         feature_1 = torch.cat((src[0], out_predict_list[0].reshape(bs, -1, h1, w1)), dim=1)
#         feature_2 = torch.cat((src[1], out_predict_list[1].reshape(bs, -1, h2, w2)), dim=1)
#         feature_3 = torch.cat((src[2], out_predict_list[2].reshape(bs, -1, h3, w3)), dim=1)
#
#         f_1_aug, f_2_aug, f_3_aug = self.inter_scale((feature_1, feature_2, feature_3))
#         f_3_aug = F.interpolate(f_3_aug, (f_1_aug.shape[-1], f_1_aug.shape[-2]), mode='bilinear', align_corners=True)
#         out_predict_3 = F.interpolate(out_predict_list[2].reshape(bs, h3, w3, 10).permute(0, 3, 1, 2), (h2, w2), mode='bilinear',
#                                       align_corners=True)
#
#         final_fuse = self.final_fuse(torch.cat((f_1_aug, f_3_aug), dim=1))
#         out_predict = 1 / 2 * (out_predict_list[0].reshape(bs, h1, w1, 10).permute(0, 3, 1, 2) + out_predict_3)
#
#         outputs_mask = self.output_upscaling(
#             torch.cat((final_fuse, out_predict), dim=1)
#         )
#
#         # out_save = outputs_mask ### (b, 10, 512, 64, 64)
#         # out_save = out_save.softmax(dim=1)
#         # plt.figure(figsize=(10, 10))
#         # for index in range(10):
#         #     ax = plt.subplot(5, 5, index + 1,)
#         #     plt.imshow(out_save[0][index].cpu().detach().numpy())
#         #     plt.savefig("feature_1.jpg", dpi=300)
#
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

# from typing import Optional
# import sys
# sys.path.append("/data1/paintercoco/xdecoder/body/decoder/")
# import imgviz
# import numpy
# import torch
# from torch import nn, Tensor
# from torch.nn import functional as F
# from .registry import register_decoder
# from ...utils import configurable
# from ...modules import PositionEmbeddingSine
# from PIL import Image
# import numpy as np
# from spatial_DPC import Spatial_DPC
# import matplotlib.pyplot as plt
#
# def save_colored_mask(mask, save_path):
#     lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
#     color_map = imgviz.label_colormap()
#     lbl_pil.putpalette(color_map.flatten())
#     lbl_pil.save(save_path)
#
# class FFNLayer(nn.Module):
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
# class MLP_1(nn.Module):
#     def __init__(self, dim):
#         super(MLP_1, self).__init__()
#         self.fc1 = nn.Linear(dim, dim*4)
#         self.fc2 = nn.Linear(dim*4, dim)
#         self.act = nn.functional.gelu
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.dropout(x)
#         return x
#
# class InterTransBlock(nn.Module):
#     def __init__(self, dim):
#         super(InterTransBlock, self).__init__()
#         self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
#         self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
#         self.Attention = MultiScaleAtten(dim)
#         self.FFN = MLP_1(dim)
#
#     def forward(self, x):
#         h = x  # (B, N, H)
#         x = self.SlayerNorm_1(x)
#
#         x = self.Attention(x)  # padding 到right_size
#         x = h + x
#
#         h = x
#         x = self.SlayerNorm_2(x)
#
#         x = self.FFN(x)
#         x = h + x
#
#         return x
#
#
# class MultiScaleAtten(nn.Module):
#     def __init__(self, dim):
#         super(MultiScaleAtten, self).__init__()
#         self.qkv_linear = nn.Linear(dim, dim * 3)
#         self.softmax = nn.Softmax(dim=-1)
#         self.proj = nn.Linear(dim, dim)
#         self.num_head = 8
#         self.scale = (dim // self.num_head) ** 0.5
#         self.initialize()
#
#     def forward(self, x):
#         B, num_blocks, _, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
#         qkv = self.qkv_linear(x).reshape(B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(4,
#                                                                                                                       0,
#                                                                                                                       1,
#                                                                                                                       2,
#                                                                                                                       5,
#                                                                                                                       3,
#                                                                                    6).contiguous()  # (3, B, num_block, num_block, head, N, C)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         atten = q @ k.transpose(-1, -2).contiguous()
#         atten = self.softmax(atten)
#         atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
#         atten_value = self.proj(atten_value)  # (B, num_block, num_block, N, C)
#         return atten_value
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)
#
#
# class SpatialAwareTrans(nn.Module):
#     def __init__(self, dim=256, num=1):
#         super(SpatialAwareTrans, self).__init__()
#         self.ini_win_size = 2
#         self.channels = [512 + 10, 512 + 10, 1024 + 10]
#         self.dim = dim
#         self.depth = 3
#         self.fc_module = nn.ModuleList()
#         self.fc_rever_module = nn.ModuleList()
#         self.num = num
#         for i in range(self.depth):
#             self.fc_module.append(nn.Linear(self.channels[i], self.dim))
#
#         for i in range(self.depth):
#             self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))
#
#         self.group_attention = []
#         for i in range(self.num):
#             self.group_attention.append(InterTransBlock(dim))
#         self.group_attention = nn.Sequential(*self.group_attention)
#         self.split_list = [8 * 8, 8 * 8, 4 * 4]
#         self.win_size_list = [8, 8, 4]
#         self.initialize()
#
#     def forward(self, x):
#         x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
#         for j, item in enumerate(x):
#             B, H, W, C = item.shape
#             win_size = self.win_size_list[j]
#             item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4,
#                                                                                                 5).contiguous()
#             item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
#             x[j] = item
#         x = tuple(x)
#         x = torch.cat(x, dim=-2)
#         for i in range(self.num):
#             x = self.group_attention[i](x)  # (B, H // win_size, W // win_size, win_size*win_size, C)
#
#         x = torch.split(x, self.split_list, dim=-2)
#         x = list(x)
#         for j, item in enumerate(x):
#             B, num_blocks, _, N, C = item.shape
#             win_size = self.win_size_list[j]
#             item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4,
#                                                                                           5).contiguous().reshape(B,
#                                                                                                                   num_blocks * win_size,
#                                                                                                                   num_blocks * win_size,
#                                                                                                                   C)
#             item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
#             x[j] = item
#         return x
#
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight.data)
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
#             nn.BatchNorm2d(512 + 10),
#             nn.Conv2d(512 + 10, 10, 3, 1, 1))
#
#         self.final_fuse = nn.Sequential(
#             nn.Conv2d(1556, 512, 3, 1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#
#         self.hidden_dim_list = [512, 512, 512, 512, 1024, 1024, 512, 512, 1024]
#         for idx in range(self.num_layers):
#             self.transformer_cross_attention_layers.append(
#                 CrossAttention(
#                     embed_size=self.hidden_dim_list[idx],
#                     heads=nheads,
#                     idx=idx
#                 )
#             )
#
#             self.transformer_ffn_layers.append(
#                 FFNLayer(
#                     d_model=self.hidden_dim_list[idx],
#                     dim_feedforward=self.hidden_dim_list[idx] * 2,
#                     dropout=0.0,
#                     normalize_before=pre_norm,
#                 )
#             )
#
#         self.channel_reduction = nn.ModuleList()
#         self.channel_reduction.append(
#             nn.Sequential(
#                 nn.Conv2d(1024, 512, 3, 1),
#                 nn.BatchNorm2d(512)
#             ))
#         self.channel_reduction.append(
#             nn.Sequential(
#                 nn.Conv2d(1024, 512, 3, 1),
#                 nn.BatchNorm2d(512)
#             ))
#
#         self.spatial_list = [64, 64, 32]
#         self.channel_list = [512, 512, 1024]
#
#         self.skip_connection = nn.ModuleList()
#         self.skip_connection.append(
#             nn.Sequential(
#                 nn.Conv2d(self.channel_list[0] + self.channel_list[1], self.channel_list[1], 3, 1, 1),
#                 nn.BatchNorm2d(self.channel_list[1])
#             ))
#         self.skip_connection.append(
#             nn.Sequential(
#                 nn.Conv2d(self.channel_list[1] + self.channel_list[2], self.channel_list[2], 3, 1, 1),
#                 nn.BatchNorm2d(self.channel_list[2])
#             )
#         )
#
#         self.inter_scale = SpatialAwareTrans(256, 1)
#         self.output_upscaling = nn.Sequential(
#             nn.ConvTranspose2d(512 * 10 * 2, 512 * 10 * 2 // 4, kernel_size=2, stride=2, groups=10),
#             nn.BatchNorm2d(512 * 10 * 2 // 4),
#             nn.LeakyReLU(inplace=True),
#             nn.ConvTranspose2d(512 * 10 * 2 // 4, 512 * 10 * 2// 8, kernel_size=2, stride=2, groups=10),
#             nn.BatchNorm2d(512 * 10 * 2// 8),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(512 * 10 * 2// 8, 10, 1, 1, groups=10)
#         )
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
#         ret["attn_arch"] = cfg['ATTENTION_ARCH']
#
#         return ret
#
#     def forward(self, ref_information, query_information, mode):
#         query_multi_scale = query_information
#         ref_multiscale_feature, ref_mask = ref_information
#         out_predict_list = []
#
#         bs_sp, c, h, w = ref_mask.tensor.shape
#         ref_mask_list = []
#         for i in range(self.num_feature_levels):
#             ref_mask_si = F.interpolate(ref_mask.tensor, (self.spatial_list[i], self.spatial_list[i]), mode='nearest')
#             ref_mask_list.append(ref_mask_si.reshape(bs_sp, c, -1).permute(0, 2, 1))
#
#         query_stage_list = []
#         ref_stage_list = []
#         src_copy = []
#
#         for i in range(self.num_feature_levels):
#             if i != 2:
#                 query_multi_si = F.interpolate(self.channel_reduction[i](query_multi_scale[i]),
#                                                (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
#                                                mode='bilinear')
#             else:
#                 query_multi_si = query_multi_scale[i]
#
#             if mode == 'test':
#                 query_multi_si = query_multi_si.repeat(bs_sp, 1, 1, 1)
#
#             query_stage_list.append(query_multi_si)
#             src_copy.append(query_multi_si.clone())
#
#         for i in range(self.num_feature_levels):
#             if i != 2:
#                 ref_multi_si = F.interpolate(self.channel_reduction[i](ref_multiscale_feature[i]),
#                                              (self.spatial_list[i], self.spatial_list[i]), align_corners=True,
#                                              mode='bilinear')
#             else:
#                 ref_multi_si = ref_multiscale_feature[i]
#             ref_stage_list.append(ref_multi_si)
#
#         spatial_cens = []
#         spatial_params = []
#
#         for level_index in range(self.num_feature_levels):
#             if level_index != 0:
#                 pre_feature = F.interpolate(pre_feature, (query_stage_list[level_index].shape[-1], query_stage_list[level_index].shape[-2]), align_corners=True,
#                                              mode='bilinear')
#                 query_stage_list[level_index] = torch.cat((query_stage_list[level_index], pre_feature), dim=1)
#                 query_stage_list[level_index] = self.skip_connection[level_index - 1](query_stage_list[level_index])
#
#             for j in range(2):
#                 src_mask_features = query_stage_list[level_index]
#                 bs_src, d, _, _ = src_mask_features.shape
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#
#                 if mode != 'test':
#                     spatial_tokens = ref_stage_list[level_index]
#                     bs_sp, d, _, _ = spatial_tokens.shape
#                     spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)
#                 else:
#                     spatial_tokens = ref_stage_list[level_index]
#                     bs_sp, d, _, _ = spatial_tokens.shape
#                     spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)
#
#                 output_pos = self.transformer_cross_attention_layers[level_index * 2 + j](
#                     spatial_tokens, spatial_tokens, src_mask_features,
#                     mask=None
#                 )
#                 y = self.transformer_ffn_layers[level_index * 2 + j](output_pos.permute(1, 0, 2)).permute(1, 0, 2)  ### b, n, d
#                 query_stage_list[level_index] = y.reshape(bs_src, self.spatial_list[level_index],
#                                                        self.spatial_list[level_index], d).permute(0, 3, 1, 2)
#             spatial_cens.append(spatial_tokens)
#             pre_feature = query_stage_list[level_index]
#
#         if mode != 'test':
#             for i in range(len(query_stage_list)):
#                 src_mask_features = query_stage_list[i]
#                 spatial_tokens = ref_stage_list[i]
#
#                 bs_src, d, _, _ = src_mask_features.shape
#                 bs_sp, d, _, _ = spatial_tokens.shape
#
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#                 spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)
#
#                 src_norm = src_mask_features / (torch.norm(src_mask_features, dim=-1, keepdim=True) + 1e-12)
#                 spatial_norm = spatial_tokens / (torch.norm(spatial_tokens, dim=-1, keepdim=True) + 1e-12)
#
#                 avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
#                 avg_atten = avg_atten.softmax(dim=-1)
#
#                 out_predict = avg_atten @ ref_mask_list[i]
#                 out_predict_list.append(out_predict)
#         else:
#             for i in range(len(query_stage_list)):
#                 src_mask_features = query_stage_list[i]
#                 spatial_tokens = ref_stage_list[i]
#
#                 bs_src, d, h, w = src_mask_features.shape
#                 bs_sp, d, _, _ = spatial_tokens.shape
#
#                 src_mask_features = src_mask_features.view(bs_src, d, -1).permute(0, 2, 1)
#                 spatial_tokens = spatial_tokens.view(bs_sp, d, -1).permute(0, 2, 1)
#
#                 src_norm = src_mask_features / (torch.norm(src_mask_features, dim=-1, keepdim=True) + 1e-12)
#                 spatial_norm = spatial_tokens / (torch.norm(spatial_tokens, dim=-1, keepdim=True) + 1e-12)
#
#                 avg_atten = (src_norm @ spatial_norm.transpose(-1, -2))
#                 avg_atten = avg_atten.softmax(dim=-1)
#
#                 out_predict = avg_atten @ ref_mask_list[i]
#                 out_predict_list.append(out_predict)
#
#         # out = out_predict_list[0].reshape(bs_src, 64, 64, 10).permute(0, 3, 1, 2)
#         # plt.figure(figsize=(10, 10))
#         # for index in range(10):
#         #     ax = plt.subplot(5, 5, index+1,)
#         #     plt.imshow(out[0][index].cpu().detach().numpy())
#         #     plt.savefig("feature.jpg",dpi=300)
#
#         results = self.forward_prediction_heads(src_copy, out_predict_list, mode)
#         return results
#
#     def forward_prediction_heads(self, src, out_predict_list, mode='train'):
#         bs, dim1, h1, w1 = src[0].shape
#         bs, dim2, h2, w2 = src[1].shape
#         bs, dim3, h3, w3 = src[2].shape
#
#         feature_1 = torch.cat((src[0], out_predict_list[0].reshape(bs, -1, h1, w1)), dim=1)
#         feature_2 = torch.cat((src[1], out_predict_list[1].reshape(bs, -1, h2, w2)), dim=1)
#         feature_3 = torch.cat((src[2], out_predict_list[2].reshape(bs, -1, h3, w3)), dim=1)
#
#         f_1_aug, f_2_aug, f_3_aug = self.inter_scale((feature_1, feature_2, feature_3))
#         f_3_aug = F.interpolate(f_3_aug, (f_1_aug.shape[-1], f_1_aug.shape[-2]), mode='bilinear', align_corners=True)
#         out_predict_3 = F.interpolate(out_predict_list[2].reshape(bs, h3, w3, 10).permute(0, 3, 1, 2), (h2, w2), mode='bilinear',
#                                       align_corners=True)
#
#         final_fuse = self.final_fuse(torch.cat((f_1_aug, f_3_aug), dim=1)) ##### (b, 512, h, w)
#         out_predict = 1 / 2 * (out_predict_list[0].reshape(bs, h1, w1, 10).permute(0, 3, 1, 2) + out_predict_3)  #### (b, 10, h, w)
#
#         ########把output_predict直接乘在特征上  #####################################
#         final_fuse = final_fuse.unsqueeze(1).repeat(1, 10, 1, 1, 1)  ### (b, 10, 512, h, w)
#         out_predict = out_predict.unsqueeze(2)   ### (b, 10, 1, h, w)
#         out_predict_aug = final_fuse * out_predict  #### (b, 10, 512, h, w)
#
#         # out_save = out_predict_aug ### (b, 10, 512, 64, 64)
#         # plt.figure(figsize=(10, 10))
#         # for index in range(10):
#         #     ax = plt.subplot(5, 5, index + 1, )
#         #     fea, _ = torch.max(out_save[0][index], dim=0)
#         #     plt.imshow(fea.cpu().detach().numpy())
#         #     plt.savefig("feature_0.jpg", dpi=300)
#
#         out_predict_aug_1 = torch.cat((out_predict_aug, final_fuse), dim=2)
#
#         out_predict_aug_1 = out_predict_aug_1.reshape(bs, -1, h1, w1)
#         outputs_mask = self.output_upscaling(out_predict_aug_1)
#
#         if mode == 'test':
#             # final_fuse = final_fuse.mean(dim=0).unsqueeze(0)
#             # out_predict = out_predict.mean(dim=0).unsqueeze(0)
#             outputs_mask = outputs_mask.mean(dim=0).unsqueeze(0)
#
#         # plt.figure(figsize=(10, 10))
#         # for index in range(10):
#         #     ax = plt.subplot(5, 5, index + 1, )
#         #     plt.imshow(outputs_mask[0][index].cpu().detach().numpy())
#         #     plt.savefig("feature_1.jpg", dpi=300)
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












