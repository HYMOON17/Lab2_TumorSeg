'''
test e2e code
normally on A100 it cost 30G Gpu
if gpu is unavailable try sliding window fn change
sw_device & device diff can reduce gpu cost

'''

import torch
import json
import numpy as np
from typing import Dict, List, Union
import os

from monai.data import PersistentDataset,Dataset, load_decathlon_datalist, decollate_batch,ThreadDataLoader,set_track_meta, MetaTensor, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, CropForegroundd, Orientationd, Spacingd, EnsureTyped, AsDiscrete,AsDiscreted,ToTensord,MapLabelValued, Invertd
from monai.inferers import sliding_window_inference
from monai.handlers.utils import from_engine

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
from monai.visualize import blend_images

import sys
sys.path.append('/data/hyungseok/Swin-UNETR')
from utils.logger import setup_logging, get_logger, save_config_as_json, save_current_code
from utils.my_utils import load_config
from utils.seed import set_seed_and_env, worker_init_fn
from models.model_manager import build_model, load_model_weights
from losses.metric import MetricManager
'''
아래는 폴더 모듈화 이전에 작성해놓은 코드
아래 코드를 진행하면서 함수 모듈화 했고, 이젠 폴더별 코드 분리 통해 외부에서 import 예정
'''

# ### 2. Model Loader
# def build_model(config: Dict, device: torch.device):
#     """
#     config['model_params']['type']에 따라 Swin / AttSwin / Contrastive 등 분기
#     """
#     if config['model_params']['type'] == "SwinUnetr":
#         from monai.networks.nets import SwinUNETR
#         model = SwinUNETR(
#             img_size=config['model_params']['img_size'],
#             in_channels=config['model_params']['in_channels'],
#             out_channels=config['model_params']['out_channels'],
#             feature_size=config['model_params']['feature_size'],
#             use_checkpoint=config['model_params']['use_checkpoint']
#         ).to(device)

#     if config['model_params']['type'] == "AttSwinUNETR":
#         import sys
#         sys.path.append('/data/hyungseok/Swin-UNETR/')
#         from models.swin_unetr_detr import SwinUNETR, TumorPrototypeIntegration

#         class AttSwinUNETR(SwinUNETR): 
#             def __init__(self, img_size, in_channels, out_channels, feature_size, use_checkpoint,
#                         # feature_dims=[48, 96, 192], prototype_dim=128):
#                         feature_dims=[48], prototype_dim=48):
#                 super(AttSwinUNETR, self).__init__(
#                     img_size=img_size,
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     feature_size=feature_size,
#                     use_checkpoint=use_checkpoint
#                 )
                
#                 # ✅ Tumor Prototype Integration 추가
#                 self.liver_prototype_integration = TumorPrototypeIntegration(feature_dims=feature_dims, prototype_dim=prototype_dim, num_heads=4, dropout=0.1)
#                 # ✅ Tumor Prototype Integration 추가
#                 self.lung_prototype_integration = TumorPrototypeIntegration(feature_dims=feature_dims, prototype_dim=prototype_dim, num_heads=4, dropout=0.1)

#             def forward(self, x_in, label=None, organ=None):
#                 hidden_states_out = self.swinViT(x_in, self.normalize)
#                 enc0 = self.encoder1(x_in)
#                 enc1 = self.encoder2(hidden_states_out[0])  # (B, 48, D, H, W)
#                 enc2 = self.encoder3(hidden_states_out[1])  # (B, 96, D, H, W)
#                 enc3 = self.encoder4(hidden_states_out[2])  # (B, 192, D, H, W)
#                 dec4 = self.encoder10(hidden_states_out[4])
#                 dec3 = self.decoder5(dec4, hidden_states_out[3])
#                 dec2 = self.decoder4(dec3, enc3)
#                 dec1 = self.decoder3(dec2, enc2)
#                 dec0 = self.decoder2(dec1, enc1)
#                 out = self.decoder1(dec0, enc0)

#                 if self.training and label is not None:
#                     # if organ == "mixed":
#                     # 배치의 절반은 liver, 나머지 절반은 lung로 가정 (B는 짝수)
#                     B = out.shape[0]
#                     half = B // 2
#                     # liver part: 인덱스 0 ~ half-1
#                     out_liver = out[:half]      # (B/2, C, D, H, W)
#                     label_liver = label[:half]
#                     # lung part: 인덱스 half ~ B-1
#                     out_lung = out[half:]
#                     label_lung = label[half:]
                                        
#                     def calc_valid_centroid(out_part, label_part):
#                         """
#                         Args:
#                             out_part: (B, C, D, H, W) - 입력 feature map
#                             label_part: (B, 1, D, H, W) - Binary label map (종양 = 1, 배경 = 0)
                        
#                         Returns:
#                             centroid: (B, C) - 종양 feature의 정확한 masked average
#                             valid_idx: (B_valid,) - 유효한 샘플 인덱스 (종양 픽셀이 존재하는 경우)
#                         """
#                         mask = (label_part == 1).float()  # (B, 1, D, H, W)
#                         spatial_dims = list(range(2, out_part.dim()))  # D, H, W 차원
                        
#                         # 각 샘플별 종양 voxel 수 계산
#                         count = mask.sum(dim=spatial_dims)  # (B, 1)
                        
#                         # 유효한 샘플 인덱스: 종양 픽셀이 1개 이상 있는 경우
#                         valid_idx = (count.squeeze(1) > 0).nonzero().squeeze(1)
#                         if valid_idx.numel() == 0:
#                             return torch.zeros(out_part.shape[0], out_part.shape[1], device=out_part.device)  # (B, C)
                        
#                         # 종양 영역에 대해 마스크된 값의 합을 voxel 수로 나눔 (0 division 방지를 위해 작은 값을 더함)
#                         centroid = (out_part * mask).sum(dim=tuple(spatial_dims)) / (count + 1e-6)  # (B, C)
#                         # return centroid, valid_idx
#                         return centroid

#                     centroid_liver = calc_valid_centroid(out_liver, label_liver)
#                     centroid_lung = calc_valid_centroid(out_lung, label_lung)

#                     out_liver = self.liver_prototype_integration([out_liver],centroid_liver)[0]
#                     out_lung = self.lung_prototype_integration([out_lung],centroid_lung)[0]
#                     # 배치 순서대로 다시 합치기
#                     out = torch.cat([out_liver, out_lung], dim=0)
            
#                 else:
#                     # organ 인자에 따라 해당 organ의 prototype 선택 (예: "lung" 또는 "liver")
#                     if organ == "lung":
#                         # Inference 시 학습된 query를 그대로 사용
#                         out = self.lung_prototype_integration.forward_inference([out])[0]

#                     elif organ == "liver":
#                         out = self.liver_prototype_integration.forward_inference([out])[0]
#                     else:
#                         # organ 정보가 없으면 기본적으로 liver prototype 사용 (또는 두 organ 모두 업데이트)
#                         out = self.liver_prototype_integration.forward_inference([out])[0]

                    
#                 logits = self.out(out.float())
#                 return logits

        
#         model = AttSwinUNETR(
#         img_size=config['model_params']['img_size'],
#         in_channels=config['model_params']['in_channels'],
#         out_channels=config['model_params']['out_channels'],
#         feature_size=config['model_params']['feature_size'],
#         use_checkpoint=config['model_params']['use_checkpoint']
#         ).to(device)
     
#     if config['model_params']['type'] == "AttSwinUNETR_0313":
#         import sys
#         sys.path.append('/data/hyungseok/Swin-UNETR/')
#         from models.swin_unetr_att import SwinUNETR, MultiScaleCrossAttention, PrototypeAttention
#         import torch.nn as nn
#         class AttSwinUNETR(SwinUNETR):
#             def __init__(self, img_size, in_channels, out_channels, feature_size, use_checkpoint,
#                         # feature_dims=[48, 96, 192], prototype_dim=128):
#                         feature_dims=[48], prototype_dim=48):
#                 super(AttSwinUNETR, self).__init__(
#                     img_size=img_size,
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     feature_size=feature_size,
#                     use_checkpoint=use_checkpoint
#                 )
                
#                 # ✅ 폐 종양과 간 종양을 위한 Prototype 추가
#                 self.prototypes_lung = nn.ParameterList([
#                     nn.Parameter(torch.randn(1, dim)) for dim in feature_dims
#                 ])
#                 self.prototypes_liver = nn.ParameterList([
#                     nn.Parameter(torch.randn(1, dim)) for dim in feature_dims
#                 ])

#                 # Prototype 업데이트를 위한 self-attention 모듈 (별도, 파라미터 공유 방지)
#                 self.liver_prototype_attention = PrototypeAttention(embed_dim=prototype_dim, num_heads=4)
#                 # Prototype 업데이트를 위한 self-attention 모듈 (별도, 파라미터 공유 방지)
#                 self.lung_prototype_attention = PrototypeAttention(embed_dim=prototype_dim, num_heads=4)
        
#                 # Multi-Scale Cross Attention 추가
#                 self.multi_scale_cross_attention = MultiScaleCrossAttention(
#                     feature_dims=feature_dims, num_heads=4
#                 )
#             def update_prototype(self, centroid, prototype, organ):
#                 """
#                 centroid: (B, C) - 각 배치의 feature 평균 (예: 종양 영역의 feature)
#                 prototype: (1, C)
#                 organ: string, 'lung' 또는 'liver'에 따라 업데이트 모듈 선택
#                 """
#                 # 배치 차원 평균을 통해 (1, C)로 변환 (이미 1일 수 있음)
#                 avg_centroid = centroid.mean(dim=0, keepdim=True)  # (1, C)
#                 # self-attention 모듈 입력 형식에 맞게 차원 확장: (B, seq_len, C)
#                 avg_centroid = avg_centroid.unsqueeze(0)  # (1, 1, C)
#                 prototype_exp = prototype.unsqueeze(0)     # (1, 1, C)
#                 # Organ에 따라 해당 prototype attention 모듈 선택
#                 if organ == "lung":
#                     updated_proto = self.lung_prototype_attention(avg_centroid, prototype_exp)
#                 elif organ == "liver":
#                     updated_proto = self.liver_prototype_attention(avg_centroid, prototype_exp)
#                 else:
#                     # 기본은 liver
#                     updated_proto = self.liver_prototype_attention(avg_centroid, prototype_exp)
#                 updated_proto = updated_proto.squeeze(0)  # (1, C)
#                 return updated_proto
            

#             def forward(self, x_in, label=None, organ=None):
#                 hidden_states_out = self.swinViT(x_in, self.normalize)
#                 enc0 = self.encoder1(x_in)
#                 enc1 = self.encoder2(hidden_states_out[0])  # (B, 48, D, H, W)
#                 enc2 = self.encoder3(hidden_states_out[1])  # (B, 96, D, H, W)
#                 enc3 = self.encoder4(hidden_states_out[2])  # (B, 192, D, H, W)
#                 dec4 = self.encoder10(hidden_states_out[4])
#                 dec3 = self.decoder5(dec4, hidden_states_out[3])
#                 dec2 = self.decoder4(dec3, enc3)
#                 dec1 = self.decoder3(dec2, enc2)
#                 dec0 = self.decoder2(dec1, enc1)
#                 out = self.decoder1(dec0, enc0)

#                 if self.training and label is not None:
#                     # if organ == "mixed":
#                     # 배치의 절반은 liver, 나머지 절반은 lung로 가정 (B는 짝수)
#                     B = out.shape[0]
#                     half = B // 2
#                     # liver part: 인덱스 0 ~ half-1
#                     out_liver = out[:half]      # (B/2, C, D, H, W)
#                     label_liver = label[:half]
#                     # lung part: 인덱스 half ~ B-1
#                     out_lung = out[half:]
#                     label_lung = label[half:]
                    
#                     def calc_valid_centroid(out_part, label_part):
#                         """
#                         out_part: (B, C, D, H, W)
#                         label_part: (B, 1, D, H, W)
                        
#                         유효한(종양 픽셀이 하나라도 존재하는) 샘플에 대해
#                         sum_features와 유효 인덱스를 반환합니다.
#                         """
                         
#                         mask = (label_part == 1).float()  # (B, 1, D, H, W)
#                         spatial_dims = list(range(2, out_part.dim()))
#                         count = mask.sum(dim=spatial_dims)  # (B, 1)
#                         # 유효한 샘플의 인덱스를 구합니다.
#                         valid_idx = (count.squeeze(1) > 0).nonzero().squeeze(1)
#                         if valid_idx.numel() == 0:
#                             return None, valid_idx
#                         # 유효한 샘플만의 sum_features 계산
#                         sum_features = (out_part * mask).sum(dim=spatial_dims)  # (B, C)
#                         valid_sum_features = sum_features[valid_idx.tolist()]  # (B_valid, C)
#                         centroid = valid_sum_features.mean(dim=0, keepdim=True) / ( (mask.sum(dim=spatial_dims)[valid_idx.tolist()]).mean() + 1e-6 )
#                         return centroid, valid_idx.tolist()

                    
#                     centroid_liver, valid_idx_liver = calc_valid_centroid(out_liver, label_liver)
#                     centroid_lung, valid_idx_lung = calc_valid_centroid(out_lung, label_lung)

#                     if centroid_liver != None :
#                         # 유효한 샘플들의 평균을 계산
#                         # valid한 샘플만 선택
#                         valid_out_liver = torch.index_select(out_liver, 0, torch.tensor(valid_idx_liver, device=out.device))
#                         # valid 샘플들에 대해서만 attention 적용
#                         updated_valid_out_liver = self.multi_scale_cross_attention([valid_out_liver], self.prototypes_liver)[0]
#                         # 원래 out_liver의 valid 인덱스 위치에 업데이트된 값을 반영
#                         out_liver = out_liver.clone()
#                         out_liver[valid_idx_liver] = updated_valid_out_liver
#                         for i, prototype in enumerate(self.prototypes_liver):
#                             updated_proto = self.update_prototype(centroid_liver, prototype, "liver")
#                             # self.prototypes_liver[i].data = updated_proto.data
#                             momentum = 0.9  # 모멘텀 계수 (예시)
#                             with torch.no_grad():
#                                 self.prototypes_liver[i].mul_(momentum).add_(updated_proto * (1 - momentum))

#                     else:
#                         # 종양이 전혀 없으면 업데이트 생략
#                         pass
                    
#                     # lung 계산
                    
#                     if centroid_lung != None :
#                         # valid한 샘플만 선택
#                         valid_out_lung = torch.index_select(out_lung, 0, torch.tensor(valid_idx_lung, device=out.device))
#                         # valid 샘플들에 대해서만 attention 적용
#                         updated_valid_out_lung = self.multi_scale_cross_attention([valid_out_lung], self.prototypes_lung)[0]
#                         # 원래 out_lung의 valid 인덱스 위치에 업데이트된 값을 반영
#                         out_lung = out_lung.clone()
#                         out_lung[valid_idx_lung] = updated_valid_out_lung
#                         for i, prototype in enumerate(self.prototypes_lung):
#                             updated_proto = self.update_prototype(centroid_lung, prototype, "lung")
#                             momentum = 0.9  # 모멘텀 계수 (예시)
#                             with torch.no_grad():
#                                 self.prototypes_lung[i].mul_(momentum).add_(updated_proto * (1 - momentum))

#                             # self.prototypes_lung[i].data = updated_proto.data
#                     else:
#                         pass
                    
#                     # 배치 순서대로 다시 합치기
#                     out = torch.cat([out_liver, out_lung], dim=0)
            
#                 else:
#                     # organ 인자에 따라 해당 organ의 prototype 선택 (예: "lung" 또는 "liver")
#                     if organ == "lung":
#                         prototypes = self.prototypes_lung

#                     elif organ == "liver":
#                         prototypes = self.prototypes_liver
#                     else:
#                         # organ 정보가 없으면 기본적으로 liver prototype 사용 (또는 두 organ 모두 업데이트)
#                         prototypes = self.prototypes_liver

#                     # # 예시로, 적용할 feature map 리스트 (예: encoder의 여러 스케일 feature)
#                     # Apply_feature = [out]
#                     # # Multi-Scale Cross Attention 적용
#                     # updated_features = self.multi_scale_cross_attention(Apply_feature, prototypes)
#                     out = self.multi_scale_cross_attention([out], prototypes)
#                     out= out[0]
#                 logits = self.out(out.float())
#                 return logits

    
#         model = AttSwinUNETR(
#         img_size=config['model_params']['img_size'],
#         in_channels=config['model_params']['in_channels'],
#         out_channels=config['model_params']['out_channels'],
#         feature_size=config['model_params']['feature_size'],
#         use_checkpoint=config['model_params']['use_checkpoint']
#         ).to(device)
    
#     if config['model_params']['type'] == "ContrastiveSwinUNETR":
#         from monai.networks.nets import SwinUNETR
#         import torch.nn as nn
#         class ContrastiveSwinUNETR(SwinUNETR):
#             def __init__(self, img_size, in_channels, out_channels, feature_size, use_checkpoint,in_dim=48, hidden_dim=128, out_dim=128):
#                 # SwinUNETR의 초기화 함수 호출
#                 super(ContrastiveSwinUNETR, self).__init__(
#                     img_size=img_size,
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     feature_size=feature_size,
#                     use_checkpoint=use_checkpoint
#                 )
                
#                 # Projection head를 Conv3d로 정의
#                 self.projection_head = nn.Sequential(
#                     nn.Conv3d(in_dim, hidden_dim, kernel_size=1),
#                     nn.BatchNorm3d(hidden_dim),
#                     nn.ReLU(inplace=True),
#                     nn.Conv3d(hidden_dim, out_dim, kernel_size=1)
#                 )

#             def forward(self, x_in):
#                 # SwinUNETR에서 정의된 forward 기능 사용
#                 hidden_states_out = self.swinViT(x_in, self.normalize)
#                 enc0 = self.encoder1(x_in)
#                 enc1 = self.encoder2(hidden_states_out[0])
#                 enc2 = self.encoder3(hidden_states_out[1])
#                 enc3 = self.encoder4(hidden_states_out[2])
#                 dec4 = self.encoder10(hidden_states_out[4])
#                 dec3 = self.decoder5(dec4, hidden_states_out[3])
#                 dec2 = self.decoder4(dec3, enc3)
#                 dec1 = self.decoder3(dec2, enc2)
#                 dec0 = self.decoder2(dec1, enc1)
#                 out = self.decoder1(dec0, enc0)

#                 # Contrastive learning을 위한 중간 feature
#                 features_before_output = out
#                 logits = self.out(out)
#                 # 검증 모드에서는 pixel_embeddings 계산을 생략
#                 if self.training:  # training=True일 때만 contrastive 학습용 features 계산
#                     # Projection head에 통과시키기
#                     pixel_embeddings = self.projection_head(features_before_output)

#                     # L2 정규화를 적용하여 embeddings을 normalize
#                     pixel_embeddings = nn.functional.normalize(pixel_embeddings, p=2, dim=1)
                    
#                     # logits과 pixel_embeddings 반환
#                     return logits, pixel_embeddings
#                 else:
#                     return logits
    
#         model = ContrastiveSwinUNETR(
#         img_size=config['model_params']['img_size'],
#         in_channels=config['model_params']['in_channels'],
#         out_channels=config['model_params']['out_channels'],
#         feature_size=config['model_params']['feature_size'],
#         use_checkpoint=config['model_params']['use_checkpoint']
#         ).to(device)
        
#         class PixelContrastLoss(nn.Module):
#             def __init__(self, config):
#                 super(PixelContrastLoss, self).__init__()
#                 self.temperature = config['CONTRASTIVE']['TEMPERATURE']
#                 self.max_views = config['CONTRASTIVE']['MAX_VIEWS']
#                 self.base_temperature = config['CONTRASTIVE']['BASE_TEMPERATURE']
#                 self.queue_size = config['CONTRASTIVE']['QUEUE_SIZE']
#                 self.dim = config['CONTRASTIVE']['DIM']
#                 self.num_classes = config['CONTRASTIVE']['NUM_CLASSES']
#                 self.ignore_label = 255
#                 self.mode = config['CONTRASTIVE']['MODE']  # 1: 기존 방식, 2: memory bank 방식, 3: hard sampling 방식

#                 if self.mode > 1:
#                     # memory bank (queue) 추가
#                     self.register_buffer("pixel_queue", torch.randn(self.num_classes, self.queue_size, self.dim))
#                     self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
#                     self.register_buffer("pixel_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))

#             @torch.no_grad()
#             def update_queue(self, embeddings, labels):
#                 # embeddings: [Batch, C, D, H, W]
#                 # labels: [Batch, D, H, W]
                
#                 for cls in range(self.num_classes):
#                     # 해당 클래스에 속하는 voxel 선택
#                     cls_indices = (labels == cls).nonzero(as_tuple=True)
#                     if len(cls_indices[0]) == 0:
#                         continue

#                     # 해당 클래스의 피처 추출
#                     cls_embeddings = embeddings[cls_indices]

#                     # 차원 확인 후 정규화
#                     if cls_embeddings.dim() == 1:
#                         cls_embeddings = cls_embeddings.unsqueeze(1)
                    
#                     batch_size = cls_embeddings.shape[0]
#                     ptr = int(self.pixel_queue_ptr[cls])

#                     # 큐 업데이트
#                     if ptr + batch_size > self.queue_size:
#                         self.pixel_queue[cls, ptr:] = nn.functional.normalize(cls_embeddings[:self.queue_size - ptr], p=2, dim=1)
#                         self.pixel_queue_ptr[cls] = 0
#                         self.pixel_queue[cls, :batch_size - (self.queue_size - ptr)] = nn.functional.normalize(cls_embeddings[self.queue_size - ptr:], p=2, dim=1)
#                         self.pixel_queue_ptr[cls] = batch_size - (self.queue_size - ptr)
#                     else:
#                         self.pixel_queue[cls, ptr:ptr + batch_size] = nn.functional.normalize(cls_embeddings, p=2, dim=1)
#                         self.pixel_queue_ptr[cls] = (ptr + batch_size) % self.queue_size

#             def _sample_classes(self, X, y):
#                 # X: [Batch, C, D, H, W]
#                 # y: [Batch, D, H, W]
#                 batch_size, feat_dim = X.shape[0], X.shape[-1]
#                 classes = torch.unique(y)
#                 classes = [clsid for clsid in classes if clsid != self.ignore_label]

#                 if len(classes) == 0:
#                     return None, None

#                 X_class_samples = []
#                 y_class_samples = []

#                 for cls in classes:
#                     cls_indices = (y == cls).nonzero(as_tuple=True)  # 3D로 인덱스 저장
#                     num_samples = min(len(cls_indices[0]), self.max_views)
#                     perm = torch.randperm(len(cls_indices[0]))[:num_samples]
                    
#                     # 각 차원의 인덱스 선택
#                     selected_batch_indices = cls_indices[0][perm]
#                     selected_depth_indices = cls_indices[1][perm]
#                     selected_height_indices = cls_indices[2][perm]
#                     selected_width_indices = cls_indices[3][perm]

#                     # 인덱스를 사용해 X에서 해당 위치의 값들을 추출
#                     X_selected = X[selected_batch_indices, :, selected_depth_indices, selected_height_indices, selected_width_indices]

#                     X_class_samples.append(X_selected)
#                     y_class_samples.append(torch.full((num_samples,), cls, dtype=torch.long).cuda())

#                 if len(X_class_samples) == 0:
#                     return None, None

#                 X_class_samples = torch.cat(X_class_samples, dim=0)
#                 y_class_samples = torch.cat(y_class_samples, dim=0)
#                 return X_class_samples, y_class_samples

#             def _sample_from_memory_bank(self, X, y):
#                 """
#                 Memory bank에서 샘플링하여 contrastive 학습에 사용할 피처들을 반환.
#                 X: [Batch, C, D, H, W]
#                 y: [Batch, D, H, W]
#                 """
#                 X_memory_samples = []
#                 y_memory_samples = []

#                 for cls in range(self.num_classes):
#                     # 해당 클래스의 voxel 인덱스 추출
#                     cls_indices = (y == cls).nonzero(as_tuple=True)

#                     if len(cls_indices[0]) > 0:
#                         # 메모리 뱅크에서 샘플링
#                         memory_indices = torch.randperm(self.queue_size)[:self.max_views]
#                         memory_features = self.pixel_queue[cls][memory_indices].cuda()

#                         # 샘플링된 메모리 피처와 라벨 추가
#                         X_memory_samples.append(memory_features)
#                         y_memory_samples.append(torch.full((memory_features.size(0),), cls, dtype=torch.long).cuda())

#                 if len(X_memory_samples) == 0:
#                     return None, None

#                 # 리스트로 저장된 샘플을 배치 차원으로 합침
#                 X_memory_samples = torch.cat(X_memory_samples, dim=0)
#                 y_memory_samples = torch.cat(y_memory_samples, dim=0)

#                 return X_memory_samples, y_memory_samples

            
#             def _hard_anchor_sampling(self, X, y_hat, y):
#                 batch_size, feat_dim = X.shape[0], X.shape[-1]
#                 classes = torch.unique(y)

#                 if len(classes) == 0:
#                     return None, None

#                 X_class_samples = []
#                 y_class_samples = []

#                 for cls in classes:
#                     hard_indices = ((y_hat == cls) & (y != cls)).nonzero(as_tuple=True)
#                     easy_indices = ((y_hat == cls) & (y == cls)).nonzero(as_tuple=True)

#                     num_hard = len(hard_indices[0])
#                     num_easy = len(easy_indices[0])
#                     n_view = self.max_views

#                     if num_hard >= n_view / 2 and num_easy >= n_view / 2:
#                         num_hard_keep = n_view // 2
#                         num_easy_keep = n_view - num_hard_keep
#                     elif num_hard >= n_view / 2:
#                         num_easy_keep = num_easy
#                         num_hard_keep = n_view - num_easy_keep
#                     elif num_easy >= n_view / 2:
#                         num_hard_keep = num_hard
#                         num_easy_keep = n_view - num_hard_keep
#                     else:
#                         if num_easy + num_hard > 0:
#                             combined_indices = (
#                                 torch.cat((hard_indices[0], easy_indices[0])),
#                                 torch.cat((hard_indices[1], easy_indices[1])),
#                                 torch.cat((hard_indices[2], easy_indices[2])),
#                                 torch.cat((hard_indices[3], easy_indices[3]))
#                             )
#                             if num_easy + num_hard < n_view:
#                                 queue_indices = torch.randperm(self.queue_size)[:(n_view - num_easy - num_hard)]
#                                 queue_features = self.pixel_queue[cls][queue_indices].cuda()
#                                 X_class_samples.append(torch.cat([X[combined_indices], queue_features], dim=0))
#                                 y_class_samples.append(torch.tensor([cls], dtype=torch.long).cuda().clone().detach())
#                             continue
#                         else:
#                             queue_indices = torch.randperm(self.queue_size)[:n_view]
#                             combined_features = self.pixel_queue[cls][queue_indices].cuda()
#                             X_class_samples.append(combined_features)
#                             y_class_samples.append(torch.tensor([cls], dtype=torch.long).cuda().clone().detach())
#                             continue

#                     perm_hard = torch.randperm(num_hard)[:num_hard_keep]
#                     perm_easy = torch.randperm(num_easy)[:num_easy_keep]
#                     selected_hard_indices = (hard_indices[0][perm_hard], hard_indices[1][perm_hard], hard_indices[2][perm_hard], hard_indices[3][perm_hard])
#                     selected_easy_indices = (easy_indices[0][perm_easy], easy_indices[1][perm_easy], easy_indices[2][perm_easy], easy_indices[3][perm_easy])

#                     if len(selected_hard_indices[0]) > 0 or len(selected_easy_indices[0]) > 0:
#                         combined_indices = (
#                             torch.cat((selected_hard_indices[0], selected_easy_indices[0])),
#                             torch.cat((selected_hard_indices[1], selected_easy_indices[1])),
#                             torch.cat((selected_hard_indices[2], selected_easy_indices[2])),
#                             torch.cat((selected_hard_indices[3], selected_easy_indices[3]))
#                         )
#                         combined_features = X[combined_indices]
#                         if combined_features.size(0) < n_view:
#                             queue_indices = torch.randperm(self.queue_size)[:(n_view - combined_features.size(0))]
#                             queue_features = self.pixel_queue[cls][queue_indices].cuda()
#                             combined_features = torch.cat([combined_features, queue_features], dim=0)

#                         X_class_samples.append(combined_features)
#                         y_class_samples.append(torch.tensor([cls], dtype=torch.long).cuda().clone().detach())

#                 if len(X_class_samples) == 0:
#                     return None, None
#                 X_class_samples = torch.stack(X_class_samples, dim=0)
#                 y_class_samples = torch.stack(y_class_samples, dim=0)
#                 return X_class_samples, y_class_samples


#             def _contrastive(self, X_anchor, y_anchor):
#                 anchor_num = X_anchor.shape[0]
#                 anchor_feature = X_anchor

#                 mask = torch.eq(y_anchor.unsqueeze(1), y_anchor.unsqueeze(0)).float().cuda()

#                 anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, anchor_feature.T), self.temperature)
#                 logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#                 logits = anchor_dot_contrast - logits_max.detach()

#                 logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_num).view(-1, 1).cuda(), 0)
#                 mask = mask * logits_mask

#                 neg_mask = 1 - mask
#                 neg_logits = torch.exp(logits) * neg_mask
#                 neg_logits = neg_logits.sum(1, keepdim=True)

#                 exp_logits = torch.exp(logits)
#                 log_prob = logits - torch.log(exp_logits + neg_logits + 1e-10)

#                 mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)

#                 loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#                 loss = loss.mean()

#                 return loss

#             def forward(self, feats, labels, predict=None):
#                 set_track_meta(False)
#                 # feats : [Batch, aug, dim, D, W, H]
#                 batch_size, num_views, feat_dim, depth, height, width = feats.size()

#                 # labels = labels.unsqueeze(1).float().clone()
#                 labels = torch.nn.functional.interpolate(labels, (feats.shape[-3], feats.shape[-2], feats.shape[-1]), mode='nearest')
#                 labels = labels.squeeze(1).long()
                
#                 feats = feats.view(batch_size * num_views, feat_dim, depth, height, width)
#                 # Option 1: 기존 방식
#                 if self.mode == 1:
#                     feats_, labels_ = self._sample_classes(feats, labels)

#             # Option 2: memory bank 방식
#                 elif self.mode == 2:
#                     feats_, labels_ = self._sample_classes(feats, labels)
#                     if feats_ is not None and labels_ is not None:
#                         self.update_queue(feats_, labels_)

#                         # memory bank에서 샘플을 추가로 가져와서 사용
#                         memory_feats, memory_labels = self._sample_from_memory_bank(feats_, labels_)
#                         if memory_feats is not None and memory_labels is not None:
#                             feats_ = torch.cat([feats_, memory_feats], dim=0)
#                             labels_ = torch.cat([labels_, memory_labels], dim=0)

#                 # Option 3: hard sampling 방식
#                 elif self.mode == 3:
#                     predict = predict.argmax(dim=1).long()
#                     feats = feats.permute(0, 2, 4, 3, 1)
#                     feats_, labels_ = self._hard_anchor_sampling(feats, predict, labels)
#                     if feats_ is not None and labels_ is not None:
#                         feats_ = feats_.view(-1, feats_.shape[-1])  # [cls * max_view, dim] 형태로 변환
#                         labels_ = labels_.view(-1)  # [cls * max_view] 형태로 변환
#                         # 각 클래스에 대해 max_view만큼 반복된 labels_ 생성
#                         labels_ = labels_.repeat_interleave(self.max_views)  # [cls * max_view] 형태로 변환
#                         # 각 클래스에 대해 실제 샘플 수에 맞게 라벨을 반복
#                         # labels_ = torch.cat([torch.full((feats_.size(1),), cls, dtype=torch.long).cuda() for cls in torch.unique(labels_)], dim=0)
#                         self.update_queue(feats_, labels_)
                        
#                         # 메모리 bank에서 샘플을 추가로 가져와서 사용
#                         memory_feats, memory_labels = self._sample_from_memory_bank(feats_, labels_)
#                         if memory_feats is not None and memory_labels is not None:
#                             feats_ = torch.cat([feats_, memory_feats], dim=0)  # 현재 샘플과 메모리 샘플 결합
#                             labels_ = torch.cat([labels_, memory_labels], dim=0)  # 라벨도 결합

#                 if feats_ is None or labels_ is None:
#                     return torch.tensor(0.0).cuda()

#                 loss = self._contrastive(feats_, labels_)
#                 set_track_meta(True)
#                 return loss

#     return model


# def load_model_weights(model, weight_path: str, device: torch.device):
#     model.load_state_dict(torch.load(weight_path, map_location=device)['model_state_dict'], strict=False)
#     # model.load_from(weights=weight)
#     logger.info(f"Model weights loaded from: {weight_path}")  # 모델 로드 경로 출력
#     logger.info("Loading checkpoint... Well")

### 3. Transform & Dataset
def get_test_transforms(config: Dict, organs: Union[str, List[str]]):
    """
    organ == "liver" / "lung" 에 따라 spacing 및 intensity range 조정
    2025-05-07 organ 입력받으면 그거에 따라 반환토록함
    """
    if isinstance(organs, str):
        organs = [organs]
    transforms = {}
    liver_test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=tuple(config['transforms']['spacing']),
                    mode=("bilinear"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=config['transforms']['liver']['a_min'], a_max=config['transforms']['liver']['a_max'], b_min=0.0, b_max=1.0, clip=True
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

    liver_label_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(config['transforms']['spacing']),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=config['transforms']['liver']['a_min'], a_max=config['transforms']['liver']['a_max'], b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    lung_test_transforms= Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(
                    keys=["image"],
                    pixdim=tuple(config['transforms']['spacing']),
                    mode=("bilinear"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=config['transforms']['lung']['a_min'], a_max=config['transforms']['lung']['a_max'], b_min=0.0, b_max=1.0, clip=True
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
    
    lung_label_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(config['transforms']['spacing']),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=config['transforms']['lung']['a_min'],
                a_max=config['transforms']['lung']['a_max'], b_min=0.0, b_max=1.0, clip=True
            ),
            # label 1을 label 0으로, label 2는 1로 변경 유지
            # MapLabelValued(keys="label", orig_labels=[0, 1, 2], target_labels=[0, 0, 1]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # MapLabelValued(keys="label", orig_labels=[1], target_labels=[2]),  # 종양을 2로 설정
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    liver_post_pred = Compose([
        Invertd(
            keys="pred",  # 예측값에만 Invertd 적용
            transform=liver_test_transforms,  # 원본과 동일한 전처리
            orig_keys="image",  # 원본 데이터 키
            # meta_keys="pred_meta_dict",  # 메타데이터 키
            # meta_key_postfix="meta_dict",  # 메타데이터 접미사
            nearest_interp=True,  # 가장 가까운 이웃 보간
            to_tensor=True,  # 텐서로 변환
            device="cpu",
        ),
        AsDiscreted(keys="pred",argmax=True,to_onehot=config['model_params']['out_channels']),
        AsDiscreted(keys="label",to_onehot=config['model_params']['out_channels'])
    ])

    lung_post_pred = Compose([
        Invertd(
            keys="pred",  # 예측값에만 Invertd 적용
            transform=lung_test_transforms,  # 원본과 동일한 전처리
            orig_keys="image",  # 원본 데이터 키
            # meta_keys="pred_meta_dict",  # 메타데이터 키
            # orig_meta_keys="image_meta_dict",
            # meta_key_postfix="meta_dict",  # 메타데이터 접미사
            nearest_interp=True,  # 가장 가까운 이웃 보간
            to_tensor=True,  # 텐서로 변환
            device="cpu",
        ),
        AsDiscreted(keys="pred",argmax=True,to_onehot=config['model_params']['out_channels']),
        AsDiscreted(keys="label",to_onehot=config['model_params']['out_channels'])
    ])

    # transforms["liver"] = {
    #     "test_tf": liver_test_transforms,
    #     "label_tf": liver_label_transforms,
    #     "post_tf": liver_post_pred,
    # }
    # transforms["lung"] = {
    #     "test_tf": lung_test_transforms,
    #     "label_tf": lung_label_transforms,
    #     "post_tf": lung_post_pred,
    # }

    for organ in organs:
        try:
            test_tf  = locals()[f"{organ}_test_transforms"]
            label_tf = locals()[f"{organ}_label_transforms"]
            post_tf  = locals()[f"{organ}_post_pred"]
        except KeyError as e:
            raise ValueError(f"전처리 정의가 누락된 organ: '{organ}' → {e}")

        transforms[organ] = {
            "test_tf": test_tf,
            "label_tf": label_tf,
            "post_tf": post_tf,
        }

    return transforms

#### Dataset및 dataloader 구성
def load_datalist(config: Dict, transform_dict : Dict, output_dir: str) -> Dict[str, Dict[str, DataLoader]]:
    '''
    2025-05-07
    장기에 따라 각각 진행토록 바꾸고 싶은데, 그전까지만 진행
    '''
    dataloaders = {}
    # 이미지 및 레이블 경로가 저장된 txt 파일을 불러오기
    def load_lungfile_from_txt(image_txt_path, image_dir):
        image_paths = []
        label_paths = []
        
        # 이미지 경로 읽기
        with open(image_txt_path, 'r') as f:
            for line in f:
                image_paths.append(os.path.join(image_dir, line.strip()))
                mask_filename = os.path.join(image_dir, line.strip()).replace('imagesTr', 'labelsTr')
                label_paths.append(mask_filename)
        return image_paths, label_paths

    def load_liverfile_from_txt(image_txt_path, image_dir):
        image_paths = []
        label_paths = []
        
        # 이미지 경로 읽기
        with open(image_txt_path, 'r') as f:
            for line in f:
                image_paths.append(os.path.join(image_dir, line.strip()))
                mask_filename = os.path.join(image_dir, line.strip()).replace('imagesTr', 'processed_labels')
                label_paths.append(mask_filename)
        return image_paths, label_paths

    # Train 데이터 로드
    liver_test_images, liver_test_labels = load_liverfile_from_txt(config['data']['liver_test_image_txt'], config['data']['liver_image_dir'])
    lung_test_images, lung_test_labels = load_lungfile_from_txt(config['data']['lung_test_image_txt'], config['data']['lung_image_dir'])

    # Dataset 구성
    liver_dataset_json = {
        "labels": {
            "0": "background",
            "1": "cancer",
        },
        "tensorImageSize": "3D",
        "test": [{"image": img, "label": lbl} for img, lbl in zip(liver_test_images, liver_test_labels)]
    }

    lung_dataset_json = {
        "labels": {
            "0": "background",
            "1": "cancer",
        },
        "tensorImageSize": "3D",
        "test": [{"image": img, "label": lbl} for img, lbl in zip(lung_test_images, lung_test_labels)],
        
    }

    # JSON 파일 저장
    with open(os.path.join(output_dir, 'liver_test_dataset.json'), 'w') as outfile:
        json.dump(liver_dataset_json, outfile)
    with open(os.path.join(output_dir, 'lung_test_dataset.json'), 'w') as outfile:
        json.dump(lung_dataset_json, outfile)
    
    liver_test_files = load_decathlon_datalist(os.path.join(output_dir, 'liver_test_dataset.json'), True, "test")
    lung_test_files = load_decathlon_datalist(os.path.join(output_dir, 'lung_test_dataset.json'), True, "test")
    
    # liver_test_ds = PersistentDataset(data=liver_test_files, transform=liver_test_transforms, cache_dir="/data/hyungseok/Swin-UNETR/data_cache/Liver_Lung/Liver/test/raw")
    # lung_test_ds = PersistentDataset(data=lung_test_files, transform=lung_test_transforms,cache_dir="/data/hyungseok/Swin-UNETR/data_cache/Liver_Lung/Lung/test/raw")
    # liver_label_test_ds = PersistentDataset(data=liver_test_files, transform=liver_label_transforms,cache_dir="/data/hyungseok/Swin-UNETR/data_cache/Liver_Lung/Liver/test/process")
    # lung_label_test_ds = PersistentDataset(data=lung_test_files, transform=lung_label_transforms,cache_dir="/data/hyungseok/Swin-UNETR/data_cache/Liver_Lung/Lung/test/process")                                                       

    # liver_test_transforms, liver_label_transforms, lung_test_transforms, lung_label_transforms =get_test_transforms(config,organ="all")
    
    liver_test_transforms, liver_label_transforms = transform_dict["liver"]["test_tf"], transform_dict["liver"]["label_tf"]
    lung_test_transforms, lung_label_transforms = transform_dict["lung"]["test_tf"], transform_dict["lung"]["label_tf"]

    liver_test_ds = Dataset(data=liver_test_files, transform=liver_test_transforms)
    lung_test_ds = Dataset(data=lung_test_files, transform=lung_test_transforms)
    #Overlay용
    liver_label_test_ds = Dataset(data=liver_test_files, transform=liver_label_transforms)
    lung_label_test_ds = Dataset(data=lung_test_files, transform=lung_label_transforms)

    
    liver_test_loader = DataLoader(liver_test_ds, num_workers=8, batch_size=1,worker_init_fn=worker_init_fn)
    lung_test_loader = DataLoader(lung_test_ds, num_workers=8, batch_size=1,worker_init_fn=worker_init_fn)
    liver_label_test_loader = DataLoader(liver_label_test_ds, num_workers=8, batch_size=1,worker_init_fn=worker_init_fn)
    lung_label_test_loader = DataLoader(lung_label_test_ds, num_workers=8, batch_size=1,worker_init_fn=worker_init_fn)
    
    # if organ =="liver":
    #     return liver_test_ds, liver_label_test_ds
    # elif organ == "lung" :
    #     return lung_test_ds, lung_label_test_ds
    # else:
    #     return liver_test_ds, liver_label_test_ds, lung_test_ds, lung_label_test_ds

    dataloaders["liver"] = {
        "test_loader": liver_test_loader,
        "label_test_loader": liver_label_test_loader,
    }
    dataloaders["lung"] = {
        "test_loader": lung_test_loader,
        "label_test_loader": lung_label_test_loader,
    }
    return dataloaders

### 4. Inference & Postprocessing
class ModelWrapper:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.model_type = config['model_params']['type']

    def infer(self, test_inputs, organ=None):
        """
        모델 종류에 따라 inference 방식을 분기한다.
        모든 경우, logits (예측값)만 반환되도록 통일한다.
        """
        # test_inputs : 테스트 이미지
        self.model.eval()
        with torch.no_grad():
            if self.model_type == "SwinUnetr":
                # 기본 모델        
                return sliding_window_inference(
                        test_inputs, self.config['model_params']['img_size'], 4, self.model
                    )
            
            elif self.model_type == "AttSwinUnetr":
                return self.model(test_inputs, label=None, organ=organ)

            elif self.model_type == "ContrastiveSwinUNETR":
                # 모델내 eval일때, 동작하도록 설계 되어 있음
                return sliding_window_inference(
                        test_inputs, self.config['model_params']['img_size'], 4, self.model
                    )
            else:
                raise ValueError(f"[Wrapper] Unknown model type: {self.model_type}")


def run_inference(model, batch, config: Dict, organ: str):
    """
    AMP, sliding window inference 포함.
    organ-aware forward 수행.
    """
    pass

def postprocess(batch, post_pred_transform):
    """
    Invertd + AsDiscreted 적용
    """
    single_item = [post_pred_transform(i) for i in decollate_batch(batch)]
    test_outputs_convert, test_labels_convert = from_engine(["pred", "label"])(single_item)

    # 둘의 차원이 맞는지 예상한대로 되는지등의 검증함수가 있으면 더 좋을듯
    # 주석으로 각 차원과정을 써놓으면 굳이 디버깅없이 알수 있을듯 - 사용 함수가 일반적이지 않아서 설명이 더 있으면 좋을듯

    '''
    2025-05-07
    확인결과 decollate_batch하면 기존 Monai의 dict구조 (image label ...)깨지며 한 샘플당 리스트안의 dict구조로 single_item 반환
    from_engine은 그러한 구조내에서 output과 label 반환 함수일뿐
    따라서 아래 부분 전에 원하는 형태로 변환 필요
    형태를 적어놓자면
    batch - dict
    from_engine의 결과 - list
    아래처럼 0번꺼낸 결과 - <class 'monai.data.meta_tensor.MetaTensor'>
    '''
    test_outputs_convert = test_outputs_convert[0].unsqueeze(0)
    test_labels_convert =test_labels_convert[0].unsqueeze(0)
   
    return test_outputs_convert, test_labels_convert

# '''
# Class로
# organ의 여러 metric 상태 유지
# batch-wise 처리 - 환자별 처리
# organ별로 분리해서 기록 - organ을 key로 한 dict로 관리
# MONAI Metric 연동 - 내부에 유지하며 organ 변경시, 자동 reset 가능
# 확장성 - 추후 클래스에 method추가
# '''

# class MetricManager:
#     def __init__(self, config, device):
#         self.config = config
#         self.device = device
#         self.organs = ["liver", "lung"]
#         self.records = {organ: [] for organ in self.organs}

#         # Monai metrics (선택적으로 사용)
#         from monai.metrics import DiceMetric
#         self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
#         from monai.metrics import HausdorffDistanceMetric, compute_average_surface_distance
#         self.compute_asd = compute_average_surface_distance

#     def update(self, organ: str, case_name: str, pred, gt, meta_info: dict):
#         """
#         한 케이스(환자)에 대한 예측 결과를 기록.
#         meta_info: dict 형태 (예: 위치, 컴포넌트 수 등)
#         """
#         self.dice_metric.reset()
#         self.dice_metric(y_pred=pred, y=gt)
#         dice_score = self.dice_metric.aggregate().item()
#         # Convert to binary numpy
#         pred_np = pred[0, 1].cpu().numpy()
#         gt_np = gt[0, 1].cpu().numpy()

#         # Additional metrics
#         asd_score = self.compute_asd(y_pred=pred, y=gt, include_background=False).item()
#         comp_stats = self._component_wise_metric(pred_np, gt_np)
#         pixel_stats = self._pixel_wise_metric(pred_np, gt_np)

        
#         # 저장 구조: case 단위 dict
#         case_result = {
#             "case_name": case_name,
#             "dice": dice_score,
#             "asd": asd_score,
#             "component_recall": comp_stats["recall_obj"],
#             **comp_stats,
#             **pixel_stats,
#             **meta_info
#         }

#         self.records[organ].append(case_result)
#         # ✅ 로그 출력
#         logger.info(
#             f"[{organ}] Case '{case_name}' — "
#             f"Dice: {dice_score:.4f}, ASD: {asd_score:.2f}"
#         )

#     def _component_check(self, mask):
#         from scipy import ndimage
#         return ndimage.label(mask)

#     def _component_wise_metric(self, pred, gt, voxel_threshold=8):
#         gt_lab, gt_num = self._component_check(gt)
#         pred_lab, pred_num = self._component_check(pred)

#         tp_obj, fn_obj, fp_obj = 0, 0, 0
#         for cid in range(1, gt_num + 1):
#             mask_gt = gt_lab == cid
#             if mask_gt.sum() < voxel_threshold:
#                 continue
#             if np.any(pred_lab[mask_gt] > 0):
#                 tp_obj += 1
#             else:
#                 fn_obj += 1
#         for pid in range(1, pred_num + 1):
#             mask_pred = pred_lab == pid
#             if mask_pred.sum() < voxel_threshold:
#                 continue
#             if not np.any(gt_lab[mask_pred] > 0):
#                 fp_obj += 1

#         recall = tp_obj / (tp_obj + fn_obj) if (tp_obj + fn_obj) > 0 else 0.0
#         precision = tp_obj / (tp_obj + fp_obj) if (tp_obj + fp_obj) > 0 else 0.0

#         return {
#             "tp_obj": tp_obj,
#             "fn_obj": fn_obj,
#             "fp_obj": fp_obj,
#             "recall_obj": recall,
#             "precision_obj": precision,
#         }
    
#     def _pixel_wise_metric(self, pred, gt):
#         tp = np.logical_and(pred == 1, gt == 1).sum()
#         fn = np.logical_and(pred == 0, gt == 1).sum()
#         fp = np.logical_and(pred == 1, gt == 0).sum()
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         return {
#             "tp": tp,
#             "fn": fn,
#             "fp": fp,
#             "recall": recall,
#             "precision": precision,
#         }
    
#     def summarize(self):
#         """
#         전체 결과 반환: organ → list of dict
#         DataFrame 등 저장에 바로 사용 가능
#         organ별 평균/표준편차 요약 + 전체 df 병합 반환
#         """
#         import pandas as pd
#         summary = {}
#         dfs = {}
#         for organ, records in self.records.items():
#             df = pd.DataFrame(records)
#             dfs[organ] = df
#             summary[organ] = df.describe().T[["mean", "std"]]  # 수치형 필드만
#         return summary, dfs

#     def summarize_scores(self):
#         """
#         organ → metric → 평균값
#         """
#         import pandas as pd
#         results = {}
#         for organ, records in self.records.items():
#             df = pd.DataFrame(records)
#             results[organ] = {k: df[k].mean() for k in df.columns if df[k].dtype != "object"}
#         return results


#     def save_csv(self, output_dir):
#         import pandas as pd, os
#         for organ, records in self.records.items():
#             df = pd.DataFrame(records)
#             df.to_csv(os.path.join(output_dir, f"{organ}_metrics.csv"), index=False)

### 6. Visualization & Save

def save_3d_results(test_outputs, save_dir, organ_name, case_name, meta=None):
    """
    모델 출력 결과를 NIfTI 형식으로 저장.
    
    Args:
        test_outputs (torch.Tensor): 모델 출력 결과 (B, C, H, W, D) 형태.
        save_dir (str): 결과를 저장할 디렉토리 경로.
        organ_name (str): 저장할 장기 이름 ("liver", "lung" 등).
        case_name (str): 파일 이름으로 사용할 케이스 이름.
        meta (dict, optional): 추가 메타 정보. 예: {'affine': np.array, 'pixdim': tuple, 'descrip': str}
    
    Returns:
        None
    """
    # 저장 디렉토리 생성
    organ_save_dir = os.path.join(save_dir, organ_name)
    os.makedirs(organ_save_dir, exist_ok=True)
    save_path = os.path.join(organ_save_dir, f"{case_name}_prediction.nii.gz")
    # 모델 출력의 첫 번째 배치, 채널 데이터 추출 (B=1, C=1 가정)
    data = test_outputs.cpu().numpy()[0, 0]

    # # meta에서 affine 정보 사용, 없으면 기본값 np.eye(4)
    # affine = meta["affine"] if meta and "affine" in meta else np.eye(4)
    # 메타에서 affine 정보 사용 (batch 차원 제거 주의)
    if meta and "affine" in meta:
        # affine이 (1,4,4)일 수도 있으니 squeeze
        affine = meta["affine"]
        if torch.is_tensor(affine):
            affine = affine.cpu().numpy()
        if affine.ndim == 3:  # (1,4,4) 형태라면
            affine = affine[0]
    else:
        affine = np.eye(4)
    # 헤더에 데이터 쉐입을 먼저 설정
    
    # NIfTI 헤더 만들기
    header = nib.Nifti1Header()
    header.set_data_shape(data.shape)
    if meta and "pixdim" in meta:
        pixdim = meta["pixdim"]
        # 텐서인 경우 변환
        if torch.is_tensor(pixdim):
            pixdim = pixdim.cpu().numpy()
        # (1, 8) 처럼 되어 있으면 squeeze
        pixdim = np.squeeze(pixdim)  # (8,)
        # 첫 번째 요소는 제외하고 x,y,z 해상도만 꺼내 쓰기
        # 보통 음수가 들어가는 경우는 좌우축 반전표시 등일 수 있으므로
        # 절댓값을 씌워서 저장하거나, 그대로 넘길지 결정
        zooms_3d = np.abs(pixdim[1:4])
        header.set_zooms(zooms_3d)
    
    
    
    nifti_img = nib.Nifti1Image(data, affine, header)
    nib.save(nifti_img, save_path)
    
    print(f"Saved 3D result for {organ_name}: {save_path}")

def render_overlay(
    image, label, prediction, show=False, index =0, organ_name="Unknown", case_name ="000", out_file=None, colormap="spring", save_individual=False
):
    """
    Render overlay images combining the input image, ground truth label, and prediction.
    Provides options for saving individual overlays or a combined overlay.

    Args:
        image (torch.Tensor): Input image to blend with label and prediction.
        # (B,C,H,W,D)
        label (torch.Tensor): Ground truth label.
        # (B,C,H,W,D)
        prediction (torch.Tensor): Model prediction.
        # (B,C,H,W,D)
        show (bool): Whether to display the figure. Default is False.
        out_file (str): Path to save the combined output figure. Default is None (no saving).
        colormap (str): Colormap for blending. Default is "spring".
        save_individual (bool): Whether to save individual overlays for image + label and image + prediction. Default is False.

    Returns:
        fig (matplotlib.figure.Figure): The rendered figure.
    """

    # index가 None이면 label에서 sagittal 방향(세번째 축) 기준 tumor voxel 수가 가장 많은 슬라이스 선택
    if index is None:
        label_np = label.squeeze().cpu().numpy()  # (H, W, D) 또는 (C, H, W, D)
        if label_np.ndim == 4:  # 채널 차원이 남아있는 경우 (C, H, W, D), 여기서 채널 1이 tumor class라고 가정
            tumor_mask = label_np[1]
        else:
            tumor_mask = label_np
        # 각 슬라이스마다 tumor 픽셀 수 계산 (H, W를 합산)
        tumor_counts = tumor_mask.sum(axis=(0, 1))
        index = int(tumor_counts.argmax())
        out_file = os.path.join(out_file, f"{organ_name}_image")
        os.makedirs(out_file, exist_ok=True)
        out_file = os.path.join(out_file,f"{case_name}_{index}.png")
    blue_cmap = matplotlib.colors.ListedColormap(["black", "blue"])
    red_cmap = matplotlib.colors.ListedColormap(["black", "red"])

    # Blend images for individual overlays using blend_images
    correct_blend = blend_images(image=image.cpu(), label=label.cpu(), alpha=0.5, cmap=red_cmap, rescale_arrays=False)
    # (C,B,H,W,D)
    # predict_blend = blend_images(image=image.cpu(), label=prediction.cpu(), alpha=0.5, cmap=blue_cmap, rescale_arrays=False)
    # (C,B,H,W,D)
    # Convert tensors to numpy arrays for combined overlay
    image_np = image.squeeze().cpu().numpy()
    label_np = label.squeeze().cpu().numpy()
    prediction_np = prediction.squeeze().cpu().numpy()

    # Normalize the image to [0, 1] for better visualization
    image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))

    # # Create RGB overlay for combined view
    # overlay = np.zeros((*image_np.shape, 3), dtype=np.float32)
    # overlay[..., 0] = label_np  # Red channel for Ground Truth
    # overlay[..., 2] = prediction_np  # Blue channel for Prediction
    # overlay = np.clip(overlay, 0, 1)  # Normalize RGB values to range [0, 1]
    
    # RGBA 마스크 (H, W, 4)
    prediction_mask = np.zeros((image_np[:,:,index].shape[0], image_np[:,:,index].shape[1], 4), dtype=np.float32)
    # 파란색 채널에 pred
    prediction_mask[..., 2] = prediction_np[..., index]   # Blue channel

    # 마스크가 있는 곳만 alpha=0.5, 배경은 alpha=0
    mask_nonzero = (prediction_np[..., index] > 0)
    prediction_mask[mask_nonzero, 3] = 0.5

    # RGBA 마스크 (H, W, 4)
    rgba_mask = np.zeros((image_np[:,:,index].shape[0], image_np[:,:,index].shape[1], 4), dtype=np.float32)
    # 빨간색 채널에 label
    rgba_mask[..., 0] = label_np[..., index]  # Red channel
    # 파란색 채널에 pred
    rgba_mask[..., 2] = prediction_np[..., index]   # Blue channel

    # 마스크가 있는 곳만 alpha=0.5, 배경은 alpha=0
    mask_nonzero = (label_np[..., index] > 0) | (prediction_np[..., index] > 0)
    rgba_mask[mask_nonzero, 3] = 0.5
    rot_img = np.rot90(image_np[:,:,index], k=1)
    rot_prediction = np.rot90(prediction_mask,k=1)
    rot_overlay = np.rot90(rgba_mask, k=1)

    # Set up the figure using Matplotlib
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    # 전체 그림의 제목(환자 이름과 슬라이스 번호) 달기
    fig.suptitle(f"{case_name} (slice={index})", fontsize=20)
    axes[0].imshow(rot_img, cmap="gray")
    axes[0].set_title("Image")
    # Plot individual overlays using blend_images
    # axes[0].imshow(torch.moveaxis(correct_blend[:, :, :,:, index], 0, -1).squeeze())
    slice_blend = correct_blend[:, 0, :, :, index]       # shape (3, H, W)
    slice_blend = slice_blend.permute(1, 2, 0).cpu().numpy()        # shape (H, W, 3)
    slice_blend = np.rot90(slice_blend, k=1)
    axes[1].imshow(slice_blend)
    axes[1].set_title("Image + Ground Truth Label Overlay")
    
    # axes[1].imshow(torch.moveaxis(predict_blend[:, :, :, :,index], 0, -1).squeeze())
    # slice_blend = predict_blend[:, 0, :, :, index]       # shape (3, H, W)
    # slice_blend = slice_blend.permute(1, 2, 0)         # shape (H, W, 3)
    # axes[1].imshow(slice_blend.cpu().numpy())

    axes[2].imshow(rot_img, cmap="gray")
    axes[2].imshow(rot_prediction)
    axes[2].set_title("Image + Prediction Overlay")

    # Plot combined overlay
    axes[3].imshow(rot_img, cmap="gray")
    axes[3].imshow(rot_overlay)
    axes[3].set_title("Combined Overlay: Ground Truth (Red) + Prediction (Blue)")

    # Remove axes for clean visualization
    for ax in axes:
        ax.axis("off")

    # Adjust layout to reduce whitespace
    plt.tight_layout()
    # 그리고 난 뒤, 상단 여백( top )을 조금 늘려줍니다.
    plt.subplots_adjust(top=0.85)

    # Save the figure if an output file is provided
    if out_file:
        plt.savefig(out_file, bbox_inches="tight")
        print(f"Overlay saved at: {out_file}")

    if save_individual:
        # Image + Ground Truth Overlay 저장
        # label_image = torch.moveaxis(correct_blend[:, :, :, :,index], 0, -1).squeeze().cpu().numpy()
        label_image = correct_blend[:, 0, :, :, index].permute(1, 2, 0).cpu().numpy()
        label_image = np.rot90(label_image, k=1)       # shape (3, H, W)
        plt.imsave(out_file.replace(".png", "_label.png"), label_image, cmap=None)

        # Image + Prediction Overlay 저장
        # prediction_image = torch.moveaxis(predict_blend[:, :, :, :, index], 0, -1).squeeze().cpu().numpy()
        # prediction_image = predict_blend[:, 0, :, :, index].permute(1, 2, 0).cpu().numpy()       # shape (3, H, W)
        # plt.imsave(out_file.replace(".png", "_prediction.png"), prediction_image, cmap=None)
        fig2, ax2 = plt.subplots(figsize=(6,6))
        ax2.imshow(rot_img, cmap="gray")
        ax2.imshow(rot_prediction)
        ax2.set_title("Prediction Only")
        prediction_all_path = out_file.replace(".png", "_prediction_all.png")
        fig2.savefig(prediction_all_path, bbox_inches="tight")
        plt.close(fig2)

        # Combined Overlay 저장 (axes[2]의 전체 플롯 저장)
        fig3, ax3 = plt.subplots(figsize=(6,6))
        ax3.imshow(rot_img, cmap="gray")
        ax3.imshow(rot_overlay)
        ax3.set_title("Combined Overlay Only")
        overlay_all_path = out_file.replace(".png", "_overlay_all.png")
        fig3.savefig(overlay_all_path, bbox_inches="tight")
        plt.close(fig3)

        # overlay_all_path = out_file.replace(".png", "_overlay_all.png")
        # axes[2].figure.savefig(overlay_all_path, bbox_inches="tight")

        print(f"Individual overlays and combined overlay saved at: {out_file}")


    # Show or close the figure
    if show:
        plt.show()
    else:
        plt.close(fig)

    return


### 7. Main Execution Loop
def evaluate_dataset(model, dataloader, post_transforms, organ: str, config: Dict, device: torch.device):
    """
    전체 루프: for batch in dataloader → inference → postprocess → metric → save
    """
    pass

def main():
    config = load_config("/data/hyungseok/Swin-UNETR/api/test.yaml")
    set_seed_and_env(config)
    
    # 테스트 환경 설정
    ckpt_name = os.path.splitext(os.path.basename(config['test_params']['ckpt_dir']))[0]
    output_dir = os.path.join(os.path.dirname(config['test_params']['ckpt_dir']), f"test_post_vis_result_total_D{ckpt_name}")
    os.makedirs(output_dir, exist_ok=True)

    # 로그 파일 설정
    setup_logging(output_dir)
    logger = get_logger()
    
    save_config_as_json(config, output_dir)
    current_script_name = __file__
    save_current_code(output_dir,current_script_name)
    # log_experiment_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config, device)
    load_model_weights(model, config['test_params']['ckpt_dir'], device)
    organ_list =["liver", "lung"]
    tf_dict = get_test_transforms(config, organs=organ_list)
    loader_dict = load_datalist(config, tf_dict,output_dir)


    organ_modules = {}
    for organ in organ_list:
        organ_modules[organ] = {
            "test_tf": tf_dict[organ]["test_tf"],
            "label_tf": tf_dict[organ]["label_tf"],
            "post_tf": tf_dict[organ]["post_tf"],
            "test_loader": loader_dict[organ]["test_loader"],
            "label_loader": loader_dict[organ]["label_test_loader"],
        }
    metric_mgr = MetricManager(config, device)
    wrapper = ModelWrapper(model, config)
    for organ, module in organ_modules.items():
        for batch_raw, batch_processed in zip(module["test_loader"], module["label_loader"]):
            # torch.cuda.empty_cache()
            case_name = os.path.basename(batch_raw["image_meta_dict"]["filename_or_obj"][0]).split(".")[0]
            test_inputs  = batch_raw["image"].to(device)
            batch_raw["pred"]  = wrapper.infer(test_inputs, organ=organ)
            test_outputs_convert, test_labels_convert = postprocess(batch_raw, module["post_tf"])
            ###
            # 기록용
            # case_name = "liver_118"
            # test_inputs.shape = (1, 1, 381, 351, 640)
            # batch_raw["pred"].shape = (1, 2, 381, 351, 640)
            # test_outputs_convert,test_labels_convert.shape =(1, 2, 512, 512, 427)
            # 
            meta_info = {
                "spacing": batch_raw["image_meta_dict"]["pixdim"][0],  # (z, y, x)
                "pre_image_shape": list(test_inputs.shape[2:]),             # (D,H,W)
                "post_image_shape": list(test_outputs_convert.shape[2:]),             # (D,H,W)
                "case_name" : case_name                                 # 환자번호
            }
            metric_mgr.update(organ, case_name, test_outputs_convert, test_labels_convert, meta_info=meta_info)
            render_overlay(
                image=batch_processed["image"],
                label=batch_processed["label"],
                prediction=batch_raw["pred"].argmax(dim=1, keepdim=True),
                index = None,
                organ_name=organ,
                case_name=case_name,
                out_file=output_dir,
                save_individual=True
            )
            save_3d_results(test_outputs_convert.argmax(dim=1, keepdim=True), save_dir=output_dir, organ_name=organ,case_name=case_name,meta=batch_raw["image_meta_dict"])        
            

    # 저장
    metric_mgr.save_csv(output_dir)

    # 통계 요약 확인
    # summary, dfs = metric_mgr.summarize()
    # print(summary["liver"])  # liver organ의 지표 요약(mean/std)
    # print(summary["lung"])  # liver organ의 지표 요약(mean/std)

    # 요약 (수치만 보기 위한 핵심 로그)
    scores = metric_mgr.summarize_scores()
    for organ in organ_list:
        logger.info(f"[Summary] {organ.upper()} — " + ", ".join([
            f"{k}: {v:.4f}" for k, v in scores[organ].items()
        ]))


        # evaluate_dataset은 있어야 싶네
        # evaluate_dataset(model, dataloader, post_transforms, organ, config, device)
        # evaluate dataset함수는 추후에 실험 모델끼리 비교하는것을 임시로 만들어놓는식으로 쓰는게 좋을듯

if __name__ == "__main__":
    main()
