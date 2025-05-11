from __future__ import print_function
import os
import numpy as np
import pandas as pd
import plotly.express as px

import argparse
import numpy as np
import torch
import os
import torch.nn as nn
from tqdm import tqdm

from typing import Dict, List, Tuple, Any
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from scipy.spatial import distance
from monai.transforms import Compose, AsDiscrete, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, CropForegroundd, Orientationd, Spacingd, EnsureTyped, RandFlipd,RandShiftIntensityd, RandCropByPosNegLabeld, MapLabelValued,RandScaleIntensityd,RandRotate90d, ToTensord
from monai.data import (
    PersistentDataset,
    ThreadDataLoader,
    DataLoader,
    Dataset,
    CacheDataset,
    DistributedSampler,
    list_data_collate,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference

import sys
sys.path.append('/data/hyungseok/Swin-UNETR')
from utils.my_utils import load_config
from utils.seed import set_seed_and_env, set_all_random_states
from utils.logger import log_experiment_config, get_logger, setup_logging, save_config_as_json, save_current_code
parser = argparse.ArgumentParser(description="Swin UNETR training")
parser.add_argument('--config', type=str, default="/data/hyungseok/Swin-UNETR/api/tsne.yaml", help="Path to the YAML config file")
args = parser.parse_args()
# YAML 설정 파일 로드

# 학습 설정 로드
config_path = args.config
config = load_config(config_path)
from datetime import datetime
# 테스트 환경 설정
date = datetime.now().strftime("%Y-%m-%d_%H-%M")
output_dir = os.path.join("/data/hyungseok/Swin-UNETR/scripts/tsne_debug_analysis",date)
os.makedirs(output_dir, exist_ok=True)
setup_logging(output_dir)
logger = get_logger
save_config_as_json(config, output_dir)
current_script_name = __file__
save_current_code(output_dir,current_script_name)
### 시드 고정
set_seed_and_env(config)
USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else "cpu")

def set_loader(config):
    num_samples = config['transforms']['num_samples']
    liver_train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            # 이미지와 mask 로드 
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd를 사용하여 등방성 voxel spacing 1.0mm로 보간
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(config['transforms']['spacing']),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config['transforms']['liver']['a_min'],
                a_max=config['transforms']['liver']['a_max'],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(config['train_params']['spatial_size']),
                pos=config['transforms']['liver']['RandCropByPosNegLabeld_params']['pos'],
                neg=config['transforms']['liver']['RandCropByPosNegLabeld_params']['neg'],
                num_samples=config['transforms']['num_samples'],
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=config['transforms']['liver']['rand_flip_prob'],
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=config['transforms']['liver']['rand_flip_prob'],
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=config['transforms']['liver']['rand_flip_prob'],
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=config['transforms']['liver']['rand_rotate_prob'],
                max_k=3,
            ),
            RandScaleIntensityd(
                keys=["image"],
                factors=config['transforms']['liver']['rand_scale_intensity_factor'], 
                prob=config['transforms']['liver']['rand_scale_intensity_prob']),
            RandShiftIntensityd(
                keys=["image"],
                offsets=config['transforms']['liver']['rand_shift_intensity_offset'],
                prob=config['transforms']['liver']['rand_shift_intensity_prob'],
            ),
            EnsureTyped(keys=["image", "label"]),  # ✅ MetaTensor 변환 추가
        ]
    )

    liver_val_transforms = Compose(
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
            EnsureTyped(keys=["image", "label"]),  # ✅ MetaTensor 변환 추가
        ]
    )

    lung_train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd를 사용하여 등방성 voxel spacing 1.0mm로 보간
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(config['transforms']['spacing']),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config['transforms']['lung']['a_min'],
                a_max=config['transforms']['lung']['a_max'],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # label 1을 label 0으로, label 2는 1로 변경 유지
            # MapLabelValued(keys="label", orig_labels=[0, 1, 2], target_labels=[0, 0, 1]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(config['train_params']['spatial_size']),
                pos=config['transforms']['lung']['RandCropByPosNegLabeld_params']['pos'],
                neg=config['transforms']['lung']['RandCropByPosNegLabeld_params']['neg'],
                num_samples=config['transforms']['num_samples'],
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=config['transforms']['lung']['rand_flip_prob'],
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=config['transforms']['lung']['rand_flip_prob'],
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=config['transforms']['lung']['rand_flip_prob'],
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=config['transforms']['lung']['rand_rotate_prob'],
                max_k=3,
            ),
            RandScaleIntensityd(keys=["image"], factors=config['transforms']['lung']['rand_scale_intensity_factor'], prob=config['transforms']['lung']['rand_scale_intensity_prob']),
            RandShiftIntensityd(
                keys=["image"],
                offsets=config['transforms']['lung']['rand_shift_intensity_offset'],
                prob=config['transforms']['lung']['rand_shift_intensity_prob'],
            ),
            # MapLabelValued(keys="label", orig_labels=[1], target_labels=[2]),  # 종양을 2로 설정
            EnsureTyped(keys=["image", "label"]),  # ✅ MetaTensor 변환 추가
        ]
    )

    lung_val_transforms = Compose(
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
            EnsureTyped(keys=["image", "label"]),  # ✅ MetaTensor 변환 추가
        ]
    )

    liver_test_transforms = Compose(
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
                EnsureTyped(keys=["image", "label"]),  # ✅ MetaTensor 변환 추가
            ]
        )

    lung_test_transforms= Compose(
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
                    keys=["image"], a_min=config['transforms']['lung']['a_min'], a_max=config['transforms']['lung']['a_max'], b_min=0.0, b_max=1.0, clip=True
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # MapLabelValued(keys="label", orig_labels=[1], target_labels=[2]),  # 종양을 2로 설정
                EnsureTyped(keys=["image", "label"]),  # ✅ MetaTensor 변환 추가
            ]
        )

    # ### Dataset
    # set_track_meta(False)
    # **📌 모든 랜덤 변환에 시드 적용** (실험값 fluctation 고정용 코드)
    set_all_random_states(lung_train_transforms.transforms, seed=1234)
    set_all_random_states(lung_val_transforms.transforms, seed=1234)
    set_all_random_states(lung_test_transforms.transforms, seed=1234)
    set_all_random_states(liver_train_transforms.transforms, seed=1234)
    set_all_random_states(liver_val_transforms.transforms, seed=1234)
    set_all_random_states(liver_test_transforms.transforms, seed=1234)

    #### For T-SNE
    lung_datasets = "/data/hyungseok/Swin-UNETR/results/debug/lung_dataset_tsne.json"
    liver_datasets = "/data/hyungseok/Swin-UNETR/results/debug/liver_dataset_tsne.json"

    # CacheDataset 및 DataLoader 구성
    liver_train_files = load_decathlon_datalist(liver_datasets, True, "training")
    liver_test_files = load_decathlon_datalist(liver_datasets, True, "validation")
    lung_train_files = load_decathlon_datalist(lung_datasets, True, "training")
    lung_test_files = load_decathlon_datalist(lung_datasets, True, "validation")
    

    liver_train_ds = Dataset(data=liver_train_files, transform=liver_val_transforms)
    lung_train_ds = Dataset(data=lung_train_files, transform=lung_val_transforms)
    ### combined_train_dataset = ConcatDataset([liver_train_ds, lung_train_ds])

    liver_test_ds = Dataset(data=liver_test_files, transform=liver_val_transforms)
    lung_test_ds = Dataset(data=lung_test_files, transform=lung_val_transforms)
    #### combined_val_dataset = ConcatDataset([liver_val_ds, lung_val_ds])
    
    # liver_train_ds = CacheDataset(data=liver_train_files, transform=liver_train_transforms,cache_rate=1.0)
    # lung_train_ds = CacheDataset(data=lung_train_files, transform=lung_train_transforms,cache_rate=1.0)
    # # combined_train_dataset = ConcatDataset([liver_train_ds, lung_train_ds])

    # liver_val_ds = CacheDataset(data=liver_val_files, transform=liver_val_transforms,cache_rate=1.0)
    # lung_val_ds = CacheDataset(data=lung_val_files, transform=lung_val_transforms,cache_rate=1.0)
    # # # # combined_val_dataset = ConcatDataset([liver_val_ds, lung_val_ds])
    
    liver_train_loader = DataLoader(liver_train_ds, batch_size=1, num_workers=8,worker_init_fn=worker_init_fn)
    lung_train_loader = DataLoader(lung_train_ds, batch_size=1, num_workers=8, worker_init_fn=worker_init_fn)
    
    liver_test_loader = DataLoader(liver_test_ds, batch_size=1, 
        num_workers=4, worker_init_fn=worker_init_fn)
    lung_test_loader = DataLoader(lung_test_ds, batch_size=1, 
        num_workers=4, worker_init_fn=worker_init_fn)
    # val_loader = DataLoader(combined_val_dataset, batch_size=1, 
    #                         num_workers=config['train_params']['num_workers'], pin_memory=config['train_params']['pin_memory'])

    return liver_train_loader, lung_train_loader, liver_test_loader, lung_test_loader

def set_model(config):
    if config['model_params']['type'] == "SwinUnetr":
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            img_size=config['model_params']['img_size'],
            in_channels=config['model_params']['in_channels'],
            out_channels=config['model_params']['out_channels'],
            feature_size=config['model_params']['feature_size'],
            use_checkpoint=config['model_params']['use_checkpoint']
        ).to(device)
     
    elif config['model_params']['type'] == "ContrastiveSwinUNETR_ml":
        from monai.networks.nets import SwinUNETR
        class ContrastiveSwinUNETR(SwinUNETR):
            def __init__(self, img_size, in_channels, out_channels, feature_size, use_checkpoint,in_dim=48, hidden_dim=128, out_dim=128):
                # SwinUNETR의 초기화 함수 호출
                super(ContrastiveSwinUNETR, self).__init__(
                    img_size=img_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    feature_size=feature_size,
                    use_checkpoint=use_checkpoint
                )
                
                # Projection head를 Conv3d로 정의
                self.projection_head0 = nn.Conv3d(in_dim, hidden_dim, kernel_size=1)
                self.projection_head1 = nn.Conv3d(in_dim*2, hidden_dim, kernel_size=1)
                self.projection_head2 = nn.Conv3d(in_dim*4, hidden_dim, kernel_size=1)
                self.projection_head3 = nn.Conv3d(in_dim*8, hidden_dim, kernel_size=1)
                self.projection_head4 = nn.Conv3d(in_dim*16, hidden_dim, kernel_size=1)
                
                
                    

            def forward(self, x_in: torch.Tensor):
                # SwinUNETR에서 정의된 forward 기능 사용
                hidden_states_out = self.swinViT(x_in, self.normalize)
                
                proj = []
                proj.append(self.projection_head0(hidden_states_out[0]))
                proj.append(self.projection_head1(hidden_states_out[1]))
                proj.append(self.projection_head2(hidden_states_out[2]))
                proj.append(self.projection_head3(hidden_states_out[3]))
                proj.append(self.projection_head4(hidden_states_out[4]))
                
                for i in range(5):
                    proj[i] = nn.functional.normalize(proj[i], p=2, dim=1)
                
                return proj # List[torch.Tensor]
                
                
    
        model = ContrastiveSwinUNETR(
        img_size=config['model_params']['img_size'],
        in_channels=config['model_params']['in_channels'],
        out_channels=config['model_params']['out_channels'],
        feature_size=config['model_params']['feature_size'],
        use_checkpoint=config['model_params']['use_checkpoint']
        )
        class PixelContrastLoss(nn.Module):
            def __init__(self, config):
                super(PixelContrastLoss, self).__init__()
                self.temperature = config['CONTRASTIVE']['TEMPERATURE']
                self.base_temperature = config['CONTRASTIVE']['BASE_TEMPERATURE']
                # self.max_views = config.CONTRASTIVE.MAX_VIEWS
                self.max_views = config['CONTRASTIVE']['MAX_VIEWS']
                self.queue_size = config['CONTRASTIVE']['QUEUE_SIZE']
                self.dim = config['CONTRASTIVE']['DIM']
                self.num_classes = config['CONTRASTIVE']['NUM_CLASSES']
                self.ignore_label = 255
                self.mode = config['CONTRASTIVE']['MODE']  # 1: 기존 방식, 2: memory bank 방식, 3: hard sampling 방식
                self.pixel_update_freq = config['CONTRASTIVE']['PIXEL_UPDATE_FREQ']
                self.max_samples = 256
                if self.mode > 1:
                    # memory bank (queue) 추가
                    self.register_buffer("pixel_queue", torch.randn(self.num_classes, self.queue_size, self.dim))
                    self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
                    self.register_buffer("pixel_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))

                    # self.register_buffer("segment_queue", torch.randn(self.num_classes, self.queue_size, self.dim))
                    # self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
                    # self.register_buffer("segment_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))

            def unravel_index(self,indices, shape):
                """
                1D 인덱스를 다차원 인덱스로 변환.
                Args:
                    indices: 1D Tensor
                    shape: 목표 텐서의 shape (depth, height, width)
                Returns:
                    unravelled: 각 차원에 대한 좌표를 담은 Tensor tuple
                """
                coords = []
                for dim in reversed(shape):
                    coords.append(indices % dim)
                    indices = indices // dim
                return tuple(reversed(coords))
            
            @torch.no_grad()
            def extract_boundary_labels(self,labels):
                """
                1. conv3d를 사용하여 경계값 마스크 생성
                Args:
                    labels: Tensor (B, D, H, W) 형태의 레이블
                Returns:
                    boundary_mask: Tensor (B, D, H, W) 형태의 경계값 마스크
                """
                batch_size, num_elements = labels.shape
                depth = height = width = int(round(num_elements ** (1/3)))
                labels = labels.view(batch_size, depth, height, width)

                device = labels.device
                kernel = torch.tensor([[[[1, -1]]]], device=device).float()  # (1, 1, 1, 2)
                kernel = kernel.unsqueeze(2)  # (1, 1, 2, 2, 2)

                diff_x = F.conv3d(labels.float().unsqueeze(1), kernel, padding='same').abs()
                diff_y = F.conv3d(labels.float().unsqueeze(1), kernel.permute(0, 1, 3, 2, 4), padding='same').abs()
                diff_z = F.conv3d(labels.float().unsqueeze(1), kernel.permute(0, 1, 4, 2, 3), padding='same').abs()

                boundary_mask = ((diff_x + diff_y + diff_z) > 0).float()
                return boundary_mask.squeeze(1)  # (B, D, H, W)
            
            def extract_features_near_boundary(self,labels, boundary_mask):
                """
                2. 경계 근처에서 클래스별로 feature를 추출
                Args:
                    labels: Tensor (B, D*H*W) 형태의 레이블
                    boundary_mask: Tensor (B, D, H, W) 형태의 경계값 마스크
                    n_samples: int, 추출할 샘플 개수 N
                Returns:
                    extracted_indicies: Tensor, 추출된 feature (N, 1)
                """
                batch_size, num_elements = labels.shape
                depth = height = width = int(round(num_elements ** (1/3)))
                labels = labels.view(batch_size, depth, height, width)
                device = labels.device
                batch_size, depth, height, width = labels.shape

                extracted_indicies = []
                for b in range(batch_size):
                    # 경계 근처 좌표 가져오기
                    boundary_indices = (boundary_mask[b] > 0).nonzero(as_tuple=False)  # (K, 3)

                    # 경계 근처 확장 좌표 계산 (3x3x3 박스)
                    grid_offsets = torch.tensor([[-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
                                                [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
                                                [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
                                                [0, -1, -1], [0, -1, 0], [0, -1, 1],
                                                [0, 0, -1], [0, 0, 0], [0, 0, 1],
                                                [0, 1, -1], [0, 1, 0], [0, 1, 1],
                                                [1, -1, -1], [1, -1, 0], [1, -1, 1],
                                                [1, 0, -1], [1, 0, 0], [1, 0, 1],
                                                [1, 1, -1], [1, 1, 0], [1, 1, 1]], device=device)

                    expanded_boundary_indices = (boundary_indices.unsqueeze(1) + grid_offsets).reshape(-1, 3)
                    valid_mask = (expanded_boundary_indices[:, 0] >= 0) & (expanded_boundary_indices[:, 0] < depth) & \
                                (expanded_boundary_indices[:, 1] >= 0) & (expanded_boundary_indices[:, 1] < height) & \
                                (expanded_boundary_indices[:, 2] >= 0) & (expanded_boundary_indices[:, 2] < width)
                    expanded_boundary_indices = expanded_boundary_indices[valid_mask].unique(dim=0)

                    # 경계 근처 feature 추출
                    # expanded_features = features[b, :, expanded_boundary_indices[:, 0], expanded_boundary_indices[:, 1], expanded_boundary_indices[:, 2]].T
                    # 경계 근처 feature 추출 (플랫된 인덱스를 직접 반환)
                    extracted_indicies.append(expanded_boundary_indices[:, 0] * height * width + expanded_boundary_indices[:, 1] * width + expanded_boundary_indices[:, 2])

                # return torch.stack(extracted_features, dim=0)  # (B, N, C)
                return extracted_indicies #(B,Ndiff) list

            @torch.no_grad()
            def update_queue(self, embeddings, labels):
                if self.mode <= 1:
                    return  # self.mode가 1 이하인 경우 queue를 업데이트하지 않음
                # embeddings: [Batch, C, D, H, W]
                # labels: [Batch, D, H, W]
                
                batch_size= embeddings.shape[0]
                boundary_masks = self.extract_boundary_labels(labels) # [B,D,H,W]
                boundary_indices = self.extract_features_near_boundary(labels, boundary_masks)
                for bs in range (batch_size):
                    this_feat = embeddings[bs].contiguous().view(self.dim, -1)
                    this_label = labels[bs].contiguous().view(-1)
                    this_label_ids = torch.unique(this_label)
                    this_boundary_indices = boundary_indices[bs].view(-1)
                    
                    for lb in this_label_ids:
                        # 해당 클래스에 속하는 voxel 선택
                        idxs = (this_label == lb).nonzero()
                        if self.mode >=2:
                            # Get indices of boundary and non-boundary pixels within the class
                            is_boundary = torch.isin(idxs, this_boundary_indices)

                            # 경계/비경계 픽셀 분리
                            boundary_indices_lb = idxs[is_boundary]
                            non_boundary_indices_lb = idxs[~is_boundary]

                            # Sample pixels from boundary and non-boundary regions
                            num_boundary_pixels = len(boundary_indices_lb)
                            num_non_boundary_pixels = len(non_boundary_indices_lb)
                            if min(self.pixel_update_freq//2, num_boundary_pixels, num_non_boundary_pixels) == num_boundary_pixels:
                                num_samples_per_boundary = min(self.pixel_update_freq//2, num_boundary_pixels, num_non_boundary_pixels)
                                num_samples_per_nonboundary = self.pixel_update_freq - num_samples_per_boundary
                            elif min(self.pixel_update_freq//2, num_boundary_pixels, num_non_boundary_pixels) == num_non_boundary_pixels:
                                num_samples_per_nonboundary = min(self.pixel_update_freq//2, num_boundary_pixels, num_non_boundary_pixels)
                                num_samples_per_boundary = self.pixel_update_freq - num_samples_per_nonboundary
                            else:
                                num_samples_per_nonboundary = self.pixel_update_freq//2
                                num_samples_per_boundary = self.pixel_update_freq//2
                                
                            
                            # K= num_samples_per_boundary+num_samples_per_nonboundary
                            sampled_boundary_indices = boundary_indices_lb[torch.randperm(num_boundary_pixels)[:num_samples_per_boundary]]
                            sampled_non_boundary_indices = non_boundary_indices_lb[torch.randperm(num_non_boundary_pixels)[:num_samples_per_nonboundary]]
                            perm = torch.cat([sampled_boundary_indices, sampled_non_boundary_indices])
                            K = len(perm)
                        # segment enqueue and dequeue
                        # feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                        # ptr = int(self.segment_queue_ptr[lb])
                        # self.segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                        # self.segment_queue_ptr[lb] = (self.segment_queue_ptr[lb] + 1) % self.queue_size
                        
                        else: 
                            num_pixel = idxs.shape[0]
                            perm = torch.randperm(num_pixel)
                            K = min(num_pixel, self.pixel_update_freq)
                            perm = perm[:K]

                        feat = this_feat[:, perm]
                        feat = torch.transpose(feat, 0, 1)
                        ptr = int(self.pixel_queue_ptr[lb])

                        if ptr + K >= self.queue_size:
                            self.pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                            self.pixel_queue_ptr[lb] = 0
                        else:
                            self.pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                            self.pixel_queue_ptr[lb] = (self.pixel_queue_ptr[lb] + 1) % self.queue_size
            
            def _hard_anchor_sampling(self, X, y, y_hat, use_hard_anchor=False):
                batch_size, feat_dim = X.shape[0], X.shape[-1]
                selected_features = []
                selected_labels = []

                # 클래스 목록 가져오기
                classes = torch.unique(y)
                for cls in classes:
                    # 1. GT 기반으로 Anchor 후보 샘플링
                    class_indices = (y == cls).nonzero()
                    num_samples = min(len(class_indices), self.max_views)

                    if num_samples == 0:
                        continue

                    if not use_hard_anchor:
                        # 기존 무작위 Anchor 방식
                        perm = torch.randperm(len(class_indices))[:num_samples]
                        selected_indices = class_indices[perm]
                    else:
                        # Hard Anchor 방식
                        hat_class_indices = (y_hat == cls).nonzero()

                        # TP와 FP/FN 구분
                        easy_mask = torch.zeros(len(class_indices), dtype=torch.bool, device=class_indices.device)
                        for idx in hat_class_indices:
                            # class_indices 중 예측과 겹치는 경우만 easy
                            easy_mask |= (class_indices == idx).all(dim=1)

                        # easy_indices: class_indices 중 예측과 겹치는 부분
                        easy_indices = class_indices[easy_mask]

                        # hard_indices: class_indices 중 예측과 겹치지 않는 부분
                        hard_indices = class_indices[~easy_mask]

                        num_hard = hard_indices.shape[0]
                        num_easy = easy_indices.shape[0]

                        if num_hard == 0 or num_easy == 0 or (num_easy+num_hard)<self.max_views:
                            # 기존 무작위 Anchor 방식
                            perm = torch.randperm(len(class_indices))[:num_samples]
                            selected_indices = class_indices[perm]
                        else:
                            # Hard/Easy 균등하게 선택
                            if num_hard >= self.max_views // 2 and num_easy >= self.max_views // 2:
                                num_hard_keep = self.max_views // 2
                                num_easy_keep = self.max_views - num_hard_keep
                            elif num_hard >= self.max_views // 2:
                                num_easy_keep = num_easy
                                num_hard_keep = self.max_views - num_easy_keep
                            elif num_easy >= self.max_views // 2:
                                num_hard_keep = num_hard
                                num_easy_keep = self.max_views - num_hard_keep
                            else:
                                raise Exception(f"Unexpected sampling issue: num_hard={num_hard}, num_easy={num_easy}")

                            perm_hard = torch.randperm(num_hard)[:num_hard_keep]
                            perm_easy = torch.randperm(num_easy)[:num_easy_keep]

                            hard_indices = hard_indices[perm_hard]
                            easy_indices = easy_indices[perm_easy]
                            selected_indices = torch.cat((hard_indices, easy_indices), dim=0)

                    # 샘플 부족 시 복제
                    if selected_indices.shape[0] < self.max_views:
                        selected_indices = selected_indices.repeat((self.max_views // selected_indices.shape[0]) + 1, 1)[:self.max_views]
                    # 선택된 Anchor 추가
                    selected_features.append(X[selected_indices[:,0],selected_indices[:,1],:])
                    selected_labels.append(y[selected_indices[:,0],selected_indices[:,1]])

                # # 최종 Anchor 병합
                # if len(selected_features) > 0:
                #     selected_features = torch.cat(selected_features, dim=0)
                #     selected_labels = torch.cat(selected_labels, dim=0)
                # else:
                #     selected_features = None
                #     selected_labels = None
                selected_features = torch.stack(selected_features, dim=0)  # (num_classes, max_views, feat_dim)
                selected_labels = torch.stack(selected_labels,dim=0).to(device=X.device)
                return selected_features, selected_labels
            

            def _contrastive(self, X_anchor, y_anchor, queue=None):
                anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]
                
                y_anchor = y_anchor.contiguous().view(-1, 1).to(X_anchor.device)
                anchor_count = n_view
                anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

                if queue is not None:
                    queue_feature = queue.view(-1, queue.shape[-1]).to(anchor_feature.device)  # [Class, Memsize, Dim] -> [Class*Memsize, Dim]
                    queue_labels = torch.arange(queue.shape[0]).repeat_interleave(queue.shape[1]).unsqueeze(dim=1).to(anchor_feature.device)  # 각 클래스 라벨 생성
                    X_contrast = torch.cat([anchor_feature, queue_feature], dim=0)
                    y_contrast = torch.cat([y_anchor, queue_labels], dim=0)
                    contrast_count = 1
                    contrast_feature = X_contrast
                else:
                    y_contrast = y_anchor
                    contrast_count = n_view
                    contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

                mask = torch.eq(y_anchor, y_contrast.T).float().cuda()

                anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                                self.temperature)
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast - logits_max.detach()

                # mask = mask.repeat(anchor_count, contrast_count)
                neg_mask = 1 - mask

                logits_mask = torch.ones_like(mask)
                diag_indices = torch.arange(anchor_feature.shape[0]).to(mask.device)  # Batch diagonal indices
                logits_mask[diag_indices, diag_indices] = 0  # Remove diagonal elements

                mask = mask * logits_mask

                neg_logits = torch.exp(logits) * neg_mask
                neg_logits = neg_logits.sum(1, keepdim=True)

                exp_logits = torch.exp(logits)

                log_prob = logits - torch.log(exp_logits + neg_logits)

                mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+ 1e-10)

                loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                loss = loss.mean()

                return loss

            def forward(self, feats, labels=None, predict=None):
                """
                Contrastive Loss 계산 루틴.

                Args:
                    feats: (B, C, D, H, W) 입력 feature 맵.
                    labels: (B, D, H, W) 실제 레이블.
                    predict: (B, D, H, W) 모델 예측 결과.

                Returns:
                    loss: Contrastive Loss 값.
                """
                # Labels 크기 조정
                # labels = labels.unsqueeze(1).float().clone()
                labels = F.interpolate(
                    labels,
                    (feats.shape[-3], feats.shape[-2], feats.shape[-1]),
                    mode='nearest'
                )
                labels = labels.squeeze(1).long()
                assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

                # Feature와 Labels 펼치기
                batch_size = feats.shape[0]
                labels = labels.contiguous().view(batch_size, -1)  # (B, D*H*W)
                predict = predict.contiguous().view(batch_size, -1)  # (B, D*H*W)
                feats = feats.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C) -> (B, D*H*W, C)
                feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

                # Hard Anchor Sampling
                use_hard_anchor = self.mode == 3
                feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict,use_hard_anchor)

                if feats_ is None or labels_ is None:
                    return torch.tensor(0.0).cuda()  # 샘플이 없을 경우 0 반환
                
                # Pixel Queue 업데이트
                queue = self.pixel_queue if self.mode in [2, 3] else None
                total_loss = self._contrastive(feats_, labels_, queue)
                with torch.no_grad():
                    if queue is not None:
                        self.update_queue(feats.detach(), labels.detach())

                return total_loss
            
            def multi_forward(self, feat_list, labels=None, predict=None):
                """
                Args:
                    feats: List[(B, C, D, H, W), ...] multi scale feature map list.
                    labels: (B, D, H, W) 실제 레이블.
                    predict: (B, D, H, W) 모델 예측 결과.

                Returns:
                    loss: Contrastive Loss 값.
                """
                total_loss = torch.tensor(0.0).cuda()
                batch_size = feat_list[0].shape[0]
                for feat in feat_list:
                    B, C, D, H, W = feat.shape
                    interp_labels = F.interpolate(
                        labels,
                        (D, H, W), mode='nearest'
                    )
                    interp_labels = interp_labels.squeeze(1).long()
                    interp_labels = interp_labels.contiguous().view(batch_size, -1)
                    
                    feat = rearrange(feat, 'b c d h w -> b (d h w) c').contiguous()
                    
                    # Hard Anchor Sampling: TODO
                    use_hard_anchor = self.mode == 3
                    feat_, label_ = self._hard_anchor_sampling(feat, interp_labels, predict, use_hard_anchor)

                    
                    # Pixel Queue 업데이트: TODO
                    queue = self.pixel_queue if self.mode in [2, 3] else None
                    
                    total_loss += self._contrastive(feat_, label_, queue)
                
                total_loss /= len(feat_list)
                
                return total_loss
     
    if config['model_params']['type'] == "ContrastiveSwinUNETR":            
        from monai.networks.nets import SwinUNETR
        class ContrastiveSwinUNETR(SwinUNETR):
            def __init__(self, img_size, in_channels, out_channels, feature_size,use_checkpoint, in_dim=48, hidden_dim=128, out_dim=128):
                # SwinUNETR의 초기화 함수 호출
                super(ContrastiveSwinUNETR, self).__init__(
                    img_size=img_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    feature_size=feature_size,
                    use_checkpoint = use_checkpoint
                )
            def forward(self, x_in):
                # SwinUNETR에서 정의된 forward 기능 사용
                hidden_states_out = self.swinViT(x_in, self.normalize)
                enc0 = self.encoder1(x_in)
                enc1 = self.encoder2(hidden_states_out[0])
                enc2 = self.encoder3(hidden_states_out[1])
                enc3 = self.encoder4(hidden_states_out[2])
                dec4 = self.encoder10(hidden_states_out[4])
                dec3 = self.decoder5(dec4, hidden_states_out[3])
                dec2 = self.decoder4(dec3, enc3)
                dec1 = self.decoder3(dec2, enc2)
                dec0 = self.decoder2(dec1, enc1)
                out = self.decoder1(dec0, enc0)

                # Contrastive learning을 위한 중간 feature
                features_before_output = out

                # Projection head에 통과시키기
                # pixel_embeddings = self.projection_head(features_before_output)

                # L2 정규화를 적용하여 embeddings을 normalize
                # pixel_embeddings = nn.functional.normalize(pixel_embeddings, p=2, dim=1)

                # logits = self.out(out)
                
                # logits과 pixel_embeddings 반환
                return nn.functional.normalize(features_before_output, p=2, dim=1)


        model = ContrastiveSwinUNETR(
        img_size=config['model_params']['img_size'],
        in_channels=config['model_params']['in_channels'],
        out_channels=config['model_params']['out_channels'],
        feature_size=config['model_params']['feature_size'],
        use_checkpoint=config['model_params']['use_checkpoint']
        ).to(device)
    
    if config['model_params']['type'] == "AttSwinUNETR_0313":
        from models.swin_unetr_att import SwinUNETR, MultiScaleCrossAttention, PrototypeAttention
        class AttSwinUNETR(SwinUNETR):
            def __init__(self, img_size, in_channels, out_channels, feature_size, use_checkpoint,
                        # feature_dims=[48, 96, 192], prototype_dim=128):
                        feature_dims=[48], prototype_dim=48):
                super(AttSwinUNETR, self).__init__(
                    img_size=img_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    feature_size=feature_size,
                    use_checkpoint=use_checkpoint
                )
                
                # ✅ 폐 종양과 간 종양을 위한 Prototype 추가
                self.prototypes_lung = nn.ParameterList([
                    nn.Parameter(torch.randn(1, dim)) for dim in feature_dims
                ])
                self.prototypes_liver = nn.ParameterList([
                    nn.Parameter(torch.randn(1, dim)) for dim in feature_dims
                ])

                # Prototype 업데이트를 위한 self-attention 모듈 (별도, 파라미터 공유 방지)
                self.liver_prototype_attention = PrototypeAttention(embed_dim=prototype_dim, num_heads=4)
                # Prototype 업데이트를 위한 self-attention 모듈 (별도, 파라미터 공유 방지)
                self.lung_prototype_attention = PrototypeAttention(embed_dim=prototype_dim, num_heads=4)
        
                # Multi-Scale Cross Attention 추가
                self.multi_scale_cross_attention = MultiScaleCrossAttention(
                    feature_dims=feature_dims, num_heads=4
                )
            def update_prototype(self, centroid, prototype, organ):
                """
                centroid: (B, C) - 각 배치의 feature 평균 (예: 종양 영역의 feature)
                prototype: (1, C)
                organ: string, 'lung' 또는 'liver'에 따라 업데이트 모듈 선택
                """
                # 배치 차원 평균을 통해 (1, C)로 변환 (이미 1일 수 있음)
                avg_centroid = centroid.mean(dim=0, keepdim=True)  # (1, C)
                # self-attention 모듈 입력 형식에 맞게 차원 확장: (B, seq_len, C)
                avg_centroid = avg_centroid.unsqueeze(0)  # (1, 1, C)
                prototype_exp = prototype.unsqueeze(0)     # (1, 1, C)
                # Organ에 따라 해당 prototype attention 모듈 선택
                if organ == "lung":
                    updated_proto = self.lung_prototype_attention(avg_centroid, prototype_exp)
                elif organ == "liver":
                    updated_proto = self.liver_prototype_attention(avg_centroid, prototype_exp)
                else:
                    # 기본은 liver
                    updated_proto = self.liver_prototype_attention(avg_centroid, prototype_exp)
                updated_proto = updated_proto.squeeze(0)  # (1, C)
                return updated_proto
            

            def forward(self, x_in, label=None, organ=None):
                hidden_states_out = self.swinViT(x_in, self.normalize)
                enc0 = self.encoder1(x_in)
                enc1 = self.encoder2(hidden_states_out[0])  # (B, 48, D, H, W)
                enc2 = self.encoder3(hidden_states_out[1])  # (B, 96, D, H, W)
                enc3 = self.encoder4(hidden_states_out[2])  # (B, 192, D, H, W)
                dec4 = self.encoder10(hidden_states_out[4])
                dec3 = self.decoder5(dec4, hidden_states_out[3])
                dec2 = self.decoder4(dec3, enc3)
                dec1 = self.decoder3(dec2, enc2)
                dec0 = self.decoder2(dec1, enc1)
                out = self.decoder1(dec0, enc0)

                if self.training and label is not None:
                    # if organ == "mixed":
                    # 배치의 절반은 liver, 나머지 절반은 lung로 가정 (B는 짝수)
                    B = out.shape[0]
                    half = B // 2
                    # liver part: 인덱스 0 ~ half-1
                    out_liver = out[:half]      # (B/2, C, D, H, W)
                    label_liver = label[:half]
                    # lung part: 인덱스 half ~ B-1
                    out_lung = out[half:]
                    label_lung = label[half:]
                    
                    def calc_valid_centroid(out_part, label_part):
                        """
                        out_part: (B, C, D, H, W)
                        label_part: (B, 1, D, H, W)
                        
                        유효한(종양 픽셀이 하나라도 존재하는) 샘플에 대해
                        sum_features와 유효 인덱스를 반환합니다.
                        """
                         
                        mask = (label_part == 1).float()  # (B, 1, D, H, W)
                        spatial_dims = list(range(2, out_part.dim()))
                        count = mask.sum(dim=spatial_dims)  # (B, 1)
                        # 유효한 샘플의 인덱스를 구합니다.
                        valid_idx = (count.squeeze(1) > 0).nonzero().squeeze(1)
                        if valid_idx.numel() == 0:
                            return None, valid_idx
                        # 유효한 샘플만의 sum_features 계산
                        sum_features = (out_part * mask).sum(dim=spatial_dims)  # (B, C)
                        valid_sum_features = sum_features[valid_idx.tolist()]  # (B_valid, C)
                        centroid = valid_sum_features.mean(dim=0, keepdim=True) / ( (mask.sum(dim=spatial_dims)[valid_idx.tolist()]).mean() + 1e-6 )
                        return centroid, valid_idx.tolist()

                    
                    centroid_liver, valid_idx_liver = calc_valid_centroid(out_liver, label_liver)
                    centroid_lung, valid_idx_lung = calc_valid_centroid(out_lung, label_lung)

                    if centroid_liver != None :
                        # 유효한 샘플들의 평균을 계산
                        # valid한 샘플만 선택
                        valid_out_liver = torch.index_select(out_liver, 0, torch.tensor(valid_idx_liver, device=out.device))
                        # valid 샘플들에 대해서만 attention 적용
                        updated_valid_out_liver = self.multi_scale_cross_attention([valid_out_liver], self.prototypes_liver)[0]
                        # 원래 out_liver의 valid 인덱스 위치에 업데이트된 값을 반영
                        out_liver = out_liver.clone()
                        out_liver[valid_idx_liver] = updated_valid_out_liver
                        for i, prototype in enumerate(self.prototypes_liver):
                            updated_proto = self.update_prototype(centroid_liver, prototype, "liver")
                            # self.prototypes_liver[i].data = updated_proto.data
                            momentum = 0.9  # 모멘텀 계수 (예시)
                            with torch.no_grad():
                                self.prototypes_liver[i].mul_(momentum).add_(updated_proto * (1 - momentum))

                    else:
                        # 종양이 전혀 없으면 업데이트 생략
                        pass
                    
                    # lung 계산
                    
                    if centroid_lung != None :
                        # valid한 샘플만 선택
                        valid_out_lung = torch.index_select(out_lung, 0, torch.tensor(valid_idx_lung, device=out.device))
                        # valid 샘플들에 대해서만 attention 적용
                        updated_valid_out_lung = self.multi_scale_cross_attention([valid_out_lung], self.prototypes_lung)[0]
                        # 원래 out_lung의 valid 인덱스 위치에 업데이트된 값을 반영
                        out_lung = out_lung.clone()
                        out_lung[valid_idx_lung] = updated_valid_out_lung
                        for i, prototype in enumerate(self.prototypes_lung):
                            updated_proto = self.update_prototype(centroid_lung, prototype, "lung")
                            momentum = 0.9  # 모멘텀 계수 (예시)
                            with torch.no_grad():
                                self.prototypes_lung[i].mul_(momentum).add_(updated_proto * (1 - momentum))

                            # self.prototypes_lung[i].data = updated_proto.data
                    else:
                        pass
                    
                    # 배치 순서대로 다시 합치기
                    out = torch.cat([out_liver, out_lung], dim=0)
            
                else:
                    # organ 인자에 따라 해당 organ의 prototype 선택 (예: "lung" 또는 "liver")
                    if organ == "lung":
                        prototypes = self.prototypes_lung

                    elif organ == "liver":
                        prototypes = self.prototypes_liver
                    else:
                        # organ 정보가 없으면 기본적으로 liver prototype 사용 (또는 두 organ 모두 업데이트)
                        prototypes = self.prototypes_liver

                    # # 예시로, 적용할 feature map 리스트 (예: encoder의 여러 스케일 feature)
                    # Apply_feature = [out]
                    # # Multi-Scale Cross Attention 적용
                    # updated_features = self.multi_scale_cross_attention(Apply_feature, prototypes)

                    # out = self.multi_scale_cross_attention([out], prototypes)
                    # out= out[0]
                # logits = self.out(out.float())
                # return logits
                return out.float()

    
        model = AttSwinUNETR(
        img_size=config['model_params']['img_size'],
        in_channels=config['model_params']['in_channels'],
        out_channels=config['model_params']['out_channels'],
        feature_size=config['model_params']['feature_size'],
        use_checkpoint=config['model_params']['use_checkpoint']
        )
  
    if config['model_params']['type'] == "AttSwinUNETR_DETR":
        import sys
        sys.path.append('/data/hyungseok/Swin-UNETR/')
        from models.swin_unetr_detr import SwinUNETR, TumorPrototypeIntegration

        class AttSwinUNETR(SwinUNETR): 
            def __init__(self, img_size, in_channels, out_channels, feature_size, use_checkpoint,
                        # feature_dims=[48, 96, 192], prototype_dim=128):
                        feature_dims=[48], prototype_dim=48):
                super(AttSwinUNETR, self).__init__(
                    img_size=img_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    feature_size=feature_size,
                    use_checkpoint=use_checkpoint
                )
                
                # ✅ Tumor Prototype Integration 추가
                self.liver_prototype_integration = TumorPrototypeIntegration(feature_dims=feature_dims, prototype_dim=prototype_dim, num_heads=4, dropout=0.1)
                # ✅ Tumor Prototype Integration 추가
                self.lung_prototype_integration = TumorPrototypeIntegration(feature_dims=feature_dims, prototype_dim=prototype_dim, num_heads=4, dropout=0.1)

            def forward(self, x_in, label=None, organ=None):
                hidden_states_out = self.swinViT(x_in, self.normalize)
                enc0 = self.encoder1(x_in)
                enc1 = self.encoder2(hidden_states_out[0])  # (B, 48, D, H, W)
                enc2 = self.encoder3(hidden_states_out[1])  # (B, 96, D, H, W)
                enc3 = self.encoder4(hidden_states_out[2])  # (B, 192, D, H, W)
                dec4 = self.encoder10(hidden_states_out[4])
                dec3 = self.decoder5(dec4, hidden_states_out[3])
                dec2 = self.decoder4(dec3, enc3)
                dec1 = self.decoder3(dec2, enc2)
                dec0 = self.decoder2(dec1, enc1)
                out = self.decoder1(dec0, enc0)

                if self.training and label is not None:
                    # if organ == "mixed":
                    # 배치의 절반은 liver, 나머지 절반은 lung로 가정 (B는 짝수)
                    B = out.shape[0]
                    half = B // 2
                    # liver part: 인덱스 0 ~ half-1
                    out_liver = out[:half]      # (B/2, C, D, H, W)
                    label_liver = label[:half]
                    # lung part: 인덱스 half ~ B-1
                    out_lung = out[half:]
                    label_lung = label[half:]
                                        
                    def calc_valid_centroid(out_part, label_part):
                        """
                        Args:
                            out_part: (B, C, D, H, W) - 입력 feature map
                            label_part: (B, 1, D, H, W) - Binary label map (종양 = 1, 배경 = 0)
                        
                        Returns:
                            centroid: (B, C) - 종양 feature의 정확한 masked average
                            valid_idx: (B_valid,) - 유효한 샘플 인덱스 (종양 픽셀이 존재하는 경우)
                        """
                        mask = (label_part == 1).float()  # (B, 1, D, H, W)
                        spatial_dims = list(range(2, out_part.dim()))  # D, H, W 차원
                        
                        # 각 샘플별 종양 voxel 수 계산
                        count = mask.sum(dim=spatial_dims)  # (B, 1)
                        
                        # 유효한 샘플 인덱스: 종양 픽셀이 1개 이상 있는 경우
                        valid_idx = (count.squeeze(1) > 0).nonzero().squeeze(1)
                        if valid_idx.numel() == 0:
                            return torch.zeros(out_part.shape[0], out_part.shape[1], device=out_part.device)  # (B, C)
                        
                        # 종양 영역에 대해 마스크된 값의 합을 voxel 수로 나눔 (0 division 방지를 위해 작은 값을 더함)
                        centroid = (out_part * mask).sum(dim=tuple(spatial_dims)) / (count + 1e-6)  # (B, C)
                        # return centroid, valid_idx
                        return centroid

                    centroid_liver = calc_valid_centroid(out_liver, label_liver)
                    centroid_lung = calc_valid_centroid(out_lung, label_lung)

                    out_liver = self.liver_prototype_integration([out_liver],centroid_liver)[0]
                    out_lung = self.lung_prototype_integration([out_lung],centroid_lung)[0]
                    # 배치 순서대로 다시 합치기
                    out = torch.cat([out_liver, out_lung], dim=0)
            
                else:
                    # organ 인자에 따라 해당 organ의 prototype 선택 (예: "lung" 또는 "liver")
                    if organ == "lung":
                        # Inference 시 학습된 query를 그대로 사용
                        out = self.lung_prototype_integration.forward_inference([out])[0]

                    elif organ == "liver":
                        out = self.liver_prototype_integration.forward_inference([out])[0]
                    else:
                        # organ 정보가 없으면 기본적으로 liver prototype 사용 (또는 두 organ 모두 업데이트)
                        out = self.liver_prototype_integration.forward_inference([out])[0]

                return nn.functional.normalize(out.float(), p=2, dim=1)
                # logits = self.out(out.float())
                # return logits

        
        model = AttSwinUNETR(
        img_size=config['model_params']['img_size'],
        in_channels=config['model_params']['in_channels'],
        out_channels=config['model_params']['out_channels'],
        feature_size=config['model_params']['feature_size'],
        use_checkpoint=config['model_params']['use_checkpoint']
        ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    # model.projection=nn.Identity() 
    if config['model_params']['type'] == "ContrastiveSwinUNETR_ml":
        model.projection_head0 = nn.Identity() 
        model.projection_head1 = nn.Identity() 
        model.projection_head2 = nn.Identity() 
        model.projection_head3 = nn.Identity() 
        model.projection_head4 = nn.Identity()
    
    # # 모델 가중치 불러오기
    weight_path = config['train_params']['t-sne_path']
    # model.load_state_dict(torch.load(weight_path, map_location=device)['model_state_dict'], strict=False)
    model.load_state_dict(torch.load(weight_path, map_location=device)['model_state_dict'], strict=False)
    # model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    logging.info(f"Model weights loaded from: {weight_path}")  # 모델 로드 경로 출력
    logging.info("Loading checkpoint... Well")
    
    # logging.info("no model loaded")
    return model,  criterion

def resize_label_to_match_feature(label, feature_map):
    """
    GT Label을 Feature Map 크기에 맞게 보간
    """
    # label = label.unsqueeze(1).float()  # (B, D, H, W) -> (B, 1, D, H, W)
    resized_label = F.interpolate(label, size=feature_map.shape[2:], mode='nearest')  # 크기 맞춤
    return resized_label.squeeze(1)  # (B, 1, D, H, W) -> (B, D, H, W)


def train_TSNE(loader, model, criterion, config, organ_name, max_samples=500):
    model.eval()
    deep_features = []
    actual = []
    meta_info = []  # (환자ID, 클래스label, voxel좌표 등)

    epoch_iterator = tqdm(loader, desc="Extracting features for t-SNE", dynamic_ncols=True)

    with torch.no_grad():
        for idx, batch in enumerate(epoch_iterator):
            # 1) Data & Forward
            with torch.cuda.amp.autocast():
                if torch.cuda.is_available():
                    images, labels = batch["image"].cuda(), batch["label"].cuda()
                output = sliding_window_inference(
                    inputs=images,
                    roi_size=config['model_params']['img_size'],
                    sw_batch_size=4,
                    predictor=model,
                    overlap=0.0,
                    device=torch.device("cpu"),
                    sw_device=torch.device("cuda")  # 보통은 입력과 동일한 GPU
                )
        
            batch_size = output.size(0)

            # 2) 각 배치의 샘플별로 feature & 메타정보 추출
            for b in range(batch_size):
                # 예시: batch["patient_id"]가 존재한다고 가정
                patient_id = os.path.basename(batch['image_meta_dict']['filename_or_obj'][0]).split('.nii.gz')[0]
                # patient_id = batch["patient_id"][b]  # 스트링이든 정수든

                for c in range(config['model_params']['out_channels']):
                    class_mask = (labels[b].squeeze(0) == c)
                    class_indices = torch.nonzero(class_mask, as_tuple=False).cpu()  # shape: [N, 3]

                    if class_indices.size(0) > 0:
                        # 원하는 개수만 랜덤 샘플링
                        num_samples_actual = min(max_samples, class_indices.size(0))
                        selected_indices = class_indices[torch.randperm(class_indices.size(0))[:num_samples_actual]]

                        # (num_samples_actual, feature_dim)
                        # output[b, :, x, y, z] => (feature_dim,)를 샘플링
                        sampled_features = output[b, :, 
                                                  selected_indices[:, 0],
                                                  selected_indices[:, 1],
                                                  selected_indices[:, 2]].numpy().T

                        # 3) feature와 메타정보를 "같은 순서"로 append
                        for i in range(num_samples_actual):
                            deep_features.append(sampled_features[i])     # (feature_dim,)
                            actual.append(c)
                            # voxel 좌표, 환자ID, 클래스 등을 dict로 저장
                            vx, vy, vz = selected_indices[i].tolist()
                            meta_info.append({
                                "patient_id": patient_id,
                                "class_label": c,
                                "voxel_index": (vx, vy, vz)
                            })

            # 메모리 정리
            del batch, images, labels, output
            torch.cuda.empty_cache()
    
    # 4) numpy array로 변환 (meta_info는 dict list 형태를 유지하거나, DataFrame으로 바로 변환)
    deep_features = np.array(deep_features)
    actual = np.array(actual)
    # meta_info = np.array(meta_info, dtype=object) # dict 리스트를 그대로 쓰는 경우가 많음

    return deep_features, actual, meta_info

def t_SNE(output,actual,val_output,val_actual,config,filename,output_dir):
    
    tsne=TSNE(n_components=2,random_state=0, perplexity=50)
    
    total_output=np.concatenate((output,val_output),axis=0)
    train_idx=output.shape[0]   
    
    cluster = np.array(tsne.fit_transform(total_output))
    
    train_cluster=cluster[:train_idx]
    val_cluster=cluster[train_idx:]
    
    fig = plt.figure(figsize=(10,10))
    dataname = ['background', 'tumor']
    
    ax1 = fig.add_subplot(221, aspect='equal')
    ax2 = fig.add_subplot(222, aspect='equal')    
    ax3 = fig.add_subplot(223, aspect='equal')    
    
    # Training data plot with color assignment
    for i, label in zip(range(config['model_params']['out_channels']), dataname):
        idx = np.where(actual == i)
        if i == 0: color = '#1E90FF'  # 훈련용 파랑 (간 배경)
        else: color = '#FFD700'  # 훈련용 노랑 (간 종양)
        ax1.scatter(train_cluster[idx, 0], train_cluster[idx, 1], marker='.', label=label, s=6, color=color, alpha=0.6)
        ax3.scatter(train_cluster[idx, 0], train_cluster[idx, 1], marker='.', label=label, s=6, color=color, alpha=0.6)
    
    # Validation data plot with color assignment
    for i, label in zip(range(config['model_params']['out_channels']), dataname):
        idx = np.where(val_actual == i)
        if i == 0: color = '#32CD32'  # 검증용 초록 (간 배경)
        else: color = '#FF6347'  # 검증용 빨강 (간 종양)
        ax2.scatter(val_cluster[idx, 0], val_cluster[idx, 1], marker='.', label=label, s=6, color=color, alpha=0.6)
        ax3.scatter(val_cluster[idx, 0], val_cluster[idx, 1], marker='.', label=label, s=6, color=color, alpha=0.6)
   
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    
    ax1.set_title('Train Data')
    ax2.set_title('Test Data')
    ax3.set_title('Combined Train + Test')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, markerscale=0.7)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, markerscale=0.7)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, markerscale=0.7)
    # 파일 확장자 처리
    if not filename.endswith('.png'):
        filename += '.png'
    # t-SNE 결과 저장
    # 저장 경로를 output_dir로 설정
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    plt.close()  # 저장 후 창을 닫습니다.
    print(f"t-SNE plot saved as {save_path}")
    return cluster

def t_SNE_all(
                lung_deep_features, lung_actual, lung_val_deep_features, lung_val_actual,
                liver_deep_features, liver_actual, liver_val_deep_features, liver_val_actual, config, output_dir
            ):
    
    # 라벨 및 색상 초기 설정
    t_sne_label_info = {
        "liver": {
            0: {"name": "background", "color": "#FF6666"},  # 배경
            1: {"name": "tumor", "color": "#800000"},       # 종양
        },
        "lung": {
            0: {"name": "background", "color": "#9ACD32"},  # 배경
            1: {"name": "tumor", "color": "#006400"},       # 종양
        },
    }

    # t-SNE 적용 데이터 병합
    total_output = np.concatenate(
        (lung_deep_features, lung_val_deep_features, liver_deep_features, liver_val_deep_features), axis=0
    )

    # 각 데이터셋의 시작과 끝 인덱스 기록
    liver_train_idx = liver_deep_features.shape[0]
    liver_val_idx = liver_train_idx + liver_val_deep_features.shape[0]
    lung_train_idx = liver_val_idx + lung_deep_features.shape[0]
    lung_val_idx = lung_train_idx + lung_val_deep_features.shape[0]

    # t-SNE 적용
    tsne = TSNE(n_components=2, random_state=0, perplexity=50)
    cluster = np.array(tsne.fit_transform(total_output))

    # t-SNE 결과 병합 (4개 데이터셋의 t-SNE 좌표)
    liver_train_cluster = cluster[:liver_train_idx]
    liver_val_cluster = cluster[liver_train_idx:liver_val_idx]
    lung_train_cluster = cluster[liver_val_idx:lung_train_idx]
    lung_val_cluster = cluster[lung_train_idx:lung_val_idx]

    # 시각화
    fig = plt.figure(figsize=(15, 15))
    datasets = [
        ("liver", liver_train_cluster, liver_actual, liver_val_cluster, liver_val_actual),
        ("lung", lung_train_cluster, lung_actual, lung_val_cluster, lung_val_actual),
    ]

    # Train 데이터 (subplot 1)
    ax1 = fig.add_subplot(221, aspect='equal')
    for organ, train_cluster, actual, _, _ in datasets:
        for label_idx, label_info in t_sne_label_info[organ].items():
            idx = np.where(np.array(actual) == label_idx)
            ax1.scatter(
                train_cluster[idx, 0], train_cluster[idx, 1],
                label=f"{organ} train {label_info['name']}",
                color=label_info["color"], s=6, alpha=0.6, marker='o'  # Train은 원형
            )
    ax1.set_title("Train Data (Liver + Lung)")
    ax1.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=8, markerscale=0.7)

    # Test 데이터 (subplot 2)
    ax2 = fig.add_subplot(222, aspect='equal')
    for organ, _, _, val_cluster, val_actual in datasets:
        for label_idx, label_info in t_sne_label_info[organ].items():
            idx = np.where(np.array(val_actual) == label_idx)
            ax2.scatter(
                val_cluster[idx, 0], val_cluster[idx, 1],
                label=f"{organ} test {label_info['name']}",
                color=label_info["color"], s=6, alpha=0.6, marker='^'  # Test는 삼각형
            )
    ax2.set_title("Test Data (Liver + Lung)")
    ax2.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=8, markerscale=0.7)

    # Train + Test 데이터 (subplot 3)
    ax3 = fig.add_subplot(223, aspect='equal')
    # Train 데이터
    for organ, train_cluster, actual, _, _ in datasets:
        for label_idx, label_info in t_sne_label_info[organ].items():
            idx_train = np.where(np.array(actual) == label_idx)
            ax3.scatter(
                train_cluster[idx_train, 0], train_cluster[idx_train, 1],
                label=f"{organ} train {label_info['name']}",
                color=label_info["color"], s=6, alpha=0.6, marker='o'  # Train은 원형
            )

    # Validation(Test) 데이터
    for organ, _, _, val_cluster, val_actual in datasets:
        for label_idx, label_info in t_sne_label_info[organ].items():
            idx_test = np.where(np.array(val_actual) == label_idx)
            ax3.scatter(
                val_cluster[idx_test, 0], val_cluster[idx_test, 1],
                label=f"{organ} test {label_info['name']}",
                color=label_info["color"], s=6, alpha=0.6, marker='^'  # Test는 삼각형
            )
    ax3.set_title("Train + Test Data (Liver + Lung)")
    ax3.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=8, markerscale=0.7)

    # 저장
    # 저장 경로를 output_dir로 설정
    save_path = os.path.join(output_dir, "liver_lung_t-sne.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # 창 닫기
    print(f"t-SNE plot saved as {save_path}")

    return cluster

def t_SNE_all_final(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]],
    save_dir: str = "./tsne_plots"
) -> None:
    """
    • splits: {
          "lung_train":  (feats, labels, meta_list),
          "lung_test":    (feats, labels, meta_list),
          "liver_train": (feats, labels, meta_list),
          "liver_test":   (feats, labels, meta_list),
      }
    • save_dir: PNG 파일을 저장할 디렉토리

    Returns:
      - cluster: 전체 t-SNE 좌표 (N×2)
      - index_map: key → (start_idx, end_idx)
    """

    # 미리 도메인과 스플릿 추출
    domains = ["liver", "lung", "both"]
    split_types = ["train", "test", "both"]
    
    # 라벨 및 색상 초기 설정
    t_sne_label_info = {
        "liver": {
            0: {"name": "background", "color": "#FF6666"},  # 배경
            1: {"name": "tumor", "color": "#800000"},       # 종양
        },
        "lung": {
            0: {"name": "background", "color": "#9ACD32"},  # 배경
            1: {"name": "tumor", "color": "#006400"},       # 종양
        },
    }
    marker_map  = {"train": "o", "test": "^"}
    subplot_idx = {"train":221, "test":222, "both":223}


    # 모든 데이터 합치기
    all_feats, all_labels, all_meta, all_tags = [], [], [], []
    index_map = {}
    offset = 0
    for key, (feats, labels, meta) in splits.items():
        organ, split = key.split("_")
        n = len(feats)
        index_map[key] = (offset, offset+n)
        offset += n
        all_feats.append(feats)
        all_labels.append(labels)
        all_meta.extend(meta)
        all_tags.extend([(organ, split)] * len(feats))

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # t-SNE 계산 (전체에 대해 한 번만)
    tsne = TSNE(n_components=2, random_state=0, perplexity=50)
    cluster = np.array(tsne.fit_transform(all_feats))

     # 4) organ × split 별로 PNG 저장
    for organ in domains:
        fig = plt.figure(figsize=(15, 15))
        for split in split_types:
            ax = fig.add_subplot(subplot_idx[split], aspect="equal")
            # 어떤 key들을 그릴지 결정
            if split == "both" and organ in ("liver", "lung"):
                keys = [f"{organ}_train", f"{organ}_test"]
            elif organ == "both" and split =="train":
                keys = ["liver_train", "lung_train"]
            elif organ == "both" and split =="test":
                keys = ["liver_test", "lung_test"]
            elif organ == "both" and split =="both":
                keys = ["liver_train", "liver_test", "lung_train", "lung_test"]
            else:
                keys = [f"{organ}_{split}"]

            for key in keys:
                st, ed = index_map[key]
                pts = cluster[st:ed]
                lbs = all_labels[st:ed]
                organ_name = key.split("_")[0]  # "liver" or "lung"
                split_name = key.split("_")[1]  # "train" or "test"

                for lbl, info in t_sne_label_info.get(organ_name, {}).items():
                    idx = np.where(lbs == lbl)
                    ax.scatter(
                        pts[idx, 0], pts[idx, 1],
                        s=6, alpha=0.6,
                        marker=marker_map.get(key.split("_")[1], "o"),
                        color=info["color"],
                        label=f"{key} {info['name']}"
                    )
            ax.set_title(f"{organ.capitalize()} | {split.capitalize()}")
            ax.axis("off")
            ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5),
                      fontsize=8, markerscale=0.7)

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"tsne_{organ}_all.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
    return cluster, index_map




def compute_uniformity(feats, t=2):
    from sklearn.metrics import pairwise_distances
    fn = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    dists = pairwise_distances(fn)
    np.fill_diagonal(dists, np.inf)  # 자기자신 제외
    return np.log(np.exp(-t * dists ** 2).mean())

def compute_stats(
    feats: np.ndarray, labels: np.ndarray
) -> Tuple[Dict[str,float], Dict[int,np.ndarray], np.ndarray]:
    """
    • feats: (N,C), labels: (N,)
    • 반환:
      - metrics: {alignment, uniformity, silhouette}
      - cents_hd: {class→(C,)} centroid in HD
      - theta: (N,) angles on unit circle after PCA2D
    """
    # L2 정규화
    fn = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    classes = np.unique(labels)
    # 1) centroid HD
    cents_hd = {c: fn[labels==c].mean(0) for c in classes}
    # 2) alignment
    ali = np.mean([
        np.linalg.norm(fn[labels==c] - cents_hd[c], axis=1).mean()
        for c in classes
    ])
    # 3) uniformity
    # pairs = [(a,b) for i,a in enumerate(classes) for b in classes[i+1:]]
    # uni = np.mean([np.linalg.norm(cents_hd[a]-cents_hd[b]) for a,b in pairs]) \
    #       if pairs else 0.0
    uni = compute_uniformity(fn)
    # 4) silhouette
    sil = float(silhouette_score(fn, labels)) if len(classes)>1 else 0.0
    # 5) PCA→2D θ
    #    (PCA는 호출 측에서 전체 데이터 PCA.fit 후 PCA.transform을 넣어주세요)
    return {"alignment":ali, "uniformity":uni, "silhouette":sil}, cents_hd, fn


def analyze_unit_sphere_all_v7(
    splits: Dict[str,Tuple[np.ndarray,np.ndarray,List[Dict[str,Any]]]],
    output_dir : str,
    mode: str="detail"   # "summary" or "detail"
) -> Dict[Tuple[str,str], Dict[str,float]]:
    # 0) 컬러·마커
    cmap = plt.get_cmap("tab10")
    color_map = {
        ("liver",0):cmap(0), ("liver",1):cmap(1),
        ("lung", 0):cmap(2), ("lung", 1):cmap(3),
        ("all",  0):cmap(4), ("all",  1):cmap(5),
    }
    marker_map = {"train":"o","test":"^"}
    cen_marker  = {"train":"p","test":"h"}

    # 1) 전체 PCA 학습
    all_feats = np.vstack([f/np.linalg.norm(f,axis=1,keepdims=True)
                           for f,_,_ in splits.values()])
    pca = PCA(2, random_state=0).fit(all_feats)

    # 2) subplot 설정
    if mode=="detail":
        rows = ["train","test","both"]
        cols = ["liver","lung","all"]
    else:  # summary
        rows = ["train","test","both"]
        cols = ["all"]

    fig, axes = plt.subplots(
        len(rows), len(cols),
        figsize=(5*len(cols),4*len(rows)),
        subplot_kw={"aspect":"equal"}
    )
    axes = np.atleast_2d(axes)
    for i,r in enumerate(rows):
        for j,c in enumerate(cols):
            ax = axes[i,j]
            ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05)
            ax.axis("off")
            ax.set_title(f"{r.capitalize()} | {c.capitalize()}")

    # 3) 모든 조합 데이터·메트릭 수집
    metrics: Dict[Tuple[str,str],Dict[str,float]] = {}
    data: Dict[Tuple[str,str],Dict[str,Any]] = {}

    for r in rows:
        for c in cols:
            # 3-1) split 필터
            names = []
            # 3-1) 기본 필터: split 기준
            if r == "both":
                names = list(splits.keys())
            else:
                names = [n for n in splits if r in n]
                # 안전 버전
                # names = [n for n in splits if n.split("_")[1] == r]


            # 3-2) organ 필터는 c ∈ {"liver", "lung"}일 때만 적용
            if mode == "detail" and c in ("liver", "lung"):
                names = [n for n in names if c in n]  # ✅ 기존 names에 대해 필터링

            # "all" 칼럼은 no additional filter

            # 3-3) feature·label 합치기
            feats = np.vstack([splits[n][0] for n in names])
            labels= np.hstack([splits[n][1] for n in names])
            meta   = sum([splits[n][2] for n in names], [])

            # 3-4) metric·centroid 계산(HD) + PCA→θ
            m, cents_hd, fn = compute_stats(feats, labels)
            proj2 = pca.transform(fn)
            theta = np.arctan2(proj2[:,1], proj2[:,0])

            # 3-5) 2D centroid 계산
            cents_2d = {
                cls: pca.transform(cents_hd[cls].reshape(1,-1))[0]
                for cls in cents_hd
            }

            # 3-6) meta 거리 기록
            all_cent = np.vstack([cents_hd[l] for l in labels])
            d2c = np.linalg.norm(fn - all_cent, axis=1)
            for idx, mtd in enumerate(meta):
                mtd["dist_to_centroid"] = float(d2c[idx])

            metrics[(r,c)] = m
            data[(r,c)]    = {"theta":theta, "labels":labels,
                               "cents2d":cents_2d}

    # 4) 그리기
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i, j]

            if r != "both":
                keys = [(r, c)]
            else:
                # both의 경우 train + test 데이터를 함께 그림
                keys = [("train", c), ("test", c)]

            for r_key, c_key in keys:
                if (r_key, c_key) not in data:
                    print("오류 발생")
                    continue  # 해당 조합이 존재하지 않을 수도 있으므로 안전 처리

                theta   = data[(r_key, c_key)]["theta"]
                labels  = data[(r_key, c_key)]["labels"]
                cents2d = data[(r_key, c_key)]["cents2d"]

                for cls in np.unique(labels):
                    mask = labels == cls
                    ax.scatter(np.cos(theta[mask]), np.sin(theta[mask]),
                            s=20, alpha=0.4,
                            color=color_map[(c if c=="all" else c_key, cls)],
                            label=f"{c if c=='all' else c_key}-{cls}-{r_key}",
                            marker=marker_map[r_key])

                # centroid 로직
                if c == "all" and r_key in ("train", "test"):
                    use_cents = data[(r_key, "all")]["cents2d"]
                else:
                    use_cents = cents2d

                for cls, coord in use_cents.items():
                    th = np.arctan2(coord[1], coord[0])
                    ax.scatter(np.cos(th), np.sin(th),
                            s=150, marker=cen_marker[r_key],
                            edgecolor="white", linewidth=1.5,
                            label=f"{c if c=='all' else c_key}-{cls}-{r_key}-cen",
                            color=color_map[(c if c=="all" else c_key, cls)])


    # 4-3) metrics text
    for i,r in enumerate(rows):
        for j,c in enumerate(cols):
            ax=axes[i,j]
            m = metrics[(r,c)]
            txt = f"Ali:{m['alignment']:.2f} Uni:{m['uniformity']:.2f}\nSil:{m['silhouette']:.2f}"
            ax.text(0, -1.15, txt, ha="center", va="top", fontsize=9)

            # 4-4) legend (중복 제거)
            h, lbl = ax.get_legend_handles_labels()
            uniq = {l:hh for hh,l in zip(h,lbl) if l not in {}}
            ax.legend(uniq.values(), uniq.keys(),
                        bbox_to_anchor=(1.02,1), loc="upper left",
                        fontsize=7)

    fig.subplots_adjust(right=0.8, bottom=0.15)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "Unit_sphere_all.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return metrics, data, splits



### Vis wiht metainfo
import pandas as pd
import plotly.express as px

def plot_tsne_with_meta(cluster, train_meta, val_meta, output_dir,organ, title='t-SNE Scatter Plot with Meta Information'):
    """
    train_cluster: (N_train, 2) numpy array, t-SNE 좌표
    train_meta: 길이 N_train의 dict 리스트 (각 dict: {"patient_id", "class_label", "voxel_index": (x,y,z)})
    val_cluster: (N_val, 2) numpy array, t-SNE 좌표
    val_meta: 길이 N_val의 dict 리스트, 위와 동일한 구조
    
    두 집합을 합쳐 Plotly scatter plot을 생성함.
    """
    train_idx=len(train_meta)
    
    train_cluster=cluster[:train_idx]
    val_cluster=cluster[train_idx:]
    
    rows = []
    
    # train 데이터 처리
    for coord, meta in zip(train_cluster, train_meta):
        row = {
            'tsne_x': coord[0],
            'tsne_y': coord[1],
            'dataset': 'train',
            'class': meta['class_label'],
            'patient_id': meta['patient_id'],
            'coord_x': meta['voxel_index'][0],
            'coord_y': meta['voxel_index'][1],
            'coord_z': meta['voxel_index'][2]
        }
        rows.append(row)
    
    # val (test) 데이터 처리
    for coord, meta in zip(val_cluster, val_meta):
        row = {
            'tsne_x': coord[0],
            'tsne_y': coord[1],
            'dataset': 'test',
            'class': meta['class_label'],
            'patient_id': meta['patient_id'],
            'coord_x': meta['voxel_index'][0],
            'coord_y': meta['voxel_index'][1],
            'coord_z': meta['voxel_index'][2]
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    # 그룹 정보: dataset와 class를 결합 (예: train_0, test_1 등)
    df['group'] = df['dataset'] + '_' + df['class'].astype(str)
    
    fig = px.scatter(df, 
                     x='tsne_x', 
                     y='tsne_y', 
                     color='group',
                     title=title,
                     hover_data=['dataset', 'class', 'patient_id', 'coord_x', 'coord_y', 'coord_z'])
    fig.write_html(os.path.join(output_dir,f"{organ}_tsne_interactive.html"))
    return df, fig

def plot_tsne_with_meta_all(cluster, liver_train_meta, liver_val_meta, lung_train_meta, lung_val_meta, output_dir,organ, title='t-SNE Scatter Plot with Meta Information'):
    """
    train_cluster: (N_train, 2) numpy array, t-SNE 좌표
    train_meta: 길이 N_train의 dict 리스트 (각 dict: {"patient_id", "class_label", "voxel_index": (x,y,z)})
    val_cluster: (N_val, 2) numpy array, t-SNE 좌표
    val_meta: 길이 N_val의 dict 리스트, 위와 동일한 구조
    
    두 집합을 합쳐 Plotly scatter plot을 생성함.
    """
        # 각 데이터셋의 시작과 끝 인덱스 기록
    liver_train_idx = len(liver_train_meta)
    liver_val_idx = liver_train_idx + len(liver_val_meta)
    lung_train_idx = liver_val_idx + len(lung_train_meta)
    lung_val_idx = lung_train_idx + len(lung_val_meta)

    
    # t-SNE 결과 병합 (4개 데이터셋의 t-SNE 좌표)
    liver_train_cluster = cluster[:liver_train_idx]
    liver_val_cluster = cluster[liver_train_idx:liver_val_idx]
    lung_train_cluster = cluster[liver_val_idx:lung_train_idx]
    lung_val_cluster = cluster[lung_train_idx:lung_val_idx]
    
    
    rows = []
    
    # liver_train 데이터 처리
    for coord, meta in zip(liver_train_cluster, liver_train_meta):
        row = {
            'tsne_x': coord[0],
            'tsne_y': coord[1],
            'dataset': 'liver_train',
            'class': meta['class_label'],
            'patient_id': meta['patient_id'],
            'coord_x': meta['voxel_index'][0],
            'coord_y': meta['voxel_index'][1],
            'coord_z': meta['voxel_index'][2]
        }
        rows.append(row)
    
    # val (test) 데이터 처리
    for coord, meta in zip(liver_val_cluster, liver_val_meta):
        row = {
            'tsne_x': coord[0],
            'tsne_y': coord[1],
            'dataset': 'liver_test',
            'class': meta['class_label'],
            'patient_id': meta['patient_id'],
            'coord_x': meta['voxel_index'][0],
            'coord_y': meta['voxel_index'][1],
            'coord_z': meta['voxel_index'][2]
        }
        rows.append(row)
    
    # lung_train 데이터 처리
    for coord, meta in zip(lung_train_cluster, lung_train_meta):
        row = {
            'tsne_x': coord[0],
            'tsne_y': coord[1],
            'dataset': 'lung_train',
            'class': meta['class_label'],
            'patient_id': meta['patient_id'],
            'coord_x': meta['voxel_index'][0],
            'coord_y': meta['voxel_index'][1],
            'coord_z': meta['voxel_index'][2]
        }
        rows.append(row)
    
    # val (test) 데이터 처리
    for coord, meta in zip(lung_val_cluster, lung_val_meta):
        row = {
            'tsne_x': coord[0],
            'tsne_y': coord[1],
            'dataset': 'lung_test',
            'class': meta['class_label'],
            'patient_id': meta['patient_id'],
            'coord_x': meta['voxel_index'][0],
            'coord_y': meta['voxel_index'][1],
            'coord_z': meta['voxel_index'][2]
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # 그룹 정보: dataset와 class를 결합 (예: train_0, test_1 등)
    df['group'] = df['dataset'] + '_' + df['class'].astype(str)
    
    fig = px.scatter(df, 
                     x='tsne_x', 
                     y='tsne_y', 
                     color='group',
                     title=title,
                     hover_data=['dataset', 'class', 'patient_id', 'coord_x', 'coord_y', 'coord_z'])
    fig.write_html(os.path.join(output_dir,f"{organ}_tsne_interactive.html"))
    return df, fig

def plot_tsne_with_meta_all_final(
    cluster: np.ndarray,
    index_map: Dict[str, Tuple[int, int]],
    results: Dict[str, Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]],
    output_dir: str,
    save_name: str = "tsne_interactive_all"
):
    """
    • cluster: 전체 t-SNE 결과 (N, 2)
    • index_map: split key -> (start_idx, end_idx)
    • meta_map: split key -> 메타데이터 리스트
    """

    rows = []
    for key, (start, end) in index_map.items():
        organ, split = key.split("_")
        cluster_part = cluster[start:end]
        meta_part = results[key][2]  # results에서 직접 meta 추출

        for coord, meta in zip(cluster_part, meta_part):
            row = {
                'tsne_x': coord[0],
                'tsne_y': coord[1],
                'dataset': f"{organ}_{split}",
                'class': meta['class_label'],
                'patient_id': meta['patient_id'],
                'coord_x': meta['voxel_index'][0],
                'coord_y': meta['voxel_index'][1],
                'coord_z': meta['voxel_index'][2],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df["group"] = df["dataset"] + "_" + df["class"].astype(str)

    fig = px.scatter(
        df,
        x="tsne_x",
        y="tsne_y",
        color="group",
        title="Unified t-SNE with Meta Information",
        hover_data=["dataset", "class", "patient_id", "coord_x", "coord_y", "coord_z"]
    )
    html_path = os.path.join(output_dir, f"{save_name}.html")
    fig.write_html(html_path)

    df.to_csv(os.path.join(output_dir, f"{save_name}.csv"), index=False)
    
    # figure를 JSON 문자열로 변환
    liver_json_str = fig.to_json()

    # 저장할 디렉토리와 파일명을 지정
    liver_file_path = os.path.join(output_dir, f"{save_name}.json")

    # JSON 파일로 저장
    with open(liver_file_path, "w") as f:
        f.write(liver_json_str)


    return df, fig



# 예시: 함수 호출 시
# df, fig = plot_tsne_with_meta(train_cluster, train_meta, val_cluster, val_meta)

'''
Below here would be past code or specific purpose functions
'''


def validate_ml(loader, model, criterion, config):
    """validation"""
    model.eval()

    deep_features = [[] for _ in range(5)]
    actual = [[] for _ in range(5)]
    
    # losses = AverageMeter()
    # top1 = AverageMeter()

    with torch.no_grad():
    
        for idx, batch in enumerate(loader):
            if torch.cuda.is_available():
                # test_inputs,test_labels  = (batch["image"].cuda(), batch["label"].cuda())
                images = batch["image"].cuda()    
                labels = batch["label"].cuda()
            
            bsz = labels.shape[0]

            # forward
            output = model(images)
            # output = sliding_window_inference(
            #                     inputs=images,
            #                     roi_size=tuple(config['model_params']['img_size']),
            #                     sw_batch_size=4,
            #                     predictor=model
            #                 )
            # print("val output",output.shape )
            # loss = criterion(output, labels)
            
            # 각 scale level별로 feature와 label을 저장
            for i in range(5):
                deep_features[i] += output[i].cpu().numpy().tolist()
                label_resized = resize_label_to_match_feature(labels, output[i])
                actual[i] += label_resized.cpu().numpy().tolist()
            
    
            # update metric
            # losses.update(loss.item(), bsz)
            # acc= accuracy(output, labels)
            # top1.update(acc, bsz)


            # if idx % config.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #            idx, len(loader), batch_time=batch_time,
            #            loss=losses, top1=top1))

    # deep_features=np.array(deep_features)
    # actual = np.array(actual)
    
    # numpy 배열로 변환
    deep_features = [np.transpose(np.array(f), (0, 2, 3, 4, 1)) for f in deep_features]
    actual = [np.array(a) for a in actual]
    
    
    #Alignment
    classes =2
    alignments = []
    uniformities = []
    for i in range(5):
        class_centroid=np.zeros((5,classes, deep_features[i].shape[-1]))

        align_per_class=[]
        center_pdist_list=[]    
        
        for label in range(classes):
            idx=np.where(actual[i]==label)
            # deep_selected = deep_features[i][idx[0], idx[1], idx[2], :]  # feature_dim 유지
            deep_selected = deep_features[i][idx]  # (N, feature_dim) 형태
            if len(idx) == 0:
                continue  # 해당 클래스 데이터가 없으면 패스

            # ⭐ 샘플링: 5000개 이하라면 전체 사용, 5000개 초과면 랜덤 샘플링
            num_samples = min(5000, deep_selected.shape[0])
            sampled_indices = np.random.choice(deep_selected.shape[0], num_samples, replace=False)
            deep_selected = deep_selected[sampled_indices]

            pdist=distance.pdist(deep_selected,metric='euclidean')
            align=np.mean(pdist)
            align_per_class.append(align)
            class_centroid[i][label]=np.mean(deep_features[i][idx],axis=0)
            
        align_per_class=np.array(align_per_class)
        alignments.append(align_per_class)  # 각 feature level의 alignment 저장
        
        #Uniformity
        class_centroid=torch.Tensor(class_centroid[i])
        pdist=nn.PairwiseDistance(p=2)
        
        for one_label in range(classes):
            for other_label in range(one_label+1,classes):
                center_pdist=pdist(class_centroid[one_label],class_centroid[other_label])
                center_pdist_list.append(center_pdist)
        
        center_pdist_list=np.array(center_pdist_list)        
        
        assert center_pdist_list.shape[0]== (classes*(classes-1))/2
            
        uniformity=np.mean(center_pdist_list)
        uniformities.append(uniformity)
    
    
    
    
    # return  deep_features,  actual, align_per_class, uniformity
    return  deep_features,  actual, alignments, uniformities

def validate(loader, model, criterion, config):
    """validation"""
    model.eval()

    deep_features = []
    actual = []
    
    with torch.no_grad():
    
        for idx, batch in enumerate(loader):
            if torch.cuda.is_available():
                images = batch["image"].cuda()    
                labels = batch["label"].cuda()
            
            bsz = labels.shape[0]

            # forward
            repre, output = model(images)
            # output = sliding_window_inference(
            #                     inputs=images,
            #                     roi_size=tuple(config['model_params']['img_size']),
            #                     sw_batch_size=4,
            #                     predictor=model
            #                 )
            # print("val output",output.shape )
            # loss = criterion(output, labels)
                        
            deep_features += output.cpu().numpy().tolist()
            actual += labels.cpu().numpy().tolist()
            
    
            # update metric
            # losses.update(loss.item(), bsz)
            # acc= accuracy(output, labels)
            # top1.update(acc, bsz)


            # if idx % config.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #            idx, len(loader), batch_time=batch_time,
            #            loss=losses, top1=top1))

    
    # numpy 배열로 변환
    deep_features=np.array(deep_features)
    actual = np.array(actual)    
    
    #Alignment
    classes =2
    alignments = []
    uniformities = []
    
    class_centroid=np.zeros(classes, deep_features.shape[-1])

    align_per_class=[]
    center_pdist_list=[]    
        
    for label in range(classes):
        idx=np.where(actual==label)
        # deep_selected = deep_features[i][idx[0], idx[1], idx[2], :]  # feature_dim 유지
        deep_selected = deep_features[idx]  # (N, feature_dim) 형태
        if len(idx) == 0:
            continue  # 해당 클래스 데이터가 없으면 패스

        # 샘플링: 5000개 이하라면 전체 사용, 5000개 초과면 랜덤 샘플링
        num_samples = min(5000, deep_selected.shape[0])
        sampled_indices = np.random.choice(deep_selected.shape[0], num_samples, replace=False)
        deep_selected = deep_selected[sampled_indices]

        pdist=distance.pdist(deep_selected,metric='euclidean')
        align=np.mean(pdist)
        align_per_class.append(align)
        class_centroid[label]=np.mean(deep_features[idx],axis=0)
        
    align_per_class=np.array(align_per_class)
    alignments.append(align_per_class)  # 각 feature level의 alignment 저장
    
    #Uniformity
    class_centroid=torch.Tensor(class_centroid)
    pdist=nn.PairwiseDistance(p=2)
    
    for one_label in range(classes):
        for other_label in range(one_label+1,classes):
            center_pdist=pdist(class_centroid[one_label],class_centroid[other_label])
            center_pdist_list.append(center_pdist)
    
    center_pdist_list=np.array(center_pdist_list)        
    
    assert center_pdist_list.shape[0]== (classes*(classes-1))/2
        
    uniformity=np.mean(center_pdist_list)
    uniformities.append(uniformity)

    
    
    
    # return  deep_features,  actual, align_per_class, uniformity
    return  deep_features,  actual, alignments, uniformities

def validate_tumor_old(loader, model, criterion, config, tumor_label=1):
    """validation"""
    model.eval()

    deep_features = []
    tumor_features_list =[]
    with torch.no_grad():
    
        for idx, batch in enumerate(loader):
            if torch.cuda.is_available():
                images = batch["image"].cuda()    
                labels = batch["label"]
            
            bsz = labels.shape[0]

            # forward
            # repre, output = model(images)
            repre = sliding_window_inference(
                                inputs=images,
                                roi_size=tuple(config['model_params']['img_size']),
                                sw_batch_size=4,
                                predictor=model,
                                overlap=0.0
                            ).cpu().detach()
            # print("val output",output.shape )
            # loss = criterion(output, labels)
                        
            # deep_features += output.cpu().numpy().tolist()
            # actual += labels.cpu().numpy().tolist()


            B, C, D, H, W = repre.shape
            output_flat = repre.view(B, C, -1)           # [B, C, N]
            labels_flat = labels.view(B, -1)                # [B, N]
            
            for b in range(B):
                mask = (labels_flat[b] == tumor_label)  # tumor 영역
                tumor_feats = output_flat[b, :, mask]     # [C, N_tumor]
                if tumor_feats.shape[1] == 0:
                    continue
                tumor_feats = tumor_feats.permute(1, 0)     # [N_tumor, C]
                num_feats = tumor_feats.shape[0]
                # 한 환자당 최소 1000개 feature 확보
                if num_feats >= 1000:
                    indices = np.random.choice(num_feats, 1000, replace=False)
                else:
                    indices = np.random.choice(num_feats, num_feats, replace=True)
                sampled_feats = tumor_feats[indices, :]
                tumor_features_list.append(sampled_feats.cpu().numpy())
    
    if len(tumor_features_list) > 0:
        combined = np.concatenate(tumor_features_list, axis=0)
        # 전체가 5000개를 초과하면 무작위로 5000개로 제한
        if combined.shape[0] > 5000:
            indices = np.random.choice(combined.shape[0], 5000, replace=False)
            combined = combined[indices]
        return combined
    else:
        return None
    # numpy 배열로 변환
    # deep_features=np.array(deep_features)
    # actual = np.array(actual)    
    tumor_features_list = np.array(tumor_features_list)
    
    # #Alignment
    # classes =2
    # alignments = []
    # uniformities = []
    
    # class_centroid=np.zeros(classes, deep_features.shape[-1])

    # align_per_class=[]
    # center_pdist_list=[]    
        
    # for label in range(classes):
    #     idx=np.where(actual==label)
    #     # deep_selected = deep_features[i][idx[0], idx[1], idx[2], :]  # feature_dim 유지
    #     deep_selected = deep_features[idx]  # (N, feature_dim) 형태
    #     if len(idx) == 0:
    #         continue  # 해당 클래스 데이터가 없으면 패스

    #     # 샘플링: 5000개 이하라면 전체 사용, 5000개 초과면 랜덤 샘플링
    #     num_samples = min(5000, deep_selected.shape[0])
    #     sampled_indices = np.random.choice(deep_selected.shape[0], num_samples, replace=False)
    #     deep_selected = deep_selected[sampled_indices]

    #     pdist=distance.pdist(deep_selected,metric='euclidean')
    #     align=np.mean(pdist)
    #     align_per_class.append(align)
    #     class_centroid[label]=np.mean(deep_features[idx],axis=0)
        
    # align_per_class=np.array(align_per_class)
    # alignments.append(align_per_class)  # 각 feature level의 alignment 저장
    
    # #Uniformity
    # class_centroid=torch.Tensor(class_centroid)
    # pdist=nn.PairwiseDistance(p=2)
    
    # for one_label in range(classes):
    #     for other_label in range(one_label+1,classes):
    #         center_pdist=pdist(class_centroid[one_label],class_centroid[other_label])
    #         center_pdist_list.append(center_pdist)
    
    # center_pdist_list=np.array(center_pdist_list)        
    
    # assert center_pdist_list.shape[0]== (classes*(classes-1))/2
        
    # uniformity=np.mean(center_pdist_list)
    # uniformities.append(uniformity)
    
    # # return  deep_features,  actual, align_per_class, uniformity
    # return  deep_features,  actual, alignments, uniformities
    return tumor_features_list

def validate_tumor(loader, model, criterion, config, tumor_label=1):
    """validation"""
    model.eval()

    patient_id_list=[]
    tumor_features_list =[]
    voxel_indices_list=[]
    with torch.no_grad():
    
        for idx, batch in enumerate(loader):
            if torch.cuda.is_available():
                images = batch["image"].cuda()    
                labels = batch["label"]
                

            repre = sliding_window_inference(
                                inputs=images,
                                roi_size=tuple(config['model_params']['img_size']),
                                sw_batch_size=4,
                                predictor=model,
                                overlap=0.0
                            ).cpu().detach()

            B, C, D, H, W = repre.shape
            output_flat = repre.view(B, C, -1)           # [B, C, N]
            labels_flat = labels.view(B, -1)                # [B, N]
            
            for b in range(B):
                mask = (labels_flat[b] == tumor_label)  # tumor 영역
                tumor_feats = output_flat[b, :, mask]     # [C, N_tumor]
                if tumor_feats.shape[1] == 0:
                    continue
                tumor_feats = tumor_feats.permute(1, 0)     # [N_tumor, C]
                num_feats = tumor_feats.shape[0]
                voxel_id = torch.where(mask)[0].cpu().numpy()
                # 한 환자당 최소 1000개 feature 확보
                if num_feats >= 1000:
                    indices = np.random.choice(num_feats, 1000, replace=False)
                else:
                    indices = np.random.choice(num_feats, num_feats, replace=True)
                sampled_feats = tumor_feats[indices, :]
                tumor_features_list.append(sampled_feats.cpu().numpy())
                patient_id_list.append(idx)
                voxel_indices_list.append(voxel_id)
    
    if len(tumor_features_list) > 0:
        combined = np.concatenate(tumor_features_list, axis=0)
        # 전체가 5000개를 초과하면 무작위로 5000개로 제한
        if combined.shape[0] > 5000:
            indices = np.random.choice(combined.shape[0], 5000, replace=False)
            combined = combined[indices]
        return combined
    else:
        return None

def t_SNE_tumor(liver_tumor_features, lung_tumor_features):
    # 리스트 내 각 배열을 concatenate하여 하나의 array로 만듦
    # liver_feats = np.concatenate(liver_tumor_features, axis=0)
    # lung_feats = np.concatenate(lung_tumor_features, axis=0)
    
    combined_feats = np.concatenate([liver_tumor_features, lung_tumor_features], axis=0)
    # 레이블: 0 -> Liver Tumor, 1 -> Lung Tumor
    labels = np.array([0] * liver_tumor_features.shape[0] + [1] * lung_tumor_features.shape[0])
    # 각 feature vector를 L2 정규화
    combined_feats = combined_feats / np.linalg.norm(combined_feats, axis=1, keepdims=True)
   
    tsne = TSNE(n_components=2, random_state=0, perplexity=50)
    embedding = tsne.fit_transform(combined_feats)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(embedding[labels==0, 0], embedding[labels==0, 1], s=6, label='Liver Tumor', color='blue')
    plt.scatter(embedding[labels==1, 0], embedding[labels==1, 1], s=6, label='Lung Tumor', color='red')
    plt.title("t-SNE Visualization: Liver vs Lung Tumor Features", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.legend(loc="best")
    plt.axis('equal')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "tsne_tumor2.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

# def t_SNE_multi_layer(train_features, train_labels, val_features, val_labels, feature_level):
    """
    t-SNE를 활용한 feature embedding 시각화
    - feature_level: 0~4 (각 스케일의 feature map 선택)
    """
    
    for level in range(0,feature_level):
        print(f"Processing t-SNE for feature level {level}...")

        tsne = TSNE(n_components=2, random_state=0, perplexity=50)
        # ✅ 1️⃣ Feature를 (N, dim) 형태로 변환
        train_feature = train_features[level].reshape(-1, train_features[level].shape[-1])  # [N, C]
        val_feature = val_features[level].reshape(-1, val_features[level].shape[-1])  # [N, C]
        # ✅ 2️⃣ Label을 (N,) 형태로 변환
        train_label = train_labels[level].flatten()  # [N]
        val_label = val_labels[level].flatten()  # [N]
        # total_output = np.concatenate((train_features[feature_level], val_features[feature_level]), axis=0)
        # total_labels = np.concatenate((train_labels[feature_level], val_labels[feature_level]), axis=0)
        

        # ✅ 3️⃣ 샘플링을 위한 인덱스 생성
        back_num, tumor_num = 4750, 250  # 샘플 개수 설정

        # Background(0) 샘플링
        train_bg_idx = np.where(train_label == 0)[0]
        val_bg_idx = np.where(val_label == 0)[0]
        train_bg_sampled = np.random.choice(train_bg_idx, min(back_num, len(train_bg_idx)), replace=False)
        val_bg_sampled = np.random.choice(val_bg_idx, min(back_num, len(val_bg_idx)), replace=False)

        # Tumor(1) 샘플링
        train_tumor_idx = np.where(train_label == 1)[0]
        val_tumor_idx = np.where(val_label == 1)[0]
        train_tumor_sampled = np.random.choice(train_tumor_idx, min(tumor_num, len(train_tumor_idx)), replace=False)
        val_tumor_sampled = np.random.choice(val_tumor_idx, min(tumor_num, len(val_tumor_idx)), replace=False)

        # ✅ 4️⃣ 샘플링된 데이터만 사용
        sampled_train_feature = np.concatenate([train_feature[train_bg_sampled], train_feature[train_tumor_sampled]], axis=0)
        sampled_val_feature = np.concatenate([val_feature[val_bg_sampled], val_feature[val_tumor_sampled]], axis=0)
        sampled_features = np.concatenate([sampled_train_feature, sampled_val_feature], axis=0)

        # t-SNE 적용
        classes=2
        cluster = tsne.fit_transform(sampled_features)
        train_idx = sampled_train_feature.shape[0]  # 샘플링된 train 데이터 개수
        train_cluster = cluster[:train_idx]
        val_cluster = cluster[train_idx:]

        fig, ax = plt.subplots(figsize=(10, 10))
        labels = ['Background', 'Tumor']
        colors = ['blue', 'red']

        for i, label in enumerate([0, 1]):  # Background(0), Tumor(1)
            train_idx = np.where(train_label[np.concatenate([train_bg_sampled, train_tumor_sampled])] == label)[0]
            val_idx = np.where(val_label[np.concatenate([val_bg_sampled, val_tumor_sampled])] == label)[0]

            ax.scatter(train_cluster[train_idx, 0], train_cluster[train_idx, 1], label=f"Train {labels[i]}", s=6, alpha=0.6, marker='o', color=colors[i])
            ax.scatter(val_cluster[val_idx, 0], val_cluster[val_idx, 1], label=f"Validation {labels[i]}", s=6, alpha=0.6, marker='^', color=colors[i])

        ax.set_title(f"t-SNE Visualization - Feature Level {level}")
        ax.legend()
        plt.savefig(os.path.join(output_dir, f"tSNE_feature_level_{level}.png"))
        plt.close()

        print(f"Feature level {level} t-SNE completed and saved!")


def main():
    
    # build data loader
    liver_train_loader, lung_train_loader, liver_test_loader, lung_test_loader = set_loader(config)

    # build model and criterion
    model, criterion = set_model(config)


    results = {}
    ##
    # 아래 장기별 진행은 하드 코딩 그러나 아래 부분도 규칙이 있어 언젠가 장기별 train/val 구조로 장기 늘어나도 문제없는 형태로 수정가능
    ##
    for (loader, organ, split) in [
        (lung_train_loader, "lung", "train"),
        (lung_test_loader, "lung", "test"),
        (liver_train_loader, "liver", "train"),
        (liver_test_loader, "liver", "test"),
    ]:
        key = f"{organ}_{split}"
        feats, labels, metas = train_TSNE(loader, model, criterion, config, organ)
        results[key] = (feats, labels, metas)
        del loader
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


    all_cluster, index_map = t_SNE_all_final(results,output_dir)
    metrics, data, splits = analyze_unit_sphere_all_v7(results,output_dir)

    print("completed")



    # print("\n ")
    # print("----------------------------------------------")
    
    # print("train_align per class",align_per_class)
    # print("train align",train_align)
    # print("\n ")
    # print("val_align per class",val_align_per_class)
    # print("val align",val_align)
    
    # print("----------------------------------------------")
    # print("train uniformity: ", uniformity)
    # print("val uniformity: ", val_uniformity)

    # plt.show()

if __name__ == '__main__':
    main()
  