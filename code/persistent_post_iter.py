# # **[MED] [3D] [SEG] Swin UNETR**
# 
# A novel segmentation model termed Swin UNEt TRansformers (Swin UNETR). Specially for the task of 3D semantic segmentation.
# HSM change ipynb to pythonfile

# [![GitHub watch](https://img.shields.io/github/watchers/LeonidAlekseev/Swin-UNETR.svg?style=social&label=Watch&maxAge=2592000)](https://github.com/LeonidAlekseev/Swin-UNETR/)


### further detail
# for mem 4090 using train dataset cahce 12, 0.5 will demand 20G
# np 1.21.6 torch1.13.0+cu117
#### Environment
#### make sure use swin_unetr conda env
# conda activate swin_unetr

####
#  CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --master_port=29501 --nproc_per_node=2 /data/Swin-Unetr/Swin-UNETR/notebook/Swin_UNETR_MSD_Liver_Lung_DDP.py
import os
import random
import shutil
import yaml
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import glob
import json
import pprint
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import einops
import warnings
import logging
import wandb
from monai.transforms import (Compose, AsDiscrete, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, CropForegroundd, Orientationd,
                               Spacingd, EnsureTyped, RandFlipd,RandShiftIntensityd, RandCropByPosNegLabeld, MapLabelValued, AsDiscreted,
                               RandScaleIntensityd,RandRotate90d, ToTensord, Resized,FgBgToIndicesd, Invertd)
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
from torch.utils.data import ConcatDataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.utils import set_determinism
from monai.config import print_config
from monai.handlers.utils import from_engine
import sys
sys.path.append("/data/hyungseok/Swin-UNETR")

from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.my_utils import *
from losses.contrastive import *

import torch.nn.functional as F


def main():
    
    parser = argparse.ArgumentParser(description="Swin UNETR training")
    parser.add_argument('--config', type=str, default="/data/hyungseok/Swin-UNETR/api/exp.yaml", help="Path to the YAML config file")
    args = parser.parse_args()
    # YAML 설정 파일 로드
    def load_config(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    # 학습 설정 로드
    config_path = args.config
    config = load_config(config_path)

    # 학습환경 설정
    set_determinism(seed=config['train_params']['seed']) # 5개 융합
    torch.manual_seed(config['train_params']['seed'])
    torch.cuda.manual_seed(config['train_params']['seed'])
    torch.cuda.manual_seed_all(config['train_params']['seed'])
    random.seed(config['train_params']['seed'])
    np.random.seed(config['train_params']['seed'])
    cudnn.benchmark = config['cuda']['benchmark']
    cudnn.deterministic = config['cuda']['deterministic']
     
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config['cuda']['CUDA_VISIBLE_DEVICES']))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 폴더 구조 설정
    experiment_name = generate_experiment_name(config)
    date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = os.path.join(config['data']['save_dir'], experiment_name,date)
    log_dir = os.path.join(config['data']['log_dir'], experiment_name,date)


    os.makedirs(output_dir, exist_ok=True) 
    os.makedirs(log_dir, exist_ok=True)
    # 로그 설정: log_dir에 로그 저장
    setup_logging(log_dir)
    # config 파일을 JSON으로 저장
    log_experiment_config()
    save_config_as_json(config, log_dir)
    # 학습 시작 메시지
    logging.info("Training started with configuration:")
    logging.info(config)

    wandb.init(project=config['wandb']['project_name'],name=config['wandb']['experiment_name'],
    config=config,dir=log_dir, mode=config['wandb']['mode'])
    
    # 기록하고 싶은 파일 리스트
    # Train, Trainer, model, loss 정도?
    # Model과 Trainer의 소스 파일 경로 가져오기
    # file_list에 추가
    file_list = []
    # 현재 실행 중인 파일 추가 (__file__ 변수 사용)
    if '__file__' in globals():
        file_list.append(os.path.abspath(__file__))
    # 임시 디렉토리 생성
    tmp_dir = config['data']['tmp_dir']
    os.makedirs(tmp_dir, exist_ok=True)

    # 파일 복사
    for file_name in file_list:
        shutil.copy(file_name, tmp_dir)
    # 임시 디렉토리를 wandb에 기록
    wandb.run.log_code(tmp_dir)
    # 사용 후 임시 디렉토리 삭제
    shutil.rmtree(tmp_dir)
    current_script_name = __file__
    save_current_code(log_dir,current_script_name)
    # print_config()
    
    # ### Transforms
    # 2024.09.25 현재 label은 종양만 해서 한가지만 남김
    # 2024.09.26 현재 patch 96 유지 못해서 일단 64로 진행
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
            FgBgToIndicesd(
                keys="label",
                fg_postfix="_fg",
                bg_postfix="_bg",
                image_key="image",
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(config['train_params']['spatial_size']),
                pos=config['transforms']['liver']['RandCropByPosNegLabeld_params']['pos'],
                neg=config['transforms']['liver']['RandCropByPosNegLabeld_params']['neg'],
                num_samples=config['transforms']['num_samples'],
                fg_indices_key="label_fg",
                bg_indices_key="label_bg",
            ),
            # RandCropByPosNegLabeld(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=tuple(config['train_params']['spatial_size']),
            #     pos=config['transforms']['liver']['RandCropByPosNegLabeld_params']['pos'],
            #     neg=config['transforms']['liver']['RandCropByPosNegLabeld_params']['neg'],
            #     num_samples=config['transforms']['num_samples'],
            #     image_key="image",
            #     image_threshold=0,
            # ),
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
            CropForegroundd(keys=["image", "label"], source_key="image",),
            FgBgToIndicesd(
                keys="label",
                fg_postfix="_fg",
                bg_postfix="_bg",
                image_key="image",
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=tuple(config['train_params']['spatial_size']),
                pos=config['transforms']['lung']['RandCropByPosNegLabeld_params']['pos'],
                neg=config['transforms']['lung']['RandCropByPosNegLabeld_params']['neg'],
                num_samples=config['transforms']['num_samples'],
                fg_indices_key="label_fg",
                bg_indices_key="label_bg",
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
    
    liver_post_pred = Compose([
        Invertd(
            keys="pred",  # 예측값에만 Invertd 적용
            transform=liver_val_transforms,  # 원본과 동일한 전처리
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
            transform=lung_val_transforms,  # 원본과 동일한 전처리
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

    # **📌 모든 랜덤 변환에 시드 적용**
    set_all_random_states(lung_train_transforms.transforms, seed=1234)
    set_all_random_states(lung_val_transforms.transforms, seed=1234)
    set_all_random_states(liver_train_transforms.transforms, seed=1234)
    set_all_random_states(liver_val_transforms.transforms, seed=1234)
    # ### Dataset
    # set_track_meta(False) - Do not activate  (전처리에서 spacingd등 메타정보 필요한것을 못하게 막음)
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
    
    lung_train_images, lung_train_labels = load_lungfile_from_txt(config['data']['lung_train_image_txt'], config['data']['lung_image_dir'])
    liver_train_images, liver_train_labels = load_liverfile_from_txt(config['data']['liver_train_image_txt'], config['data']['liver_image_dir'])
    # Validation 데이터 로드
    lung_val_images, lung_val_labels = load_lungfile_from_txt(config['data']['lung_val_image_txt'],  config['data']['lung_image_dir'])
    liver_val_images, liver_val_labels = load_liverfile_from_txt(config['data']['liver_val_image_txt'],  config['data']['liver_image_dir'])
    
   # Dataset 구성
    liver_dataset_json = {
        "labels": {
            "0": "background",
            "1": "cancer",
        },
        "tensorImageSize": "3D",
        "training": [{"image": img, "label": lbl} for img, lbl in zip(liver_train_images, liver_train_labels)],
        "validation": [{"image": img, "label": lbl} for img, lbl in zip(liver_val_images, liver_val_labels)]
    }

    lung_dataset_json = {
        "labels": {
            "0": "background",
            "1": "cancer",
        },
        "tensorImageSize": "3D",
        "training": [{"image": img, "label": lbl} for img, lbl in zip(lung_train_images, lung_train_labels)],
        "validation": [{"image": img, "label": lbl} for img, lbl in zip(lung_val_images, lung_val_labels)]
    }

    ###json 파일 경로
    lung_datasets = os.path.join(output_dir, 'lung_dataset.json')
    liver_datasets = os.path.join(output_dir, 'liver_dataset.json')

    with open(lung_datasets, 'w') as outfile:
        json.dump(lung_dataset_json, outfile)
    pprint.pprint(lung_dataset_json)
    with open(liver_datasets, 'w') as outfile:
        json.dump(liver_dataset_json, outfile)
    pprint.pprint(liver_dataset_json)
    
    #### For Debug run below
    # lung_datasets = "/data/Swin-Unetr/Swin-UNETR/results/debug/lung_dataset_tsne.json"
    # liver_datasets = "/data/Swin-Unetr/Swin-UNETR/results/debug/liver_dataset_tsne.json"

    # CacheDataset 및 DataLoader 구성
    liver_train_files = load_decathlon_datalist(liver_datasets, True, "training")
    liver_val_files = load_decathlon_datalist(liver_datasets, True, "validation")
    lung_train_files = load_decathlon_datalist(lung_datasets, True, "training")
    lung_val_files = load_decathlon_datalist(lung_datasets, True, "validation")

    
    liver_train_ds = PersistentDataset(data=liver_train_files, transform=liver_train_transforms, cache_dir="/data/hyungseok/Swin-UNETR/data_cache/Liver_Lung/Liver/train")
    lung_train_ds = PersistentDataset(data=lung_train_files, transform=lung_train_transforms,cache_dir="/data/hyungseok/Swin-UNETR/data_cache/Liver_Lung/Lung/train")
    # combined_train_dataset = ConcatDataset([liver_train_ds, lung_train_ds])

    liver_val_ds = PersistentDataset(data=liver_val_files, transform=liver_val_transforms,cache_dir="/data/hyungseok/Swin-UNETR/data_cache/Liver_Lung/Liver/val/raw")
    lung_val_ds = PersistentDataset(data=lung_val_files, transform=lung_val_transforms,cache_dir="/data/hyungseok/Swin-UNETR/data_cache/Liver_Lung/Lung/val/raw")
                                                                                
    # combined_val_dataset = ConcatDataset([liver_val_ds, lung_val_ds])
    
    liver_train_loader = DataLoader(liver_train_ds, batch_size=config['train_params']['batch_size'], 
        num_workers=config['train_params']['num_workers'], pin_memory=config['train_params']['pin_memory'],
        shuffle= True, worker_init_fn=worker_init_fn)
    lung_train_loader = DataLoader(lung_train_ds, batch_size=config['train_params']['batch_size'], 
        num_workers=config['train_params']['num_workers'], pin_memory=config['train_params']['pin_memory'],
        shuffle= True,worker_init_fn=worker_init_fn)
    liver_val_loader = DataLoader(liver_val_ds, batch_size=config['train_params']['val_batch_size'], 
        num_workers=config['train_params']['val_num_workers'], pin_memory=config['train_params']['pin_memory'],
        worker_init_fn=worker_init_fn)
    lung_val_loader = DataLoader(lung_val_ds, batch_size=config['train_params']['val_batch_size'], 
        num_workers=config['train_params']['val_num_workers'], pin_memory=config['train_params']['pin_memory'],
        worker_init_fn=worker_init_fn)
    # val_loader = DataLoader(combined_val_dataset, batch_size=config['train_params']['val_batch_size'], 
    #                         num_workers=config['train_params']['num_workers'], pin_memory=config['train_params']['pin_memory'])


    #### Model
    ##### Create

    
    if config['model_params']['type'] == "SwinUnetr":
        model = SwinUNETR(
            img_size=config['model_params']['img_size'],
            in_channels=config['model_params']['in_channels'],
            out_channels=config['model_params']['out_channels'],
            feature_size=config['model_params']['feature_size'],
            use_checkpoint=config['model_params']['use_checkpoint']
        )
          
    elif config['model_params']['type'] == "AttSwinUNETR":
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

                    
                logits = self.out(out.float())
                return logits

    
        model = AttSwinUNETR(
        img_size=config['model_params']['img_size'],
        in_channels=config['model_params']['in_channels'],
        out_channels=config['model_params']['out_channels'],
        feature_size=config['model_params']['feature_size'],
        use_checkpoint=config['model_params']['use_checkpoint']
        )
        
        
    # ### Load weights

    weight = torch.load(config['data']['weights_path'])
    model.load_from(weights=weight)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # ### Training


    def validation(epoch_iterator_val_liver, epoch_iterator_val_lung):
        model.eval()
        dice_metric.reset()  # Reset metric at the start of validation
        with torch.no_grad():
            for organ_type, epoch_iterator_val in zip(["liver", "lung"], [epoch_iterator_val_liver, epoch_iterator_val_lung]):
                temp_dice_metric.reset()
                for step, batch in enumerate(epoch_iterator_val):
                    val_inputs = batch["image"].cuda()
                    with torch.cuda.amp.autocast():
                        # val_outputs = sliding_window_inference(val_inputs, config['model_params']['img_size'], num_samples, lambda x: model(x, label=None,organ=organ_type))
                        batch["pred"] = sliding_window_inference(val_inputs, config['model_params']['img_size'], num_samples, model).detach().cpu()
                    if organ_type == "liver":
                        batch = [liver_post_pred(i) for i in decollate_batch(batch)]
                    else:
                        batch = [lung_post_pred(i) for i in decollate_batch(batch)]
                        
                    val_output_convert, val_labels_convert = from_engine(["pred", "label"])(batch)
                    val_output_convert = val_output_convert[0].unsqueeze(0)
                    val_labels_convert =val_labels_convert[0].unsqueeze(0)
                    dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                    temp_dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                    epoch_iterator_val.set_description(
                        f"Validate {organ_type} ({step + 1} / {len(epoch_iterator_val)} Steps)"
                    )
                 # organ별 aggregate 후 출력 (임시 metric은 organ마다 새로 생성했으므로 reset 불필요)
                organ_dice = temp_dice_metric.aggregate().item()
                logging.info(f"{organ_type.capitalize()} Mean Dice: {organ_dice:.4f}")
            mean_dice_val = dice_metric.aggregate().item()
            dice_metric.reset()
        return mean_dice_val



    def train(epoch, liver_train_loader,lung_train_loader):
        model.train()
        epoch_loss = 0
        epoch_contrastive_loss = 0
        dice_metric_train.reset()  # Reset Dice metric at the beginning of each epoch
        
        # liver_train_loader와 lung_train_loader를 iterator 형태로 변환
        liver_iter = iter(liver_train_loader)
        lung_iter = iter(lung_train_loader)
        iterations = max(len(liver_train_loader), len(lung_train_loader))
        epoch_iterator = tqdm(
            range(iterations), desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        step=0
        for step, batch in enumerate(epoch_iterator):
            # x, y = (batch["image"].cuda(), batch["label"].cuda())
            try:
                liver_batch = next(liver_iter)
            except StopIteration:
                liver_iter = iter(liver_train_loader)
                liver_batch = next(liver_iter)
            try:
                lung_batch = next(lung_iter)
            except StopIteration:
                lung_iter = iter(lung_train_loader)
                lung_batch = next(lung_iter)
            with torch.cuda.amp.autocast():
                # contrastive loss를 사용하는 경우
                if config['train_params']['use_contrastive_loss']:
                    # Training Dice score calculation
                    x = torch.cat((liver_batch["image"], lung_batch["image"]), dim=0).cuda()
                    y = torch.cat((liver_batch["label"], lung_batch["label"]), dim=0).cuda()
                    logit_map= model(x,label=y)
                    
                else:
                    # ContrastiveLoss를 사용하지 않음
                    x = torch.cat((liver_batch["image"], lung_batch["image"]), dim=0).cuda()
                    y = torch.cat((liver_batch["label"], lung_batch["label"]), dim=0).cuda()
                    logit_map = model(x)
                    loss = loss_function(logit_map, y)
            # Gradient Accumulation을 위해 Loss를 나누어 축적
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            # 매 `accumulation_steps` 마다 optimizer.step()을 호출
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
             # 원래 손실을 복원하여 누적
            epoch_loss += loss.item() * accumulation_steps
            
            # Training Dice score 계산 (10 스텝마다 수행)
            if step % 10 == 0:  # 매 10 스텝마다 수행
                y_pred = decollate_batch(logit_map)
                y_pred = [post_pred(i) for i in y_pred]
                y = decollate_batch(y)
                y = [post_label(i) for i in y]
                dice_metric_train(y_pred, y)
        
            epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)"
            % (epoch, max_epoch, loss * accumulation_steps))  # 원래 Loss로 표시
        
        # 마지막 남은 미니 배치 처리
        if (step + 1) % accumulation_steps != 0 and accumulation_steps != 1:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        epoch_loss /= (step+1)
        epoch_contrastive_loss /= (step + 1)  # contrastive loss의 에포크 평균 계산
        # Aggregate dice score
        train_dice = dice_metric_train.aggregate().item()
        dice_metric_train.reset()
        return epoch_loss ,epoch_contrastive_loss , train_dice

    max_epoch = config['train_params']['max_epoch']
    eval_epoch = config['train_params']['eval_epoch']
    if config['train_params']['loss_type'] == "DiceCELoss":
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    elif config['train_params']['loss_type'] == "DiceCELoss+Cont3":
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
        pixel_contrast_loss_function  = BalSCL([0.95,0.05],config['CONTRASTIVE']['TEMPERATURE']).cuda()
    elif config['train_params']['loss_type'] == "DiceCELoss+Cont1":
        loss_function = DiceFocalLoss(to_onehot_y=True,softmax=True,gamma=2.0,lambda_dice=1.0,lambda_focal=1.0)
        pixel_contrast_loss_function  = BalSCL([0.95,0.05],config['CONTRASTIVE']['TEMPERATURE']).cuda()
    if config['train_params']['optim_type'] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['train_params']['learning_rate']), weight_decay=float(config['train_params']['weight_decay']))
    elif config['train_params']['optim_type'] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=float(config['train_params']['learning_rate']), momentum=0.9)
        
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config['train_params']['warmup'], max_epochs=config['train_params']['max_epoch'])
    scaler = torch.cuda.amp.GradScaler() 
    post_label = AsDiscrete(to_onehot=config['model_params']['out_channels']) # class n
    post_pred = AsDiscrete(argmax=True, to_onehot=config['model_params']['out_channels']) # class n
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    temp_dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric_train = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    accumulation_steps = config['train_params']['gradient_accumulation_steps']
    
    epoch = 0
    dice_val_best = 0.0
    epoch_best = 0
    epoch_loss_values = []
    epoch_contrastive_loss_values=[]
    metric_values = []

    if config['train_params']['resume'] :
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(config['train_params']['resume_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 스케줄러 상태 복구
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        epoch = checkpoint['epoch']
        dice_val_best = checkpoint['best_dice']
        logging.info(f"Resuming from epoch {epoch} with best dice {dice_val_best:.4f}")
    
    while epoch < max_epoch:
        epoch_loss,epoch_contrastive_loss, train_dice = train(epoch, liver_train_loader, lung_train_loader)
        logging.info(f"Epoch {epoch}, Training loss {epoch_loss:.4f}, Cont loss {epoch_contrastive_loss : .4f}, Training Dice {train_dice:.4f}")
        wandb.log({"lr": scheduler.get_last_lr()[0], "training_loss": epoch_loss, "Cont_loss":epoch_contrastive_loss, "training_dice": train_dice},step=epoch)
        scheduler.step()
        # WandB로 학습 손실 기록
        epoch_loss_values.append(epoch_loss)
        epoch_contrastive_loss_values.append(epoch_contrastive_loss)
        if (epoch % eval_epoch == 0 and epoch != 0) or epoch == max_epoch:
            liver_epoch_iterator_val = tqdm(
                liver_val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            lung_epoch_iterator_val = tqdm(
                lung_val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(liver_epoch_iterator_val,lung_epoch_iterator_val)
            metric_values.append(dice_val)
            # WandB로 검증 결과 기록
            wandb.log({"validation_dice": dice_val},step=epoch)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                epoch_best = epoch
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),  # 스케줄러 상태 추가
                    'epoch': epoch,
                    'best_dice': dice_val_best
                }, os.path.join(output_dir, "best_metric_model.pth"))
                logging.info(f"\nModel Was Saved ! Current {epoch} epoch is Best Avg. Dice: {dice_val_best}")
            else:
                logging.info(f"\nModel Was Not Saved ! Current Best Avg Dice at {epoch_best}epoch: {dice_val_best} Current {epoch}epoch Avg. Dice: {dice_val}")
        if (epoch % config['train_params']['save_epoch'] == 0 and epoch != 0):
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),  # 스케줄러 상태 추가
                    'epoch': epoch,
                    'best_dice': dice_val_best
                },  os.path.join(output_dir,f"checkpoint_epoch_{epoch}.pth"))
            logging.info(f"\nCheckpoint frequently saved at epoch {epoch}.")
        epoch += 1
        # flush 호출하여 로그 즉시 기록
        for handler in logging.getLogger().handlers:
            handler.flush()

    logging.info(f"train completed, best_metric: {dice_val_best:.4f} at epoch: {epoch_best}")
    np.savez(os.path.join(output_dir, "training_metrics.npz"), epoch_loss_values=epoch_loss_values, metric_values=metric_values)

# 최종 학습 결과를 WandB에 기록
    wandb.log({"best_metric": dice_val_best, "final_epoch": epoch_best})
    # checkpoint = torch.load(os.path.join(output_dir, "best_metric_model.pth"))
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    ### show the progress of training
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("epoch Average Loss")
    x = [eval_epoch * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [eval_epoch * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    # plt.show()
    # 경로와 파일명 지정하여 저장
    plt.savefig(os.path.join(log_dir, f"training_progress_{epoch}.png"))
    plt.close()  # 저장 후 창 닫기



if __name__ == "__main__":
    main()
