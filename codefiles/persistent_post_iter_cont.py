
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
#  CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --master_port=29501 --nproc_per_node=2 /data/hyungseok/Swin-UNETR/notebook/Swin_UNETR_MSD_Liver_Lung_DDP.py
import os
import random
# os.environ["CUDA_VISIBLE_DEVICES"]= "2"
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
from monai.transforms import Compose, AsDiscrete,AsDiscreted, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, CropForegroundd, Orientationd, Spacingd, EnsureTyped, RandFlipd,RandShiftIntensityd, RandCropByPosNegLabeld, MapLabelValued,RandScaleIntensityd,RandRotate90d, ToTensord, FgBgToIndicesd, Invertd
from monai.data import (
    PersistentDataset,
    DataLoader,
    Dataset,
    list_data_collate,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
from monai.handlers.utils import from_engine
from monai.data.meta_tensor import MetaTensor
from torch.utils.data import ConcatDataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from monai.config import print_config
import sys
sys.path.append('/data/hyungseok/Swin-UNETR')
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.my_utils import *



def main():
    
    parser = argparse.ArgumentParser(description="Swin UNETR training")
    parser.add_argument('--config', type=str, default="/data/hyungseok/Swin-UNETR/api/exp_cont.yaml", help="Path to the YAML config file")
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
    set_track_meta(True)
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
    
    ####json 파일 경로
    lung_datasets = os.path.join(output_dir, 'lung_dataset.json')
    liver_datasets = os.path.join(output_dir, 'liver_dataset.json')

    with open(lung_datasets, 'w') as outfile:
        json.dump(lung_dataset_json, outfile)
    pprint.pprint(lung_dataset_json)
    with open(liver_datasets, 'w') as outfile:
        json.dump(liver_dataset_json, outfile)
    pprint.pprint(liver_dataset_json)
    
    # #### For Debug run below
    lung_debug_datasets = "/data/hyungseok/Swin-UNETR/results/debug/lung_dataset_train_check.json"
    liver_debug_datasets = "/data/hyungseok/Swin-UNETR/results/debug/liver_dataset_train_check.json"

    # CacheDataset 및 DataLoader 구성
    liver_train_files = load_decathlon_datalist(liver_datasets, True, "training")
    liver_val_files = load_decathlon_datalist(liver_datasets, True, "validation")
    lung_train_files = load_decathlon_datalist(lung_datasets, True, "training")
    lung_val_files = load_decathlon_datalist(lung_datasets, True, "validation")
    liver_train_debug_files = load_decathlon_datalist(liver_debug_datasets, True, "training")
    lung_train_debug_files = load_decathlon_datalist(lung_debug_datasets, True, "training")
    
    liver_train_ds = PersistentDataset(data=liver_train_files, transform=liver_train_transforms, cache_dir="/data/hyungseok/Swin-UNETR/data_cache/Liver_Lung/Liver/train")
    lung_train_ds = PersistentDataset(data=lung_train_files, transform=lung_train_transforms,cache_dir="/data/hyungseok/Swin-UNETR/data_cache/Liver_Lung/Lung/train")
    # combined_train_dataset = ConcatDataset([liver_train_ds, lung_train_ds])

    liver_val_ds = PersistentDataset(data=liver_val_files, transform=liver_val_transforms,cache_dir="/data/hyungseok/Swin-UNETR/data_cache/Liver_Lung/Liver/val/raw")
    lung_val_ds = PersistentDataset(data=lung_val_files, transform=lung_val_transforms,cache_dir="/data/hyungseok/Swin-UNETR/data_cache/Liver_Lung/Lung/val/raw")
                                                                                
    liver_train_check_ds = PersistentDataset(data=liver_train_debug_files, transform=liver_val_transforms,cache_dir="/data/hyungseok/Swin-UNETR/data_cache/Liver_Lung/Liver/val/train_check")
    lung_train_check_ds = PersistentDataset(data=lung_train_debug_files, transform=lung_val_transforms,cache_dir="/data/hyungseok/Swin-UNETR/data_cache/Liver_Lung/Lung/val/train_check")

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

    liver_train_check_loader = DataLoader(liver_train_check_ds, batch_size=config['train_params']['val_batch_size'], 
        num_workers=config['train_params']['val_num_workers'], pin_memory=config['train_params']['pin_memory'],
        worker_init_fn=worker_init_fn)
    lung_train_check_loader = DataLoader(lung_train_check_ds, batch_size=config['train_params']['val_batch_size'], 
        num_workers=config['train_params']['val_num_workers'], pin_memory=config['train_params']['pin_memory'],
        worker_init_fn=worker_init_fn)


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
        
    if config['model_params']['type'] == "ContrastiveSwinUNETR":
        
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
                self.projection_head = nn.Sequential(
                    nn.Conv3d(in_dim, hidden_dim, kernel_size=1),
                    nn.BatchNorm3d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(hidden_dim, out_dim, kernel_size=1)
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
                logits = self.out(out)
                # 검증 모드에서는 pixel_embeddings 계산을 생략
                if self.training:  # training=True일 때만 contrastive 학습용 features 계산
                    # Projection head에 통과시키기
                    pixel_embeddings = self.projection_head(features_before_output)

                    # L2 정규화를 적용하여 embeddings을 normalize
                    pixel_embeddings = nn.functional.normalize(pixel_embeddings, p=2, dim=1)
                    
                    # logits과 pixel_embeddings 반환
                    return logits, pixel_embeddings
                else:
                    return logits
    
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
                self.max_views = config['CONTRASTIVE']['MAX_VIEWS']
                self.base_temperature = config['CONTRASTIVE']['BASE_TEMPERATURE']
                self.queue_size = config['CONTRASTIVE']['QUEUE_SIZE']
                self.dim = config['CONTRASTIVE']['DIM']
                self.num_classes = config['CONTRASTIVE']['NUM_CLASSES']
                self.ignore_label = 255
                self.mode = config['CONTRASTIVE']['MODE']  # 1: 기존 방식, 2: memory bank 방식, 3: hard sampling 방식

                if self.mode > 1:
                    # memory bank (queue) 추가
                    self.register_buffer("pixel_queue", torch.randn(self.num_classes, self.queue_size, self.dim))
                    self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
                    self.register_buffer("pixel_queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))

            @torch.no_grad()
            def update_queue(self, embeddings, labels):
                # embeddings: [Batch, C, D, H, W]
                # labels: [Batch, D, H, W]
                
                for cls in range(self.num_classes):
                    # 해당 클래스에 속하는 voxel 선택
                    cls_indices = (labels == cls).nonzero(as_tuple=True)
                    if len(cls_indices[0]) == 0:
                        continue

                    # 해당 클래스의 피처 추출
                    cls_embeddings = embeddings[cls_indices]

                    # 차원 확인 후 정규화
                    if cls_embeddings.dim() == 1:
                        cls_embeddings = cls_embeddings.unsqueeze(1)
                    
                    batch_size = cls_embeddings.shape[0]
                    ptr = int(self.pixel_queue_ptr[cls])

                    # 큐 업데이트
                    if ptr + batch_size > self.queue_size:
                        self.pixel_queue[cls, ptr:] = nn.functional.normalize(cls_embeddings[:self.queue_size - ptr], p=2, dim=1)
                        self.pixel_queue_ptr[cls] = 0
                        self.pixel_queue[cls, :batch_size - (self.queue_size - ptr)] = nn.functional.normalize(cls_embeddings[self.queue_size - ptr:], p=2, dim=1)
                        self.pixel_queue_ptr[cls] = batch_size - (self.queue_size - ptr)
                    else:
                        self.pixel_queue[cls, ptr:ptr + batch_size] = nn.functional.normalize(cls_embeddings, p=2, dim=1)
                        self.pixel_queue_ptr[cls] = (ptr + batch_size) % self.queue_size

            def _sample_classes(self, X, y):
                # X: [Batch, C, D, H, W]
                # y: [Batch, D, H, W]
                batch_size, feat_dim = X.shape[0], X.shape[-1]
                classes = torch.unique(y)
                classes = [clsid for clsid in classes if clsid != self.ignore_label]

                if len(classes) == 0:
                    return None, None

                X_class_samples = []
                y_class_samples = []

                for cls in classes:
                    cls_indices = (y == cls).nonzero(as_tuple=True)  # 3D로 인덱스 저장
                    num_samples = min(len(cls_indices[0]), self.max_views)
                    perm = torch.randperm(len(cls_indices[0]))[:num_samples]
                    
                    # 각 차원의 인덱스 선택
                    selected_batch_indices = cls_indices[0][perm]
                    selected_depth_indices = cls_indices[1][perm]
                    selected_height_indices = cls_indices[2][perm]
                    selected_width_indices = cls_indices[3][perm]

                    # 인덱스를 사용해 X에서 해당 위치의 값들을 추출
                    X_selected = X[selected_batch_indices, :, selected_depth_indices, selected_height_indices, selected_width_indices]

                    X_class_samples.append(X_selected)
                    y_class_samples.append(torch.full((num_samples,), cls, dtype=torch.long).cuda())

                if len(X_class_samples) == 0:
                    return None, None

                X_class_samples = torch.cat(X_class_samples, dim=0)
                y_class_samples = torch.cat(y_class_samples, dim=0)
                return X_class_samples, y_class_samples

            def _sample_from_memory_bank(self, X, y):
                """
                Memory bank에서 샘플링하여 contrastive 학습에 사용할 피처들을 반환.
                X: [Batch, C, D, H, W]
                y: [Batch, D, H, W]
                """
                X_memory_samples = []
                y_memory_samples = []

                for cls in range(self.num_classes):
                    # 해당 클래스의 voxel 인덱스 추출
                    cls_indices = (y == cls).nonzero(as_tuple=True)

                    if len(cls_indices[0]) > 0:
                        # 메모리 뱅크에서 샘플링
                        memory_indices = torch.randperm(self.queue_size)[:self.max_views]
                        memory_features = self.pixel_queue[cls][memory_indices].cuda()

                        # 샘플링된 메모리 피처와 라벨 추가
                        X_memory_samples.append(memory_features)
                        y_memory_samples.append(torch.full((memory_features.size(0),), cls, dtype=torch.long).cuda())

                if len(X_memory_samples) == 0:
                    return None, None

                # 리스트로 저장된 샘플을 배치 차원으로 합침
                X_memory_samples = torch.cat(X_memory_samples, dim=0)
                y_memory_samples = torch.cat(y_memory_samples, dim=0)

                return X_memory_samples, y_memory_samples

            
            def _hard_anchor_sampling(self, X, y_hat, y):
                batch_size, feat_dim = X.shape[0], X.shape[-1]
                classes = torch.unique(y)

                if len(classes) == 0:
                    return None, None

                X_class_samples = []
                y_class_samples = []

                for cls in classes:
                    hard_indices = ((y_hat == cls) & (y != cls)).nonzero(as_tuple=True)
                    easy_indices = ((y_hat == cls) & (y == cls)).nonzero(as_tuple=True)

                    num_hard = len(hard_indices[0])
                    num_easy = len(easy_indices[0])
                    n_view = self.max_views

                    if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                        num_hard_keep = n_view // 2
                        num_easy_keep = n_view - num_hard_keep
                    elif num_hard >= n_view / 2:
                        num_easy_keep = num_easy
                        num_hard_keep = n_view - num_easy_keep
                    elif num_easy >= n_view / 2:
                        num_hard_keep = num_hard
                        num_easy_keep = n_view - num_hard_keep
                    else:
                        if num_easy + num_hard > 0:
                            combined_indices = (
                                torch.cat((hard_indices[0], easy_indices[0])),
                                torch.cat((hard_indices[1], easy_indices[1])),
                                torch.cat((hard_indices[2], easy_indices[2])),
                                torch.cat((hard_indices[3], easy_indices[3]))
                            )
                            if num_easy + num_hard < n_view:
                                queue_indices = torch.randperm(self.queue_size)[:(n_view - num_easy - num_hard)]
                                queue_features = self.pixel_queue[cls][queue_indices].cuda()
                                X_class_samples.append(torch.cat([X[combined_indices], queue_features], dim=0))
                                y_class_samples.append(torch.tensor([cls], dtype=torch.long).cuda().clone().detach())
                            continue
                        else:
                            queue_indices = torch.randperm(self.queue_size)[:n_view]
                            combined_features = self.pixel_queue[cls][queue_indices].cuda()
                            X_class_samples.append(combined_features)
                            y_class_samples.append(torch.tensor([cls], dtype=torch.long).cuda().clone().detach())
                            continue

                    perm_hard = torch.randperm(num_hard)[:num_hard_keep]
                    perm_easy = torch.randperm(num_easy)[:num_easy_keep]
                    selected_hard_indices = (hard_indices[0][perm_hard], hard_indices[1][perm_hard], hard_indices[2][perm_hard], hard_indices[3][perm_hard])
                    selected_easy_indices = (easy_indices[0][perm_easy], easy_indices[1][perm_easy], easy_indices[2][perm_easy], easy_indices[3][perm_easy])

                    if len(selected_hard_indices[0]) > 0 or len(selected_easy_indices[0]) > 0:
                        combined_indices = (
                            torch.cat((selected_hard_indices[0], selected_easy_indices[0])),
                            torch.cat((selected_hard_indices[1], selected_easy_indices[1])),
                            torch.cat((selected_hard_indices[2], selected_easy_indices[2])),
                            torch.cat((selected_hard_indices[3], selected_easy_indices[3]))
                        )
                        combined_features = X[combined_indices]
                        if combined_features.size(0) < n_view:
                            queue_indices = torch.randperm(self.queue_size)[:(n_view - combined_features.size(0))]
                            queue_features = self.pixel_queue[cls][queue_indices].cuda()
                            combined_features = torch.cat([combined_features, queue_features], dim=0)

                        X_class_samples.append(combined_features)
                        y_class_samples.append(torch.tensor([cls], dtype=torch.long).cuda().clone().detach())

                if len(X_class_samples) == 0:
                    return None, None
                X_class_samples = torch.stack(X_class_samples, dim=0)
                y_class_samples = torch.stack(y_class_samples, dim=0)
                return X_class_samples, y_class_samples


            def _contrastive(self, X_anchor, y_anchor):
                anchor_num = X_anchor.shape[0]
                anchor_feature = X_anchor

                mask = torch.eq(y_anchor.unsqueeze(1), y_anchor.unsqueeze(0)).float().cuda()

                anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, anchor_feature.T), self.temperature)
                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast - logits_max.detach()

                logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(anchor_num).view(-1, 1).cuda(), 0)
                mask = mask * logits_mask

                neg_mask = 1 - mask
                neg_logits = torch.exp(logits) * neg_mask
                neg_logits = neg_logits.sum(1, keepdim=True)

                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits + neg_logits + 1e-10)

                mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)

                loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                loss = loss.mean()

                return loss

            def forward(self, feats, labels, predict=None):
                set_track_meta(False)
                # feats : [Batch, aug, dim, D, W, H]
                batch_size, num_views, feat_dim, depth, height, width = feats.size()

                # labels = labels.unsqueeze(1).float().clone()
                labels = torch.nn.functional.interpolate(labels, (feats.shape[-3], feats.shape[-2], feats.shape[-1]), mode='nearest')
                labels = labels.squeeze(1).long()
                
                feats = feats.view(batch_size * num_views, feat_dim, depth, height, width)
                # Option 1: 기존 방식
                if self.mode == 1:
                    feats_, labels_ = self._sample_classes(feats, labels)

            # Option 2: memory bank 방식
                elif self.mode == 2:
                    feats_, labels_ = self._sample_classes(feats, labels)
                    if feats_ is not None and labels_ is not None:
                        self.update_queue(feats_, labels_)

                        # memory bank에서 샘플을 추가로 가져와서 사용
                        memory_feats, memory_labels = self._sample_from_memory_bank(feats_, labels_)
                        if memory_feats is not None and memory_labels is not None:
                            feats_ = torch.cat([feats_, memory_feats], dim=0)
                            labels_ = torch.cat([labels_, memory_labels], dim=0)

                # Option 3: hard sampling 방식
                elif self.mode == 3:
                    predict = predict.argmax(dim=1).long()
                    feats = feats.permute(0, 2, 4, 3, 1)
                    feats_, labels_ = self._hard_anchor_sampling(feats, predict, labels)
                    if feats_ is not None and labels_ is not None:
                        feats_ = feats_.view(-1, feats_.shape[-1])  # [cls * max_view, dim] 형태로 변환
                        labels_ = labels_.view(-1)  # [cls * max_view] 형태로 변환
                        # 각 클래스에 대해 max_view만큼 반복된 labels_ 생성
                        labels_ = labels_.repeat_interleave(self.max_views)  # [cls * max_view] 형태로 변환
                        # 각 클래스에 대해 실제 샘플 수에 맞게 라벨을 반복
                        # labels_ = torch.cat([torch.full((feats_.size(1),), cls, dtype=torch.long).cuda() for cls in torch.unique(labels_)], dim=0)
                        self.update_queue(feats_, labels_)
                        
                        # 메모리 bank에서 샘플을 추가로 가져와서 사용
                        memory_feats, memory_labels = self._sample_from_memory_bank(feats_, labels_)
                        if memory_feats is not None and memory_labels is not None:
                            feats_ = torch.cat([feats_, memory_feats], dim=0)  # 현재 샘플과 메모리 샘플 결합
                            labels_ = torch.cat([labels_, memory_labels], dim=0)  # 라벨도 결합

                if feats_ is None or labels_ is None:
                    return torch.tensor(0.0).cuda()

                loss = self._contrastive(feats_, labels_)
                set_track_meta(True)
                return loss

        
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
            # 간과 폐 배치를 결합하여 단일 배치 생성
            x = torch.cat((liver_batch["image"], lung_batch["image"]), dim=0).cuda()
            y = torch.cat((liver_batch["label"], lung_batch["label"]), dim=0).cuda()
            with torch.cuda.amp.autocast():
                # contrastive loss를 사용하는 경우
                if config['train_params']['use_contrastive_loss']:
                    # Training Dice score calculation
                    logit_map,repre = model(x)
                    
                    # y_pred = [post_pred(i) for i in y_pred]
                    y_pred = decollate_batch(logit_map)
                    y_pred = [post_pred(i) for i in y_pred]
                    y_pred = torch.stack(y_pred,dim=0)
                    loss = loss_function(logit_map, y)
                    contrast_loss = pixel_contrast_loss_function(repre.unsqueeze(1), y, y_pred)
                    # DiceCELoss + ContrastiveLoss 가중합
                    loss = loss + config['train_params']['lambda_contrast'] * contrast_loss
                    epoch_contrastive_loss += contrast_loss.item()
                else:
                    # ContrastiveLoss를 사용하지 않음
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
    # if config['train_params']['loss_type'] == "DiceCELoss":
    #     loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    # elif config['train_params']['loss_type'] == "DiceCELoss+Cont3":
    #     loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
    #     pixel_contrast_loss_function  = PixelContrastLoss(config)
    # elif config['train_params']['loss_type'] == "DiceCELoss+Cont1":
    #     loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
    #     pixel_contrast_loss_function  = PixelContrastLoss(config)
    # elif config['train_params']['loss_type'] == "DiceCELoss+Cont2":
    #     loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
    #     pixel_contrast_loss_function  = PixelContrastLoss(config)

    loss_type = config['train_params']['loss_type']
    if loss_type == "DiceCELoss":
        # 1) plain DiceCE must not be using contrastive loss
        assert not config['train_params'].get("use_contrastive_loss", False), (
            "config['train_params']['use_contrastive_loss'] must be False when using DiceCELoss"
        )
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

    elif loss_type == "DiceCELoss+Cont1":
        # 2a) Cont1 ⇒ CONTRASTIVE MODE == 1
        assert config['CONTRASTIVE'].get("MODE") == 1, (
            "For DiceCELoss+Cont1, config['CONTRASTIVE']['MODE'] must be 1"
        )
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        pixel_contrast_loss_function = PixelContrastLoss(config)

    elif loss_type == "DiceCELoss+Cont2":
        # 2b) Cont2 ⇒ CONTRASTIVE MODE == 2
        assert config['CONTRASTIVE'].get("MODE") == 2, (
            "For DiceCELoss+Cont2, config['CONTRASTIVE']['MODE'] must be 2"
        )
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        pixel_contrast_loss_function = PixelContrastLoss(config)

    elif loss_type == "DiceCELoss+Cont3":
        # 2c) Cont3 ⇒ CONTRASTIVE MODE == 3
        assert config['CONTRASTIVE'].get("MODE") == 3, (
            "For DiceCELoss+Cont3, config['CONTRASTIVE']['MODE'] must be 3"
        )
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        pixel_contrast_loss_function = PixelContrastLoss(config)

    else:
        raise ValueError(f"Unrecognized loss_type: {loss_type}")

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
    train_metric_values =[]
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
    
    # 에포크별 저장한 Dice 값을 기록하기 위한 dict
    saved_checkpoints: dict[int, float] = {}
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
                logging.info(f"\nModel Was Saved ! Current {epoch}epoch is Best Avg. Dice: {dice_val_best}")
            else:
                logging.info(f"\nModel Was Not Saved ! Current Best Avg Dice at {epoch_best}epoch: {dice_val_best} Current {epoch}epoch Avg. Dice: {dice_val}")
        if (epoch % config['train_params']['save_epoch'] == 0 and epoch != 0):
            saved_checkpoints[epoch] = dice_val

            liver_epoch_iterator_train_check = tqdm(
                liver_train_check_loader, desc="Train subset validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            lung_epoch_iterator_train_check = tqdm(
                lung_train_check_loader, desc="Train subset validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_train_check = validation(liver_epoch_iterator_train_check,lung_epoch_iterator_train_check)
            train_metric_values.append(dice_train_check)
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),  # 스케줄러 상태 추가
                    'epoch': epoch,
                    'best_dice': dice_val_best
                }, os.path.join(output_dir,f"checkpoint_epoch_{epoch}.pth"))
            logging.info(f"\nCheckpoint frequently saved at epoch {epoch}.")
             # 저장된 체크포인트들 중 top-3만 남기고 나머지 삭제
            prune_checkpoints(output_dir, saved_checkpoints, top_k=3)
        epoch += 1
        # flush 호출하여 로그 즉시 기록
        for handler in logging.getLogger().handlers:
            handler.flush()

    logging.info(f"train completed, best_metric: {dice_val_best:.4f} at epoch: {epoch_best}")
    np.savez(os.path.join(output_dir, "training_metrics.npz"), epoch_loss_values=epoch_loss_values, metric_values=metric_values, train_metric_values=train_metric_values)

# 최종 학습 결과를 WandB에 기록
    wandb.log({"best_metric": dice_val_best, "final_epoch": epoch_best})
    # checkpoint = torch.load(os.path.join(output_dir, "best_metric_model.pth"))
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    ### show the progress of training
    plt.figure("train", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("epoch Average Loss")
    x = [eval_epoch * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 3, 2)
    plt.title("Train subset Mean Dice")
    x = [config['train_params']['save_epoch'] * (i + 1) for i in range(len(train_metric_values))]
    y = train_metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 3, 3)
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