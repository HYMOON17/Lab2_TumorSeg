import os
import torch

from monai.transforms import (
    AddChanneld,
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    RandScaleIntensityd,
    Resize,
    Resized,
    EnsureTyped,
)


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

# 날짜 기반 하위 폴더 생성
def generate_experiment_name(config):
    model_type = config['model_params']['type']
    dataset_name = config['data']['dataset_name']
    learning_rate = config['train_params']['learning_rate']
    batch_size = config['train_params']['batch_size']
    loss_type = config['train_params']['loss_type']
    patch_size = "x".join(map(str, config['model_params']['img_size']))
    experiment_name = f"{model_type}-{dataset_name}-lr{learning_rate}-bs{batch_size}-loss{loss_type}-patch{patch_size}"
    return experiment_name

def validate_and_convert_config(config):
    for key, value in config.items():
        # 값이 dict 또는 list인 경우 string으로 변환
        if isinstance(value, (dict, list)):
            config[key] = str(value)
        # 그 외의 타입 중 int, float, str, bool, torch.Tensor가 아닌 경우 에러 출력
        elif not isinstance(value, (int, float, str, bool, torch.Tensor)):
            raise ValueError(f"Invalid type for config parameter '{key}': {type(value)}")
    return config




def lung_liver_transforms(config, device):
    liver_train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            # 이미지와 mask 로드 
            AddChanneld(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=config['transforms']['liver']['a_min'],
                a_max=config['transforms']['liver']['a_max'],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd를 사용하여 등방성 voxel spacing 1.0mm로 보간
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(config['transforms']['spacing']),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
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
        ]
    )

    liver_val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=config['transforms']['liver']['a_min'], a_max=config['transforms']['liver']['a_max'], b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(config['transforms']['spacing']),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )

    liver_test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=config['transforms']['liver']['a_min'], a_max=config['transforms']['liver']['a_max'], b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(config['transforms']['spacing']),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image"], device=device, track_meta=True),
        ]
    )

    lung_train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            # 이미지와 mask 로드 
            AddChanneld(keys=["image", "label"]),
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
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Resized(
            #     keys=["image", "label"],
            #     spatial_size=(340, 340, 340),
            #     mode=("trilinear", "nearest"),
            # ),

            # Spacingd를 사용하여 등방성 voxel spacing 1.0mm로 보간
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(config['transforms']['spacing']),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
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
        ]
    )

    lung_val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=config['transforms']['lung']['a_min'],
                a_max=config['transforms']['lung']['a_max'], b_min=0.0, b_max=1.0, clip=True
            ),
            # label 1을 label 0으로, label 2는 1로 변경 유지
            # MapLabelValued(keys="label", orig_labels=[0, 1, 2], target_labels=[0, 0, 1]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Resized(
            #     keys=["image", "label"],
            #     spatial_size=(340, 340, 340),
            #     mode=("trilinear", "nearest"),
            # ),
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(config['transforms']['spacing']),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )

    lung_test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=config['transforms']['lung']['a_min'],
                a_max=config['transforms']['lung']['a_max'], b_min=0.0, b_max=1.0, clip=True
            ),
            # label 1을 label 0으로, label 2는 1로 변경 유지
            # MapLabelValued(keys="label", orig_labels=[0, 1, 2], target_labels=[0, 0, 1]),
            CropForegroundd(keys=["image"], source_key="image"),
            Orientationd(keys=["image"], axcodes="RAS"),
            # Resized(
            #     keys=["image"],
            #     spatial_size=(340, 340, 340),
            #     mode=("trilinear"),
            # ),
            Spacingd(
                keys=["image", "label"],
                pixdim=tuple(config['transforms']['spacing']),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image"], device=device, track_meta=True),
        ]
    )
    return lung_train_transforms, lung_val_transforms, lung_test_transforms, liver_train_transforms, liver_val_transforms, liver_test_transforms
