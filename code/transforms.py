import os
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, CropForegroundd, Orientationd,
                              FgBgToIndicesd,RandCropByPosNegLabeld,RandFlipd,RandRotate90d,RandScaleIntensityd,RandShiftIntensityd,
                              Spacingd, EnsureTyped, AsDiscrete,AsDiscreted,ToTensord,MapLabelValued, Invertd)
from typing import Dict, List, Union
from utils.seed import set_all_random_states
from monai.data import decollate_batch
from monai.handlers.utils import from_engine

### 3. Transform & Dataset
def get_test_transforms(config: Dict, organs: Union[str, List[str]],is_train: bool):
    """
    organ == "liver" / "lung" 에 따라 spacing 및 intensity range 조정
    2025-05-07 organ 입력받으면 그거에 따라 반환토록함
    """
    if isinstance(organs, str):
        organs = [organs]
    transforms = {}

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

    for organ in organs:

        if is_train:
            try:
                first_tf  = locals()[f"{organ}_train_transforms"]
                second_tf = locals()[f"{organ}_test_transforms"]
                post_tf  = locals()[f"{organ}_post_pred"]
            except KeyError as e:
                raise ValueError(f"전처리 정의가 누락된 organ: '{organ}' → {e}")
        
        else:
            try:
                first_tf  = locals()[f"{organ}_test_transforms"]
                second_tf = locals()[f"{organ}_label_transforms"]
                post_tf  = locals()[f"{organ}_post_pred"]
            except KeyError as e:
                raise ValueError(f"전처리 정의가 누락된 organ: '{organ}' → {e}")
            
        transforms[organ] = {
            "first_tf": first_tf,
            "second_tf": second_tf,
            "post_tf": post_tf,
        }
    for organ in transforms:
        for key in ["first_tf", "second_tf"]:
            if transforms[organ][key] is not None:
                set_all_random_states(transforms[organ][key].transforms, seed=1234)


    return transforms


def postprocess(batch, post_pred_transform):
    """
    Invertd + AsDiscreted 적용

    추가 확인해야하는 부분들은 .cpu가 sliding에서 작업하는게 나을지 여기단에서 할지등임
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