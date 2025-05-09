import os
import json
from typing import Dict,Tuple, List
from torch.utils.data import ConcatDataset
from monai.data import (PersistentDataset,Dataset,CacheDataset,
    ThreadDataLoader,DataLoader,DistributedSampler,
    load_decathlon_datalist
)
from utils.seed import worker_init_fn

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

def load_txt_paths(txt_path: str, image_dir: str, label_replacement: str) -> Tuple[List[str], List[str]]:
    image_paths, label_paths = [], []
    with open(txt_path, 'r') as f:
        for line in f:
            image_path = os.path.join(image_dir, line.strip())
            label_path = image_path.replace("imagesTr", label_replacement)
            image_paths.append(image_path)
            label_paths.append(label_path)
    return image_paths, label_paths

def construct_test_json(images: List[str], labels: List[str]) -> dict:
    return {
        "labels": {"0": "background", "1": "cancer"},
        "tensorImageSize": "3D",
        "test": [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]
    }

def construct_train_json(train_images: List[str], train_labels: List[str], val_images: List[str], val_labels: List[str]) -> dict:
    return {
        "labels": {"0": "background", "1": "cancer"},
        "tensorImageSize": "3D",
        "training": [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)],
        "validation": [{"image": img, "label": lbl} for img, lbl in zip(val_images, val_labels)]
    }

def construct_json(mode: str, images: List[str], labels: List[str],
                   val_images: List[str] = None, val_labels: List[str] = None) -> Dict:
    if mode == "train":
        return {
            "labels": {"0": "background", "1": "cancer"},
            "tensorImageSize": "3D",
            "training": [{"image": i, "label": l} for i, l in zip(images, labels)],
            "validation": [{"image": i, "label": l} for i, l in zip(val_images, val_labels)]
        }
    else:
        return {
            "labels": {"0": "background", "1": "cancer"},
            "tensorImageSize": "3D",
            "test": [{"image": i, "label": l} for i, l in zip(images, labels)]
        }

def select_dataset_class(config: Dict) -> type:
    # 후에 확장해서 아예 Dataset자체를 반환토록
    if config['cuda']['mode'] == 'server':
        return PersistentDataset
    elif config['cuda']['mode'] == 'local':
        return CacheDataset
    else:
        return Dataset
    

def load_datalist(config: Dict, transform_dict : Dict, output_dir: str, is_train:bool,mode:str=None) -> Dict[str, Dict[str, DataLoader]]:
    '''
    2025-05-07
    장기에 따라 각각 진행토록 바꾸고 싶은데, 그전까지만 진행
    '''
    dataloaders = {}
    dataset_class = select_dataset_class(config)
    # cache 없을경우 문제발생

    # 이미지 및 레이블 경로가 저장된 txt 파일을 불러오기 (과거)
    # Train 데이터 로드
    # liver_test_images, liver_test_labels = load_liverfile_from_txt(config['data']['liver_test_image_txt'], config['data']['liver_image_dir'])
    # lung_test_images, lung_test_labels = load_lungfile_from_txt(config['data']['lung_test_image_txt'], config['data']['lung_image_dir'])
    
    if is_train:
        # Train 데이터 로드
        lung_train_images, lung_train_labels = load_txt_paths(config['data']['lung_train_image_txt'],
                                                               config['data']['lung_image_dir'],
                                                               'labelsTr')
        liver_train_images, liver_train_labels = load_txt_paths(config['data']['liver_train_image_txt'],
                                                                 config['data']['liver_image_dir'],
                                                                 'processed_labels')
        # Validation 데이터 로드
        lung_val_images, lung_val_labels = load_txt_paths(config['data']['lung_val_image_txt'], 
                                                           config['data']['lung_image_dir'],
                                                           'labelsTr')
        liver_val_images, liver_val_labels = load_txt_paths(config['data']['liver_val_image_txt'],
                                                              config['data']['liver_image_dir'],
                                                              'processed_labels')
        
        # liver_json = construct_train_json(liver_train_images, liver_train_labels, liver_val_images, liver_val_labels)
        # lung_json = construct_train_json(lung_train_images, lung_train_labels, lung_val_images, lung_val_labels)
        
        liver_json = construct_json("train", liver_train_images, liver_train_labels, liver_val_images, liver_val_labels)
        lung_json = construct_json("train", lung_train_images, lung_train_labels, lung_val_images, lung_val_labels)
        
        with open(os.path.join(output_dir, 'liver_train_dataset.json'), 'w') as f:
            json.dump(liver_json, f)
        with open(os.path.join(output_dir, 'lung_train_dataset.json'), 'w') as f:
            json.dump(lung_json, f)

        # #### For Debug run below - check train is well
        lung_debug_datasets = "/data/hyungseok/Swin-UNETR/results/debug/lung_dataset_train_check.json"
        liver_debug_datasets = "/data/hyungseok/Swin-UNETR/results/debug/liver_dataset_train_check.json"

        liver_train_files = load_decathlon_datalist(liver_json, True, "training")
        liver_val_files = load_decathlon_datalist(liver_json, True, "validation")
        lung_train_files = load_decathlon_datalist(lung_json, True, "training")
        lung_val_files = load_decathlon_datalist(lung_json, True, "validation")
        liver_train_debug_files = load_decathlon_datalist(liver_debug_datasets, True, "training")
        lung_train_debug_files = load_decathlon_datalist(lung_debug_datasets, True, "training")

        liver_train_tf, liver_val_tf = transform_dict["liver"]["first_tf"], transform_dict["liver"]["second_tf"]
        lung_train_tf, lung_val_tf = transform_dict["lung"]["first_tf"], transform_dict["lung"]["second_tf"]

        liver_train_ds = dataset_class(data=liver_train_files, transform=liver_train_tf, cache_dir=config['data']['liver_train_cache_dir'])
        lung_train_ds = dataset_class(data=lung_train_files, transform=lung_train_tf, cache_dir=config['data']['lung_train_cache_dir'])
        liver_val_ds = dataset_class(data=liver_val_files, transform=liver_val_tf, cache_dir=config['data']['liver_val_cache_dir'])
        lung_val_ds = dataset_class(data=lung_val_files, transform=lung_val_tf, cache_dir=config['data']['lung_val_cache_dir'])
        liver_train_check_ds = PersistentDataset(data=liver_train_debug_files, transform=liver_val_tf,cache_dir=config['data']['liver_debug_cache_dir'])
        lung_train_check_ds = PersistentDataset(data=lung_train_debug_files, transform=lung_val_tf,cache_dir=config['data']['lung_debug_cache_dir'])

        liver_val_loader = DataLoader(liver_val_ds, batch_size=config['train_params']['val_batch_size'],
                                      num_workers=config['train_params']['val_num_workers'], pin_memory=True,
                                      worker_init_fn=worker_init_fn)
        lung_val_loader = DataLoader(lung_val_ds, batch_size=config['train_params']['val_batch_size'],
                                     num_workers=config['train_params']['val_num_workers'], pin_memory=True,
                                     worker_init_fn=worker_init_fn)
        
        liver_train_check_loader = DataLoader(liver_train_check_ds, batch_size=config['train_params']['val_batch_size'], 
        num_workers=config['train_params']['val_num_workers'], pin_memory=config['train_params']['pin_memory'],
        worker_init_fn=worker_init_fn)
        lung_train_check_loader = DataLoader(lung_train_check_ds, batch_size=config['train_params']['val_batch_size'], 
            num_workers=config['train_params']['val_num_workers'], pin_memory=config['train_params']['pin_memory'],
            worker_init_fn=worker_init_fn)

        if mode == "base":
            combined_train_dataset = ConcatDataset([liver_train_ds, lung_train_ds])
            train_loader = DataLoader(combined_train_dataset, batch_size=config['train_params']['batch_size'],
                                  num_workers=config['train_params']['num_workers'], pin_memory=True,
                                  shuffle=True, worker_init_fn=worker_init_fn)
            
            dataloaders["liver"] = {"train_loader": train_loader, "val_loader": liver_val_loader, "train_check_loader": liver_train_check_loader}
            dataloaders["lung"] = {"train_loader": train_loader, "val_loader": lung_val_loader, "train_check_loader": lung_train_check_loader}
        else:
            liver_train_loader = DataLoader(liver_train_ds, batch_size=config['train_params']['batch_size'], 
                                num_workers=config['train_params']['num_workers'], pin_memory=config['train_params']['pin_memory'],
                                shuffle= True, worker_init_fn=worker_init_fn)
            lung_train_loader = DataLoader(lung_train_ds, batch_size=config['train_params']['batch_size'], 
                                num_workers=config['train_params']['num_workers'], pin_memory=config['train_params']['pin_memory'],
                                shuffle= True,worker_init_fn=worker_init_fn)
            dataloaders["liver"] = {"train_loader": liver_train_loader, "val_loader": liver_val_loader, "train_check_loader": liver_train_check_loader}
            dataloaders["lung"] = {"train_loader": lung_train_loader, "val_loader": lung_val_loader, "train_check_loader": lung_train_check_loader}


    else:
        # test
        liver_imgs, liver_lbls = load_txt_paths(
            config['data']['liver_test_image_txt'],
            config['data']['liver_image_dir'],
            'processed_labels'
        )
        lung_imgs, lung_lbls = load_txt_paths(
            config['data']['lung_test_image_txt'],
            config['data']['lung_image_dir'],
            'labelsTr'
        )

        liver_json = construct_test_json(liver_imgs, liver_lbls)
        lung_json = construct_test_json(lung_imgs, lung_lbls)

    
        # JSON 파일 저장
        with open(os.path.join(output_dir, f'liver_{mode}_dataset.json'), 'w') as outfile:
            json.dump(liver_json, outfile)
        with open(os.path.join(output_dir, f'lung_{mode}_dataset.json'), 'w') as outfile:
            json.dump(lung_json, outfile)
        
        liver_test_files = load_decathlon_datalist(liver_json, True, "test")
        lung_test_files = load_decathlon_datalist(lung_json, True, "test")


        liver_test_transforms, liver_label_transforms = transform_dict["liver"]["first_tf"], transform_dict["liver"]["second_tf"]
        lung_test_transforms, lung_label_transforms = transform_dict["lung"]["first_tf"], transform_dict["lung"]["second_tf"]

        liver_test_ds = Dataset(data=liver_test_files, transform=liver_test_transforms)
        lung_test_ds = Dataset(data=lung_test_files, transform=lung_test_transforms)
        #Overlay용
        liver_label_test_ds = Dataset(data=liver_test_files, transform=liver_label_transforms)
        lung_label_test_ds = Dataset(data=lung_test_files, transform=lung_label_transforms)

        
        liver_test_loader = DataLoader(liver_test_ds, num_workers=8, batch_size=1,worker_init_fn=worker_init_fn)
        lung_test_loader = DataLoader(lung_test_ds, num_workers=8, batch_size=1,worker_init_fn=worker_init_fn)
        liver_label_test_loader = DataLoader(liver_label_test_ds, num_workers=8, batch_size=1,worker_init_fn=worker_init_fn)
        lung_label_test_loader = DataLoader(lung_label_test_ds, num_workers=8, batch_size=1,worker_init_fn=worker_init_fn)
        

        dataloaders["liver"] = {
            "test_loader": liver_test_loader,
            "label_test_loader": liver_label_test_loader,
        }
        dataloaders["lung"] = {
            "test_loader": lung_test_loader,
            "label_test_loader": lung_label_test_loader,
        }
    return dataloaders
