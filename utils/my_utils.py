import numpy as np
import os
import logging
import json
import platform
import torch
import monai
import numpy
from tqdm import tqdm
import traceback
from typing import Dict, List, Tuple, Any
# 학습 재개를 위한 체크포인트 저장 함수
def acc_save_checkpoint(accelerator, model, optimizer, global_step, dice_val_best, global_step_best, epoch_loss_values, metric_values, output_dir):
    if accelerator.is_main_process:
        # 모델 및 옵티마이저 상태는 accelerator를 사용해 저장
        accelerator.save_state(output_dir)
        
        # 기타 변수들은 npz 파일로 저장
        np.savez(
            os.path.join(output_dir, "additional_state.npz"),
            global_step=global_step,
            dice_val_best=dice_val_best,
            global_step_best=global_step_best,
            epoch_loss_values=np.array(epoch_loss_values),
            metric_values=np.array(metric_values)
        )
        print(f"Checkpoint saved at step {global_step}")

# 체크포인트 불러오기 함수
def acc_load_checkpoint(accelerator, model, optimizer, output_dir):
    # 모델 및 옵티마이저 상태는 accelerator를 사용해 로드
    accelerator.load_state(output_dir)
    
    # 기타 변수들은 npz 파일로 로드
    additional_state = np.load(os.path.join(output_dir, "additional_state.npz"))
    global_step = int(additional_state["global_step"])
    dice_val_best = float(additional_state["dice_val_best"])
    global_step_best = int(additional_state["global_step_best"])
    epoch_loss_values = additional_state["epoch_loss_values"].tolist()
    metric_values = additional_state["metric_values"].tolist()
    
    print(f"Checkpoint loaded from step {global_step}")
    return global_step, dice_val_best, global_step_best, epoch_loss_values, metric_values

# # TqdmLoggingHandler 정의
# class TqdmLoggingHandler(logging.Handler):
#     def __init__(self, level=logging.NOTSET):
#         super().__init__(level)

#     def emit(self, record):
#         try:
#             msg = self.format(record)
#             tqdm.write(msg)
#             self.flush()
#         except Exception:
#             self.handleError(record)


# def setup_logging(log_dir):
#     # 로그 포맷 설정
#     log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
#     # 로그 파일 저장 경로를 log_dir로 지정
#     log_file_path = os.path.join(log_dir, "training.log")
    
#     # Root logger 가져오기
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)

#     # 기존 핸들러 제거 (중복 방지)
#     if logger.hasHandlers():
#         logger.handlers.clear()

#     # 파일 핸들러 추가
#     file_handler = logging.FileHandler(log_file_path)
#     file_handler.setLevel(logging.INFO)
#     file_handler.setFormatter(logging.Formatter(log_format))

#     # tqdm 핸들러 추가
#     tqdm_handler = TqdmLoggingHandler()
#     tqdm_handler.setLevel(logging.INFO)
#     tqdm_handler.setFormatter(logging.Formatter(log_format))

#     # 로거에 핸들러 추가
#     logger.addHandler(file_handler)
#     logger.addHandler(tqdm_handler)

#     logging.info("Logging is set up. Log file: %s", log_file_path)
    


def sample_pixels_randomly(X, y, num_sampled_pixels=8192):
    """
    BCL 원본 방식처럼 배치 크기만 유지하면서 픽셀을 무작위로 샘플링하는 함수
    - X: [Batch, C, D, H, W]  (Feature Map)
    - y: [Batch, D, H, W]  (Label Map)
    - num_sampled_pixels: 총 샘플링할 픽셀 수
    """
    device = X.device
    batch_size, num_channels, D, H, W = X.shape  # 입력 feature map 크기

    # 전체 픽셀 위치 가져오기
    all_pixels = (y >= 0).nonzero(as_tuple=True)  # 모든 픽셀 인덱스 가져오기

    # 랜덤하게 num_sampled_pixels 개수만큼 샘플링
    sampled_indices = torch.randint(0, len(all_pixels[0]), (num_sampled_pixels,))
    sampled_indices = [idx[sampled_indices] for idx in all_pixels]

    # 샘플링된 위치에서 feature 가져오기
    X_samples = X[sampled_indices[0], :, sampled_indices[1], sampled_indices[2], sampled_indices[3]]
    y_samples = y[sampled_indices[0], sampled_indices[1], sampled_indices[2], sampled_indices[3]]

    return X_samples, y_samples



from typing import Dict, List, Union
import yaml


### 1. Config & Seed
# def load_config(config_path: str) -> Dict:
#     with open(config_path, 'r') as file:
#         return yaml.safe_load(file)

def safe_eval(val: str):
    lowered = val.lower()
    if lowered == "true":
        return True
    elif lowered == "false":
        return False

    try:
        return eval(val, {"__builtins__": None}, {})
    except:
        return val

def apply_overrides(config: dict, overrides: List[str]) -> dict:
    """
    override = ["train_params.batch_size=4", "model_params.img_size=[128,128,128]"]
    """
    for item in overrides:
        if '=' not in item:
            raise ValueError(f"Invalid override format: {item}")
        key_str, value_str = item.split("=", 1)
        keys = key_str.split(".")

        d = config
        for k in keys[:-1]:
            if k not in d:
                raise KeyError(f"Key '{k}' not found in config")
            d = d[k]

        d[keys[-1]] = safe_eval(value_str)
    return config

def load_config(path: str, overrides: List[str] = None) -> dict:
    """
    - path: path to YAML config
    - overrides: optional list of "key=value" strings
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if overrides:
        config = apply_overrides(config, overrides)

    return config

import socket

def get_primary_ip():
    """외부 라우팅을 기준으로 서버가 사용하는 IP"""
    try:
        # 임의의 외부 주소와 소켓 연결을 시도해 서버의 실제 IP 추론
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # 실제로 연결되진 않음
        ip = s.getsockname()[0]
        # ip는 115.145.145.167 이런형태
        s.close()
        return ip[-3:]
    except Exception:
        return "000"

def resolve_data_path(config: dict,path: str) -> str:
    if "<data_root>" in path:
        return path.replace("<data_root>", config(f"{get_primary_ip()}.data_root"))
    return path


import glob

def prune_checkpoints(output_dir, saved_checkpoints, top_k: int = 3):
    """output_dir 내 checkpoint_epoch_*.pth 중
       saved_checkpoints에 기록된 top_k epoch만 남기고 나머지 삭제."""
    pattern = os.path.join(output_dir, "checkpoint_epoch_*.pth")
    files = glob.glob(pattern)
    # 삭제 대상이 10개 이하이면 무시
    if len(files) <= 5:
        return
    # 파일명에서 epoch 번호 추출
    epochs = []
    for f in files:
        try:
            epochs.append(int(os.path.basename(f).split("_")[-1].split(".pth")[0]))
        except ValueError:
            continue
    # (epoch, metric, filepath) 리스트
    trio = [(e, saved_checkpoints.get(e, -1.0), os.path.join(output_dir, f"checkpoint_epoch_{e}.pth"))
            for e in epochs]
    # Dice 내림차순 상위 K개 선정
    keep_files = {
        filepath
        for _, _, filepath in sorted(trio, key=lambda x: x[1], reverse=True)[:top_k]
    }
    # 그 외 파일 삭제 시도 (실패해도 경고만)
    for _, _, filepath in trio:
        if filepath not in keep_files:
            try:
                os.remove(filepath)
                # logging.info(f"Pruned checkpoint {os.path.basename(filepath)}")
            except Exception as e:
                logging.warning(f"Failed to remove {filepath}: {e}")

from monai.transforms import RandomizableTransform, MapTransform
from monai.config import KeysCollection
from monai.data.meta_obj import get_track_meta
from monai.utils.type_conversion import convert_to_tensor
import numpy as np
from typing import Optional, Mapping, Hashable
from monai.data.meta_tensor import MetaTensor
from monai.transforms.transform import Randomizable

class RandInvertIntensityd(RandomizableTransform, MapTransform):
    """
    Dictionary-based random intensity inversion.

    Invert image intensity as: new_val = 1.0 - old_val
    Only applies to values assumed normalized in [0, 1].

    Args:
        keys: input keys to apply the transform.
        prob: probability of applying the transform.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(self, keys: KeysCollection, prob: float = 0.1, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ):
        super().set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        self.randomize(None)
        for key in self.key_iterator(d):
            # Always convert to tensor to maintain consistency, regardless of _do_transform
            d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            if self._do_transform:
                d[key] = 1.0 - d[key]
        return d


if __name__ == "__main__":
    import argparse
    import pprint
    parser = argparse.ArgumentParser(description="Debug config loading")

    parser.add_argument(
        '--config', type=str,
        default="your_config.yaml",   # 실제 config 경로로 바꿔주세요
        help="Path to the YAML config file"
    )

    parser.add_argument(
        '--override', nargs='*', default=[],
        help="Override config parameters, e.g., train_params.batch_size=4"
    )

    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)

    print("\n🔧 Final Config (with overrides if given):")
    pprint.pprint(config)
