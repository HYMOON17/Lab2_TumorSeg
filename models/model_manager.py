import torch
from typing import Dict
import logging
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger()
###
# detr등 다른 모델들도 추가해서 확인할 수 있으면 함
###

def build_model(config: Dict, device: torch.device):
    model_type = config["model_params"]["type"]

    if model_type == "SwinUnetr":
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            img_size=config['model_params']['img_size'],
            in_channels=config['model_params']['in_channels'],
            out_channels=config['model_params']['out_channels'],
            feature_size=config['model_params']['feature_size'],
            use_checkpoint=config['model_params']['use_checkpoint']
        ).to(device)

    elif model_type == "ContrastiveSwinUNETR":
        from .swin_unetr_cont import ContrastiveSwinUNETR
        model = ContrastiveSwinUNETR(
            img_size=config['model_params']['img_size'],
            in_channels=config['model_params']['in_channels'],
            out_channels=config['model_params']['out_channels'],
            feature_size=config['model_params']['feature_size'],
            use_checkpoint=config['model_params']['use_checkpoint']
        ).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model

def load_model_weights(model, weight_path: str, device: torch.device):
    model.load_state_dict(torch.load(weight_path, map_location=device)['model_state_dict'], strict=False)
    # model.load_from(weights=weight)
    logger.info(f"Model weights loaded from: {weight_path}")  # 모델 로드 경로 출력
    logger.info("Loading checkpoint... Well")

def load_model_pretrained_weights(model, config,device: torch.device):
    weight = torch.load(config['data']['weights_path'])
    model.load_from(weights=weight)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)