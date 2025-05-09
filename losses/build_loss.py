# losses/build.py
import torch
from monai.losses import DiceCELoss, DiceFocalLoss, TverskyLoss, GeneralizedDiceFocalLoss
from losses.balanced_contrastive import BalSCL  # 추후 해당 파일에 구현 필요
from losses.sup_contrastive import PixelContrastLoss
def build_loss(config):
    loss_type = config['train_params']['loss_type']
    contrastive_cfg = config.get('CONTRASTIVE', {})

    if loss_type == "DiceCELoss":
        assert not config['train_params'].get("use_contrastive_loss", False), (
            "config['train_params']['use_contrastive_loss'] must be False when using DiceCELoss"
        )
        return DiceCELoss(to_onehot_y=True, softmax=True), None

    elif loss_type == "DiceFocalLoss":
        assert not config['train_params'].get("use_contrastive_loss", False), (
            "config['train_params']['use_contrastive_loss'] must be False when using DiceCELoss"
        )
        return DiceFocalLoss(to_onehot_y=True, softmax=True, gamma=2.0, lambda_dice=1.0, lambda_focal=1.0), None

    elif loss_type == "TverskyLoss":
        assert not config['train_params'].get("use_contrastive_loss", False), (
            "config['train_params']['use_contrastive_loss'] must be False when using DiceCELoss"
        )
        return TverskyLoss(to_onehot_y=True, softmax=True, alpha=0.3, beta=0.7), None

    elif loss_type == "GeneralizedDiceFocalLoss":
        assert not config['train_params'].get("use_contrastive_loss", False), (
            "config['train_params']['use_contrastive_loss'] must be False when using DiceCELoss"
        )
        return GeneralizedDiceFocalLoss(to_onehot_y=True, softmax=True, gamma=1.5), None

    elif loss_type == "DiceCELoss+Cont1":
        # 2a) Cont1 ⇒ CONTRASTIVE MODE == 1
        assert config['CONTRASTIVE'].get("MODE") == 1, (
            "For DiceCELoss+Cont1, config['CONTRASTIVE']['MODE'] must be 1"
        )
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        pixel_contrast_loss_function = PixelContrastLoss(config)
        return loss_function, pixel_contrast_loss_function
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

    elif loss_type == "DiceCELoss+BALCont":
        # 2a) Cont1 ⇒ CONTRASTIVE MODE == 1
        assert config['CONTRASTIVE'].get("MODE") == 1, (
            "For DiceCELoss+Cont1, config['CONTRASTIVE']['MODE'] must be 1"
        )
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        pixel_contrast_loss_function = BalSCL([0.95, 0.05], contrastive_cfg['TEMPERATURE']).cuda()
        return loss_function, pixel_contrast_loss_function

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")