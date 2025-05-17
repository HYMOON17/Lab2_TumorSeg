### further detail
# for mem 4090 using train dataset cahce 12, 0.5 will demand 20G
# np 1.21.6 torch1.13.0+cu117
#### Environment
#### make sure use swin_unetr conda env
# conda activate swin_unetr

####
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"]= "2"
import shutil
import yaml
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np
import logging
import wandb
from monai.transforms import AsDiscrete
from monai.data import (
    decollate_batch,
    set_track_meta,
)
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.config import print_config
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config_manager import EnvConfigManager
cfg_mgr = EnvConfigManager()
cfg_mgr.prepend_root_to_sys_path()
ROOT_DIR = cfg_mgr.project_root()
server_id = cfg_mgr.server_id
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.my_utils import load_config, prune_checkpoints
from utils.seed import set_seed_and_env
from utils.logger import generate_experiment_name, setup_logging, log_experiment_config, save_config_as_json, get_logger,save_current_code,save_current_code_wandb
from transforms import get_test_transforms, postprocess
from loader import load_datalist
from models.model_manager import build_model,load_model_pretrained_weights
from losses.build_loss import build_loss
def main():
    
    parser = argparse.ArgumentParser(description="Swin UNETR training")
    parser.add_argument('--config', type=str, default=str(ROOT_DIR/"config/exp.yaml"), help="Path to the YAML config file")
    parser.add_argument('--override', nargs='*', default=[], help="Override config parameters, e.g., train_params.batch_size=4")
    args = parser.parse_args()

    # 학습 설정 로드
    config_path = args.config
    config = load_config(config_path, overrides=args.override)
    config = cfg_mgr.resolve_config(config)  # 실제 경로로 치환 완료
    # 학습환경 설정
    set_seed_and_env(config)
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
    logger = get_logger()
    # config 파일을 JSON으로 저장
    log_experiment_config()
    save_config_as_json(config, log_dir)
    # 학습 시작 메시지
    logger.info("Training started with configuration:")
    logger.info(config)

    wandb.init(project=config['wandb']['project_name'],name=config['wandb']['experiment_name'],
    config=config,dir=log_dir, mode=config['wandb']['mode'])
    
    # 기록하고 싶은 파일 리스트
    # Train, Trainer, model, loss 정도?
    # Model과 Trainer의 소스 파일 경로 가져오기
    
    current_script_name = __file__
    # save_current_code(log_dir,current_script_name)
    save_current_code_wandb(log_dir,current_script_name, log_to_wandb=True)
    
    print_config()
    
    # ### Transforms
    # 2024.09.25 현재 label은 종양만 해서 한가지만 남김
    # 2024.09.26 현재 patch 96 유지 못해서 일단 64로 진행
    
    tf_dict = get_test_transforms(config, organs=["liver","lung"], is_train=True)

    # ### Dataset
    set_track_meta(True)
    # iter아닌 경우 mode에 base박으셈
    dataloader = load_datalist(config, transform_dict= tf_dict, output_dir=output_dir,is_train=True,mode=None,server_id=server_id)
    

    #### Model
    ##### Create
    model = build_model(config,device)
    # ### Load weights
    load_model_pretrained_weights(model, config,device)
    



    # ### Training

    max_epoch = config['train_params']['max_epoch']
    eval_epoch = config['train_params']['eval_epoch']
    seg_loss_fn, cont_loss_fn = build_loss(config)
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
    start_epoch = epoch  # resume 시: checkpoint['epoch'], fresh 시: 0
    if config['train_params']['resume'] :
        logger.info("Loading checkpoint...")
        checkpoint = torch.load(config['train_params']['resume_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 스케줄러 상태 복구
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        epoch = checkpoint['epoch']
        start_epoch = checkpoint['epoch']
        dice_val_best = checkpoint['best_dice']
        logger.info(f"Resuming from epoch {epoch} with best dice {dice_val_best:.4f}")

    

    def validation(epoch_iterator_val_liver, epoch_iterator_val_lung):
        model.eval()
        dice_metric.reset()  # Reset metric at the start of validation
        with torch.no_grad():
            for organ_type, epoch_iterator_val in zip(["liver", "lung"], [epoch_iterator_val_liver, epoch_iterator_val_lung]):
                temp_dice_metric.reset()
                for step, batch in enumerate(epoch_iterator_val):
                    val_inputs = batch["image"].cuda()
                    with torch.cuda.amp.autocast():
                        batch["pred"] = sliding_window_inference(val_inputs, config['model_params']['img_size'], config['transforms']['num_samples'], model).detach().cpu()
                    if organ_type == "liver":
                        val_output_convert, val_labels_convert = postprocess(batch, tf_dict["liver"]["post_tf"])
                    else:
                        val_output_convert, val_labels_convert = postprocess(batch, tf_dict["lung"]["post_tf"])
                        
                    dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                    temp_dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                    epoch_iterator_val.set_description(
                        f"Validate {organ_type} ({step + 1} / {len(epoch_iterator_val)} Steps)"
                    )
                 # organ별 aggregate 후 출력 (임시 metric은 organ마다 새로 생성했으므로 reset 불필요)
                organ_dice = temp_dice_metric.aggregate().item()
                logger.info(f"{organ_type.capitalize()} Mean Dice: {organ_dice:.4f}")
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
                if config['train_params']['use_contrastive_loss'] and cont_loss_fn is not None:
                    # Training Dice score calculation
                    logit_map,repre = model(x)
                    
                    # y_pred = [post_pred(i) for i in y_pred]
                    y_pred = decollate_batch(logit_map)
                    y_pred = [post_pred(i) for i in y_pred]
                    y_pred = torch.stack(y_pred,dim=0)
                    loss = seg_loss_fn(logit_map, y)
                    contrast_loss = cont_loss_fn(repre.unsqueeze(1), y, y_pred)
                    # DiceCELoss + ContrastiveLoss 가중합
                    loss = loss + config['train_params']['lambda_contrast'] * contrast_loss
                    epoch_contrastive_loss += contrast_loss.item()
                else:
                    # ContrastiveLoss를 사용하지 않음
                    logit_map = model(x)
                    loss = seg_loss_fn(logit_map, y)
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
        train_dice =0.0 # 에러방지용
        #혹 아래 문제시, step이 10이 안된다거나, y_pred,y가 잘못된 상황
        if len(dice_metric_train.get_buffer()) > 0:
            train_dice = dice_metric_train.aggregate().item()
        dice_metric_train.reset()
        return epoch_loss ,epoch_contrastive_loss , train_dice
    
    # 에포크별 저장한 Dice 값을 기록하기 위한 dict
    saved_checkpoints: dict[int, float] = {}
    while epoch < max_epoch:
        epoch_loss,epoch_contrastive_loss, train_dice = train(epoch, dataloader["liver"]["train_loader"], dataloader["lung"]["train_loader"])
        logger.info(f"Epoch {epoch}, Training loss {epoch_loss:.4f}, Cont loss {epoch_contrastive_loss : .4f}, Training Dice {train_dice:.4f}")
        wandb.log({"lr": scheduler.get_last_lr()[0], "training_loss": epoch_loss, "Cont_loss":epoch_contrastive_loss, "training_dice": train_dice},step=epoch)
        scheduler.step()
        # WandB로 학습 손실 기록
        epoch_loss_values.append(epoch_loss)
        epoch_contrastive_loss_values.append(epoch_contrastive_loss)
        if (epoch % eval_epoch == 0 and epoch != 0) or epoch == max_epoch:
            liver_epoch_iterator_val = tqdm(
                dataloader["liver"]["val_loader"], desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            lung_epoch_iterator_val = tqdm(
                dataloader["lung"]["val_loader"], desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
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
                logger.info(f"\nModel Was Saved ! Current {epoch}epoch is Best Avg. Dice: {dice_val_best}")
            else:
                logger.info(f"\nModel Was Not Saved ! Current Best Avg Dice at {epoch_best}epoch: {dice_val_best} Current {epoch}epoch Avg. Dice: {dice_val}")
        if (epoch % config['train_params']['save_epoch'] == 0 and epoch != 0):
            saved_checkpoints[epoch] = dice_val

            liver_epoch_iterator_train_check = tqdm(
                dataloader["liver"]["train_check_loader"], desc="Train subset validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            lung_epoch_iterator_train_check = tqdm(
                dataloader["lung"]["train_check_loader"], desc="Train subset validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
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
            logger.info(f"\nCheckpoint frequently saved at epoch {epoch}.")
             # 저장된 체크포인트들 중 top-3만 남기고 나머지 삭제
            prune_checkpoints(output_dir, saved_checkpoints, top_k=3)
        epoch += 1
        # flush 호출하여 로그 즉시 기록
        for handler in logging.getLogger().handlers:
            handler.flush()
        np.savez(os.path.join(output_dir, "training_metrics_temp.npz"),start_epoch=start_epoch,last_epoch=epoch,epoch_loss_values=epoch_loss_values,epoch_contrastive_loss_values=epoch_contrastive_loss_values, metric_values=metric_values, train_metric_values=train_metric_values)

    logger.info(f"train completed, best_metric: {dice_val_best:.4f} at epoch: {epoch_best}")
    np.savez(os.path.join(output_dir, "training_metrics_temp.npz"),start_epoch=start_epoch,last_epoch=epoch,epoch_loss_values=epoch_loss_values,epoch_contrastive_loss_values=epoch_contrastive_loss_values, metric_values=metric_values, train_metric_values=train_metric_values)

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


'''
python /data/hyungseok/Swin-UNETR/code/train.py \
  --override test_params.ckpt_dir="/data/hyungseok/Swin-UNETR/Experiments/Models/SwinUnetr-Post_Lung-lr1e-4-bs3-lossDiceCELoss-patch64x64x64/2025-04-09_21-03/best_metric_model.pth"
'''
if __name__ == "__main__": 
    main()