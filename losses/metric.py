'''
Class로
organ의 여러 metric 상태 유지
batch-wise 처리 - 환자별 처리
organ별로 분리해서 기록 - organ을 key로 한 dict로 관리
MONAI Metric 연동 - 내부에 유지하며 organ 변경시, 자동 reset 가능
확장성 - 추후 클래스에 method추가
'''
import numpy as np
from scipy import ndimage
from utils.logger import get_logger
from monai.metrics import DiceMetric
from monai.metrics import HausdorffDistanceMetric, compute_average_surface_distance
import pandas as pd
import os
class MetricManager:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.organs = ["liver", "lung"]
        self.records = {organ: [] for organ in self.organs}
        self.logger = get_logger()

        # Monai metrics (선택적으로 사용)
        self.dice_metric_all = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.dice_metric_tumor = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

        # self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        
        self.compute_asd = compute_average_surface_distance

    def update(self, organ: str, case_name: str, pred, gt, meta_info: dict):
        """
        한 케이스(환자)에 대한 예측 결과를 기록.
        meta_info: dict 형태 (예: 위치, 컴포넌트 수 등)
        """
        self.dice_metric_all.reset()
        self.dice_metric_tumor.reset()

        self.dice_metric_all(y_pred=pred, y=gt)
        self.dice_metric_tumor(y_pred=pred, y=gt)

        dice_all = self.dice_metric_all.aggregate().item()
        dice_tumor = self.dice_metric_tumor.aggregate().item()

        # dice_score = self.dice_metric.aggregate().item()
        # Convert to binary numpy
        pred_np = pred[0, 1].cpu().numpy()
        gt_np = gt[0, 1].cpu().numpy()

        # Additional metrics
        asd_score = self.compute_asd(y_pred=pred, y=gt, include_background=False).item()
        comp_stats = self._component_wise_metric(pred_np, gt_np)
        pixel_stats = self._pixel_wise_metric(pred_np, gt_np)

        
        # 저장 구조: case 단위 dict
        case_result = {
            "case_name": case_name,
            "dice_all": dice_all,
            "dice_tumor": dice_tumor,
            "asd": asd_score,
            "component_recall": comp_stats["recall_obj"],
            **comp_stats,
            **pixel_stats,
            **meta_info
        }

        self.records[organ].append(case_result)
        # ✅ 로그 출력
        self.logger.info(
            f"[{organ}] Case '{case_name}' — "
            f"Dice(All): {dice_all:.4f}, Dice(Tumor): {dice_tumor:.4f}, ASD: {asd_score:.2f}"
        )

    def _component_check(self, mask):       
        return ndimage.label(mask)

    def _component_wise_metric(self, pred, gt, voxel_threshold=8):
        gt_lab, gt_num = self._component_check(gt)
        pred_lab, pred_num = self._component_check(pred)

        tp_obj, fn_obj, fp_obj = 0, 0, 0
        for cid in range(1, gt_num + 1):
            mask_gt = gt_lab == cid
            if mask_gt.sum() < voxel_threshold:
                continue
            if np.any(pred_lab[mask_gt] > 0):
                tp_obj += 1
            else:
                fn_obj += 1
        for pid in range(1, pred_num + 1):
            mask_pred = pred_lab == pid
            if mask_pred.sum() < voxel_threshold:
                continue
            if not np.any(gt_lab[mask_pred] > 0):
                fp_obj += 1

        recall = tp_obj / (tp_obj + fn_obj) if (tp_obj + fn_obj) > 0 else 0.0
        precision = tp_obj / (tp_obj + fp_obj) if (tp_obj + fp_obj) > 0 else 0.0

        return {
            "tp_obj": tp_obj,
            "fn_obj": fn_obj,
            "fp_obj": fp_obj,
            "recall_obj": recall,
            "precision_obj": precision,
        }
    
    def _pixel_wise_metric(self, pred, gt):
        tp = np.logical_and(pred == 1, gt == 1).sum()
        fn = np.logical_and(pred == 0, gt == 1).sum()
        fp = np.logical_and(pred == 1, gt == 0).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "recall": recall,
            "precision": precision,
        }
    
    def summarize(self):
        """
        전체 결과 반환: organ → list of dict
        DataFrame 등 저장에 바로 사용 가능
        organ별 평균/표준편차 요약 + 전체 df 병합 반환
        """
        
        summary = {}
        dfs = {}
        for organ, records in self.records.items():
            df = pd.DataFrame(records)
            dfs[organ] = df
            summary[organ] = df.describe().T[["mean", "std"]]  # 수치형 필드만
        return summary, dfs

    def summarize_scores(self):
        """
        organ → metric → 평균값
        """
        
        results = {}
        for organ, records in self.records.items():
            df = pd.DataFrame(records)
            results[organ] = {k: df[k].mean() for k in df.columns if df[k].dtype != "object"}
        return results


    def save_csv(self, output_dir):
        
        for organ, records in self.records.items():
            df = pd.DataFrame(records)
            df.to_csv(os.path.join(output_dir, f"{organ}_metrics.csv"), index=False)
