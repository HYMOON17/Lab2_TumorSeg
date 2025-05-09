"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
from monai.data import set_track_meta

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
