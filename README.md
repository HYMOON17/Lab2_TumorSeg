## Acknowledgement

This project references code from the following repository:

- [Swin-UNETR](https://github.com/LeonidAlekseev/Swin-UNETR)
  - License: MIT License
  - The model architecture from this repository was used and modified.
  Licensed under the MIT License.

Additional modifications and original contributions in this repository are licensed under the Apache License 2.0.

# 🧠 Swin-UNETR 기반 다기관 종양 분할 실험

본 프로젝트는 Swin UNETR 기반 모델을 확장하여 간/폐 종양에 대한 다양한 실험을 진행하는 연구 코드입니다.

- **실험 목적**: 단일 모델 기반 multi-organ segmentation 성능 향상
- **주요 내용**: Contrastive learning, Attention mechanism, Query-based 구조 비교
- **목표**: 재현성, 확장성, 실험 자동화를 고려한 모듈화

---

## 🗂️ 폴더 구조

```
.
├── code/                   # train.py, test_main.py 등 실행 파이프라인 모듈
├── models/                 # 다양한 Swin UNETR 변형 모델 정의
├── docs/                   # 실험 진행 절차, naming rule, 기록 위치 정리등 각종 루틴 명문화
├── losses/                 # Segmentation + Contrastive loss 정의
├── utils/                  # 로깅, 시드 설정, 스케줄러 등 공용 유틸
├── config/                 # 실험 공통 설정파일 디렉토리
├── api/                    # 실험별 개별 yaml 구성 파일
├── Experiments/            # 실험 결과, 로그 저장
├── data_cache/             # MONAI PersistentDataset 캐시 경로
├── nohup_exp.sh            # 터미널에서 백그라운드로 자동화 스크립트
└── environment.yml         # Conda 환경 설정 파일
```

---

## ⚙️ 실험 실행 방법

```bash
# Conda 환경 준비
conda env create -f environment.yml
conda activate swin_unetr

# 학습 실행
python code/train.py --config api/exp_cont.yaml 

# 테스트 실행
python code/test_main.py --config api/exp_cont.yaml --override test_params.ckpt_dir="" cuda.CUDA_VISIBLE_DEVICES="[6]"
```

---

## 🧪 실험 관리 전략

- 실험 설정으로 `api/exp_*.yaml` 구성
- 실험 로그는 `Experiments/Logs/<실험명>/날짜/` 구조로 저장
- 실험 결과는 `Experiments/Models/<실험명>/날짜/` 구조로 저장
- `wandb`를 통한 실험 로깅 + `npz`, `png` 저장 포함
- `.gitignore`로 중간 결과 제외하여 깔끔한 버전 관리
- 그외 세부적 사항은 `/docs`의 내용 참고

---

## 📌 Git 운영 규칙

- `main` 브랜치는 정제된 코드만 반영
- 실험별 브랜치는 필요시 분기

---

## 🙋‍♂️ 작성자

문형석 / 인공지능 석사과정  
📧 fbkevin@g.skku.edu  
🔗 https://github.com/HYMOON17
