import random
import numpy as np
import os
import torch
from monai.utils import set_determinism
from typing import Dict
import torch.backends.cudnn as cudnn

def set_seed_and_env(config: Dict):
    set_determinism(seed=config['train_params']['seed']) # 5개 융합
    torch.manual_seed(config['train_params']['seed'])
    torch.cuda.manual_seed(config['train_params']['seed'])
    torch.cuda.manual_seed_all(config['train_params']['seed'])
    random.seed(config['train_params']['seed'])
    np.random.seed(config['train_params']['seed'])
    cudnn.benchmark = config['cuda']['benchmark']
    cudnn.deterministic = config['cuda']['deterministic']
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config['cuda']['CUDA_VISIBLE_DEVICES']))

    return

def worker_init_fn(worker_id):
    seed = 1234 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # GPU에서도 동일한 시드 설정



def set_all_random_states(transform_list, seed=1234):
    """
    MONAI 변환 리스트의 모든 랜덤 변환(`RandomizableTransform` 상속)에 대해 시드를 설정하는 함수.

    Args:
        transform_list: `Compose()` 내 변환 리스트
        seed: 고정할 시드 값
    """
    for t in transform_list:
        if hasattr(t, "set_random_state"):
            t.set_random_state(seed)
            print(f"✅ Transform `{t.__class__.__name__}` 에 시드 {seed} 적용 완료.")
        else:
            print(f"⚠️ Transform `{t.__class__.__name__}` 는 랜덤성이 없거나 `set_random_state()`를 지원하지 않음.")


'''
| 항목                                         | 설명                                                                                                          | 조치                    |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------- | --------------------- |
| **`set_seed_and_env()` 호출 위치**             | `main.py`의 **최상단에서 바로 호출**해야 효과가 있음 (torch/np/random 등 import 후 초기 실행 전에)                                   | ✅ 이미 잘 하고 있다면 문제 없음   |
| **`worker_init_fn`이 `DataLoader`에 연결되었는가** | `num_workers > 0`일 때, 이걸 `DataLoader(..., worker_init_fn=worker_init_fn)`으로 **반드시 명시해야** worker마다 seed가 고정됨 | ✅ 반드시 연결 필요           |
| **`set_determinism()` 부작용 인식**             | 이 함수는 `torch.backends.cudnn.deterministic=True` 등을 **강제 설정**함. 속도가 느려질 수 있음 → 실험 성능 디버깅 시에는 꺼도 됨            | ⚠️ 필요 시 config에서 토글   |
| **GPU 수와 설정 일치 확인**                        | `CUDA_VISIBLE_DEVICES` 설정이 실제 사용 환경과 불일치할 경우 오류 발생 → 로컬/서버 병행 시 특히 주의                                       | ⚠️ `.env` 파일로 관리해도 좋음 |
| **MONAI의 transform에 시드 주는 방식**             | `Compose([...])` 안의 `RandomizableTransform`들은 `set_all_random_states()` 호출 시점이 중요함 → Compose **직후에** 실행해야 함 | ✅ 위치만 잘 지키면 완벽        |

set_all_random_states()는 train_transforms.transforms처럼 내부 리스트에 직접 접근해야 동작함.
시드 값 자체도 나중에 로깅에 기록하면 좋음 (e.g. logger.info(f"Seed used: {seed}")).
torch.manual_seed() 등을 따로 또 호출하지 않아도 set_determinism()이 대체로 커버함. 다만 너처럼 명시적으로 중복 설정하는 건 좋은 습관.
'''