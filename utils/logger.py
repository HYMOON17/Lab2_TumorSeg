import os
import sys
import shutil
import logging
import json
import platform
import torch
import monai
import numpy
from tqdm import tqdm
import traceback
import hashlib
# 전역 중복 방지 플래그
LOGGER_NAME = "main_logger"
_logger = None  # 글로벌 변수처럼 동작

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            if msg:
                tqdm.write(msg)
                self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(log_dir: str, log_filename: str = "run.log"):
    global _logger

    if _logger is not None:
        return _logger  # 이미 설정됨
    
    os.makedirs(log_dir, exist_ok=True)
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    log_file_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 중복 출력 방지

    # 기존 핸들러 제거 (중복 방지)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # 파일 핸들러 추가 (기존 파일 확인 후 구분선 추가)
    if os.path.exists(log_file_path):
        with open(log_file_path, "a") as log_file:
            log_file.write("\n" + "="*50 + " NEW LOG SESSION " + "="*50 + "\n")
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))

        # tqdm 핸들러: INFO 이상의 로그만 기록
        tqdm_handler = TqdmLoggingHandler()
        tqdm_handler.setLevel(logging.INFO)
        tqdm_handler.setFormatter(logging.Formatter(log_format))

        logger.addHandler(file_handler)
        logger.addHandler(tqdm_handler)

    # Uncaught 예외 로깅
    def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
        # 에러 발생 파일, 함수명, 라인 번호 추출
        tb_summary = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # 가장 최근 트레이스백의 파일명과 라인번호 가져오기
        last_call = traceback.extract_tb(exc_traceback)[-1]
        file_name = last_call.filename
        line_number = last_call.lineno
        function_name = last_call.name

        # 상세한 에러 로그 기록
        logger.error(
            f"Uncaught exception in file '{file_name}', line {line_number}, in function '{function_name}': {exc_value}",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        logger.debug(f"Full traceback:\n{tb_summary}")

    sys.excepthook = log_uncaught_exceptions

    logger.info(f"Logging setup completed. Log file at: {log_file_path}")
    _logger = logger
    return logger

def get_logger():
    global _logger
    if _logger is None:
        raise RuntimeError("Logger not initialized. Call setup_logging() first.")
    return _logger

def save_config_as_json(config, log_dir):
    logger = get_logger()
    config_json_path = os.path.join(log_dir, "config.json")
    with open(config_json_path, 'w') as json_file:
        json.dump(config, json_file, indent=4)
    logger.info(f"Configuration saved to {config_json_path}")

# 날짜 기반 하위 폴더 생성
def generate_experiment_name(config):
    model_type = config['model_params']['type']
    dataset_name = config['data']['dataset_name']
    learning_rate = config['train_params']['learning_rate']
    batch_size = config['train_params']['batch_size']
    loss_type = config['train_params']['loss_type']
    patch_size = "x".join(map(str, config['model_params']['img_size']))
    experiment_name = f"{model_type}-{dataset_name}-lr{learning_rate}-bs{batch_size}-loss{loss_type}-patch{patch_size}"
    return experiment_name

def log_experiment_config():
    logger = get_logger()
    exp_config = {
        "python_version": platform.python_version(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "CPU",
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "torchvision_version": torch.__version__,
        "monai_version": monai.__version__,
        "numpy_version": numpy.__version__,
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }
    
    logger.info("Experiment Configuration:")
    for key, value in exp_config.items():
        logger.info(f"{key}: {value}")
        
        
def save_current_code(output_dir, current_script_path=None, extra_files=None):
    """
    현재 실행 중인 학습 스크립트와 추가 파일을 지정된 경로에 저장합니다.
    
    :param output_dir: 저장 경로
    :param current_script_path: 현재 실행 중인 학습 스크립트 경로 (명시적으로 전달)
    :param extra_files: 추가로 저장할 파일 경로 리스트 (기본값: None)
    """
    # 저장 경로 생성
    logger = get_logger()
    output_dir = os.path.join(output_dir,"save_code")
    os.makedirs(output_dir, exist_ok=True)
    
    # 현재 학습 스크립트 파일 저장
    if current_script_path:
        current_script_name = os.path.basename(current_script_path)
        current_script_save_path = os.path.join(output_dir, current_script_name)
        shutil.copy(current_script_path, current_script_save_path)
        logger.info(f"Saved current script: {current_script_save_path}")
    
    # 추가 파일 저장
    if extra_files:
        for file_path in extra_files:
            if os.path.isfile(file_path):
                extra_file_name = os.path.basename(file_path)
                extra_file_save_path = os.path.join(output_dir, extra_file_name)
                shutil.copy(file_path, extra_file_save_path)
                logger.info(f"Saved extra file: {extra_file_save_path}")
            else:
                logger.info(f"Extra file not found or invalid: {file_path}")

def save_current_code_wandb(output_dir, current_script_path=None, extra_files=None, log_to_wandb=False, config=None):
    """
    현재 실행 중인 학습 스크립트와 추가 파일을 지정된 경로에 저장합니다.
    선택적으로 wandb에도 코드 기록 가능.

    :param output_dir: 로컬 저장 경로
    :param current_script_path: 실행 중 스크립트 경로
    :param extra_files: 추가로 저장할 파일 리스트
    :param log_to_wandb: True일 경우 wandb에 코드 로그
    :param config: wandb 업로드 시 사용할 config (tmp_dir 필요)
    """
    logger = get_logger()
    save_dir = os.path.join(output_dir, "save_code")
    os.makedirs(save_dir, exist_ok=True)

    file_list = []

    # 현재 실행 파일 저장
    if current_script_path:
        dst_path = os.path.join(save_dir, os.path.basename(current_script_path))
        shutil.copy(current_script_path, dst_path)
        file_list.append(dst_path)
        logger.info(f"Saved current script: {dst_path}")

    # 추가 파일 저장
    if extra_files:
        for file_path in extra_files:
            if os.path.isfile(file_path):
                dst_path = os.path.join(save_dir, os.path.basename(file_path))
                shutil.copy(file_path, dst_path)
                file_list.append(dst_path)
                logger.info(f"Saved extra file: {dst_path}")
            else:
                logger.warning(f"Invalid extra file path: {file_path}")

    # ✅ wandb 코드 업로드
    if log_to_wandb and config is not None:
        import wandb
        import time

        tmp_dir = config['data']['tmp_dir']
        os.makedirs(tmp_dir, exist_ok=True)

        for file_path in file_list:
            shutil.copy(file_path, tmp_dir)
        wandb.run.log_code(tmp_dir)
        logger.info(f"W&B: Logged code from {tmp_dir}")
        time.sleep(5)  # 업로드 여유
        shutil.rmtree(tmp_dir)
        logger.info(f"W&B: Temporary code dir removed: {tmp_dir}")


def compute_experiment_hash(config: dict) -> str:
    """
    config 전체를 직렬화해서 SHA1 해시 생성
    """
    cfg_str = json.dumps(config, sort_keys=True)
    return hashlib.sha1(cfg_str.encode()).hexdigest()[:8]

'''

⚠️ 주의사항 및 개선 포인트 (중요도 순)

| 항목                                                   | 설명                                  | 권장 여부        |
| ---------------------------------------------------- | ----------------------------------- | ------------ |
| `compute_experiment_hash()` → 로그에 기록하는 라인 예시 추가해도 좋음 | 추적 편의성 향상                           | **optional** |


experiment_name 생성 함수 → 실험 hash 기반 자동 이름 생성 옵션도 고려 가능
save_current_code의 호출 위치 주의

CLI 기반이나 wandb 쓰는 경우, output_dir 설정 전 호출 시 경로 문제 생길 수 있음
→ 반드시 main 내부에서 output_dir 생성 후 호출

기능별 분할 고려 가능

현재 logger 관련 기능 + config 저장 + 코드 저장이 모두 들어가 있음
→ 시간이 지나면 logger.py, config_io.py, env_info.py 식으로 쪼개는 것도 추천 (지금은 하나로 OK)

'''