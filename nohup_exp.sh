# #### Version 1
# #!/bin/bash

# # Swin UNETR 학습 실행 스크립트

# # Python 실행 파일 경로 (swin_unetr 환경의 Python 경로)
# PYTHON_PATH="/home/work/.conda/envs/swin_unetr/bin/python"

# # 학습 스크립트 경로
# SCRIPT_PATH="/home/work/hyungseok/Swin-UNETR/Experiments/Logs/SwinUnetr-Lung-lr1e-4-bs6-lossDiceCELoss-patch64x64x64/2025-01-08_08-11/save_code/Swin_UNETR_MSD_Lung_unsettle.py"
# # "/home/work/hyungseok/Swin-UNETR/notebook/Swin_UNETR_MSD_Liver_Lung_md.py"

# # Config 파일 경로
# CONFIG_PATH="/home/work/hyungseok/Swin-UNETR/Experiments/Logs/SwinUnetr-Lung-lr1e-4-bs6-lossDiceCELoss-patch64x64x64/2025-01-08_08-11/config.json"
# # "/home/work/hyungseok/Swin-UNETR/Experiments/Logs/SwinUnetr_multi_decoder-Liver_Lung-lr4e-4-bs3-lossDiceCELoss+Cont3-patch64x64x64/2024-12-16_01-03/config.json"
# # 14 "/home/work/hyungseok/Swin-UNETR/Experiments/Logs/SwinUnetr_multi_decoder-Liver_Lung-lr4e-4-bs3-lossDiceCELoss+Cont2-patch64x64x64/2024-12-15_15-34/config.json"
# # "/home/work/hyungseok/Swin-UNETR/Experiments/Logs/SwinUnetr_multi_decoder-Liver_Lung-lr4e-4-bs3-lossDiceCELoss+Cont3-patch64x64x64/2024-12-14_08-48/config.json"
# # "/home/work/hyungseok/Swin-UNETR/Experiments/Logs/SwinUnetr_multi_decoder-Liver_Lung-lr4e-4-bs3-lossDiceCELoss+Cont2-patch64x64x64/2024-12-15_05-21/config.json"
# # "/home/work/hyungseok/Swin-UNETR/Experiments/Logs/SwinUnetr_multi_decoder-Liver_Lung-lr4e-4-bs3-lossDiceCELoss+Cont3-patch64x64x64/2024-12-14_08-48/config.json"
# # "/home/work/hyungseok/Swin-UNETR/Experiments/Logs/SwinUnetr_multi_decoder-Liver_Lung-lr4e-4-bs3-lossDiceCELoss+Cont2-patch64x64x64/2024-12-14_09-08/config.json"
# # "/home/work/hyungseok/Swin-UNETR/Experiments/Logs/SwinUnetr_multi_decoder-Liver_Lung-lr4e-4-bs3-lossDiceCELoss+Cont2-patch64x64x64/2024-12-13_06-59/config.json"
# # "/home/work/hyungseok/Swin-UNETR/Experiments/Logs/SwinUnetr_multi_decoder-Liver_Lung-lr4e-4-bs3-lossDiceCELoss+Cont3-patch64x64x64/2024-12-13_06-58/config.json"
# # "/home/work/hyungseok/Swin-UNETR/api/debug3.yaml"
# # "/home/work/hyungseok/Swin-UNETR/api/debug.yaml"

# # 로그 저장 경로: Config 파일 경로의 폴더
# LOG_DIR=$(dirname "$CONFIG_PATH")

# # 로그 파일 및 PID 파일 경로
# LOG_FILE="$LOG_DIR/nohup.log"
# PID_FILE="$LOG_DIR/pid.txt"

# # nohup 명령으로 실행
# nohup "$PYTHON_PATH" "$SCRIPT_PATH" --config "$CONFIG_PATH" > "$LOG_FILE" 2>&1 &

# # PID 저장
# echo $! > "$PID_FILE"
# echo "Experiment started. Logs: $LOG_FILE, PID: $(cat $PID_FILE)"

# # 10초 대기 (로그 파일 생성 시간 확보)
# echo "Waiting for log file to be created..."
# sleep 10

# # 실시간 로그 확인
# echo "Displaying logs. Press Ctrl+C to stop viewing logs without stopping the experiment."
# tail -f "$LOG_FILE"




######################################3
### Version 2 resume과 일반 나눠서 확인깔쥐

#!/bin/bash

# Swin UNETR 학습 실행 스크립트

# Python 실행 파일 경로 (swin_unetr 환경의 Python 경로)
PYTHON_PATH="/home/hsmoon/anaconda3/envs/swin_unetr/bin/python"

# 학습 스크립트 경로
SCRIPT_PATH="/data/hyungseok/Swin-UNETR/Experiments/Logs/ContrastiveSwinUNETR-Post_Liver_Lung-lr1e-4-bs1-lossDiceCELoss+Cont1-patch64x64x64/2025-05-13_02-38/save_code/train.py"
# "/data/hyungseok/Swin-UNETR/notebook/tempp/temppp_0401/persistent_post_lung.py"
# "/data/hyungseok/Swin-UNETR/Experiments/Logs/SwinUnetr-DEBUG_Liver_Lung-lr1e-4-bs3-lossDiceCELoss-patch64x64x64/2025-04-03_10-43/save_code/persistent_check_iter_classy.py"
# "/data/hyungseok/Swin-UNETR/Experiments/Logs/BalancedContrastiveSwinUNETR-Liver_Lung-lr1e-4-bs1-lossDiceCELoss+Cont1-patch64x64x64/2025-02-05_17-30/save_code/Swin_UNETR_MSD_Liver_Lung_bcl.py"
# "/data/hyungseok/Swin-UNETR/notebook/Swin_UNETR_MSD_Liver_Lung_bcl.py"
# "/data/hyungseok/Swin-UNETR/notebook/ver2/Swin_UNETR_MSD_Liver_Lung_md.py"
# "/data/hyungseok/Swin-UNETR/notebook/Swin_UNETR_MSD_Liver_sol.py"
# "/data/hyungseok/Swin-UNETR/notebook/Swin_UNETR_MSD_Lung_sol.py"
            
# Config 파일 경로
CONFIG_PATH="/data/hyungseok/Swin-UNETR/Experiments/Logs/ContrastiveSwinUNETR-Post_Liver_Lung-lr1e-4-bs1-lossDiceCELoss+Cont1-patch64x64x64/2025-05-13_02-38/config.json"
# "/data/hyungseok/Swin-UNETR/Experiments/Logs/ContrastiveSwinUNETR-Post_Liver_Lung-lr1e-4-bs1-lossDiceCELoss+Cont1-patch64x64x64/2025-04-25_21-23/config.json"
# "/data/hyungseok/Swin-UNETR/Experiments/Logs/SwinUnetr-Post_Liver_Lung-lr1e-4-bs2-lossTverskyLoss-patch64x64x64/2025-04-16_21-44/config.json"
# "/data/hyungseok/Swin-UNETR/api/lung.yaml"
# "/data/hyungseok/Swin-UNETR/Experiments/Logs/SwinUnetr-DEBUG_Liver_Lung-lr1e-4-bs3-lossDiceCELoss-patch64x64x64/2025-04-03_10-43/config.json"
# "/data/hyungseok/Swin-UNETR/Experiments/Logs/BalancedContrastiveSwinUNETR-Liver_Lung-lr1e-4-bs1-lossDiceCELoss+Cont1-patch64x64x64/2025-02-05_17-30/config.json"
# "/data/hyungseok/Swin-UNETR/api/debug2.yaml"
# "/data/hyungseok/Swin-UNETR/api/exp.yaml"
# "/data/hyungseok/Swin-UNETR/api/liver.yaml"

# Resume 여부 설정
RESUME=true

# 실행 시간 생성
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

if $RESUME; then
    # resume_config_check.py를 통해 경로 및 설정 검사
    $PYTHON_PATH /data/hyungseok/Swin-UNETR/utils/resume_config_check.py "$SCRIPT_PATH" "$CONFIG_PATH"
    if [[ $? -ne 0 ]]; then
        echo "Error: Resume configuration validation failed."
        exit 1
    fi

    # Log 디렉토리 설정 (resume 기준)
    LOG_DIR=$(dirname "$CONFIG_PATH")
    echo "Resuming experiment in: $LOG_DIR"
else
    # 새로운 실험 경로 생성
    LOG_DIR="/data/hyungseok/Swin-UNETR/api/background/Experiment_$TIMESTAMP"
    mkdir -p "$LOG_DIR"
    echo "Starting new experiment in: $LOG_DIR"
fi

# 로그 파일 및 PID 파일 경로
LOG_FILE="$LOG_DIR/nohup.log"
PID_FILE="$LOG_DIR/pid.txt"

# nohup 명령으로 실행
nohup "$PYTHON_PATH" "$SCRIPT_PATH" --config "$CONFIG_PATH" >> "$LOG_FILE" 2>&1 &

# 로그 파일에 실행 구분선 추가
echo "========================================================" >> "$LOG_FILE"
echo "Experiment started at $(date +"%Y-%m-%d %H:%M:%S")" >> "$LOG_FILE"
echo "Script: $SCRIPT_PATH" >> "$LOG_FILE"
echo "Config: $CONFIG_PATH" >> "$LOG_FILE"
echo "========================================================" >> "$LOG_FILE"

# PID 저장
echo $! > "$PID_FILE"
echo "Experiment started. Logs: $LOG_FILE, PID: $(cat $PID_FILE)"

# 10초 대기 (로그 파일 생성 시간 확보)
echo "Waiting for log file to be created..."
sleep 10

# 실시간 로그 확인
echo "Displaying logs. Press Ctrl+C to stop viewing logs without stopping the experiment."
tail -f "$LOG_FILE"