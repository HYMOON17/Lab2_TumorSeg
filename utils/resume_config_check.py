import json
import os
import sys

def validate_config(script_path, config_path):
    try:
        # 1차 검사: SCRIPT_PATH와 CONFIG_PATH의 상위 경로 비교
        script_parent = os.path.dirname(os.path.dirname(script_path))
        config_parent = os.path.dirname(config_path)
        if script_parent != config_parent:
            print(f"Error: SCRIPT_PATH와 CONFIG_PATH의 상위 경로가 일치하지 않습니다.")
            print(f"SCRIPT_PATH 상위 경로: {script_parent}")
            print(f"CONFIG_PATH 상위 경로: {config_parent}")
            return False

        # 2차 검사: config.json 내부 값 확인
        with open(config_path, 'r') as f:
            config = json.load(f)

        train_params = config.get('train_params', {})
        resume = train_params.get('resume', False)
        resume_path = train_params.get('resume_path', '')

        if not resume:
            print("Error: 'resume' is not set to true in config.json.")
            return False

        if not os.path.isfile(resume_path):
            print(f"Error: 'resume_path' does not point to a valid file: {resume_path}")
            return False

        print("Config validation passed.")
        return True
    except Exception as e:
        print(f"Error while validating config: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python resume_config_check.py <script_path> <config_path>")
        sys.exit(1)

    script_path = sys.argv[1]
    config_path = sys.argv[2]

    if not os.path.isfile(script_path):
        print(f"Error: Script file not found at {script_path}")
        sys.exit(1)

    if not os.path.isfile(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    if not validate_config(script_path, config_path):
        sys.exit(1)
