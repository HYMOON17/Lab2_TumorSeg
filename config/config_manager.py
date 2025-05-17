import os
import yaml
import socket
import torch

class EnvConfigManager:
    def __init__(self, config_path: str = None):
        # 1) 로드할 YAML 경로
        if config_path is None:
            ## 파일 위치에 따라 적절히 조정
            config_path = os.path.join(
                os.path.dirname(__file__),
                os.pardir, "config", "env_config.yaml"
            )
            # config_path = os.path.join(
            #     os.path.dirname(__file__),
            #     "env_config.yaml"
            # )
        with open(config_path, "r") as f:
            self._env = yaml.safe_load(f)

        # 2) 현재 서버 ID 감지
        self.server_id = int(self._detect_server())
        self.server_cfg = self._env.get("servers", {}).get(self.server_id, {})
        if not self.server_cfg:
            raise KeyError(
                f"❌ 서버 [{self.server_id}] 설정을 env_config.yaml에서 찾을 수 없습니다.\n"
                f"다음과 같이 설정되어야 합니다:\n\n"
                f"servers:\n  {self.server_id}:\n    root_dir: '/경로/...' ..."
            )

    def _detect_server(self) -> str:
        """서버 ID를 자동으로 감지 (IP 마지막 3자리 사용)"""
        try:
            # 외부 라우팅 IP 얻기
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip.split(".")[-1]  # 마지막 3자리
        except Exception:
            return "000"

    # def get(self, key: str, default=None):
    #     """서버 설정에서 키 조회, fallback 지원"""
    #     value = self.server_cfg.get(key, default)
    #     if value is None:
    #         raise KeyError(
    #             f"⚠️ '{key}'가 서버 [{self.server_id}] 설정에 없고, 기본값도 지정되지 않았습니다."
    #         )
    #     return value

    def get(self, key: str, default=None, allow_null: bool = False):
        #     """서버 설정에서 키 조회, fallback 지원"""
        value = self.server_cfg.get(key, default)

        if value is None:
            if allow_null:
                return ""
            raise KeyError(
                f"⚠️ '{key}'가 서버 [{self.server_id}] 설정에 없고, 기본값도 지정되지 않았습니다."
            )
        return value


    @property
    def project_root(self):
        return self.get("root_dir")

    @property
    def liver_dir(self):
        return self.get("liver_dir")

    @property
    def lung_dir(self):
        return self.get("lung_dir")
    
    @property
    def txt_dir(self):
        return self.get("txt_dir")

    @property
    def cache_root(self):
        return self.get("cache_dir", allow_null=True)  # ✅ null 허용
    
    @property
    def weight_path(self):
        return self.get("weight_path")
    
    @property
    def tmp_dir(self):
        return self.get("tmp_dir")

    def resolve_config(self, config: dict) -> dict:
        """
        전체 config dict 내부의 문자열 중 <...> 패턴이 포함된 값을 resolve()로 치환.
        """
        def resolve_recursive(obj):
            if isinstance(obj, str):
                if "<" in obj and ">" in obj:  # 토큰 포함된 문자열만 처리
                    return self.resolve(obj)
                else:
                    return obj
            elif isinstance(obj, dict):
                return {k: resolve_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_recursive(i) for i in obj]
            else:
                return obj

        return resolve_recursive(config)

    # 통합 경로 치환 함수
    def resolve(self, path: str) -> str:
        if not isinstance(path, str):
            raise TypeError(f"resolve(path): 문자열만 입력 가능합니다. 현재: {type(path)}")

        replacements = {
            "<root_dir>": self.project_root,
            "<cache_dir>": self.cache_root or "/tmp/cache",
            "<txt_dir>": self.txt_dir or "/tmp/txt",
            "<liver_dir>": self.liver_dir,
            "<lung_dir>": self.lung_dir,
            "<weight_path>": self.weight_path,
            "<tmp_dir>": self.tmp_dir or "/tmp",
        }

        for token, actual in replacements.items():
            if token in path:
                path = path.replace(token, actual)

        return path
    
    def resolve_root_dir(self, path: str) -> str:
        """'<root_dir>' 토큰을 실제 경로로 치환"""
        return path.replace("<root_dir>", self.data_root)

    def resolve_liver_dir(self, path: str) -> str:
        """'<liver_dir>' 토큰을 실제 경로로 치환"""
        return path.replace("<liver_dir>", self.cache_root)
    
    def resolve_lung_dir(self, path: str) -> str:
        """'<lung_dir>' 토큰을 실제 경로로 치환"""
        return path.replace("<lung_dir>", self.cache_root)
    
    def resolve_txt_dir(self, path: str) -> str:
        """'<txt_dir>' 토큰을 실제 경로로 치환"""
        return path.replace("<txt_dir>", self.cache_root)
    
    def resolve_cache_dir(self, path: str) -> str:
        """'<cache_dir>' 토큰을 실제 경로로 치환"""
        return path.replace("<cache_dir>", self.cache_root)
    
    def resolve_weight_path(self, path: str) -> str:
        """'<weight_path>' 토큰을 실제 경로로 치환"""
        return path.replace("<weight_path>", self.cache_root)
    
    def resolve_tmp_dir(self, path: str) -> str:
        """'<tmp_dir>' 토큰을 실제 경로로 치환"""
        return path.replace("<tmp_dir>", self.cache_root)

    def prepend_root_to_sys_path(self):
        """코드 어디서 실행하든 project_root를 sys.path에 추가"""
        import sys
        root = self.project_root
        if root not in sys.path:
            sys.path.append(root)

if __name__ == "__main__":
    import argparse
    import pprint
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.my_utils import load_config

    cfg_mgr = EnvConfigManager()
    parser = argparse.ArgumentParser(description="Debug config loading")

    parser.add_argument(
        '--config', type=str,
        default="config/exp.yaml",   # 실제 config 경로로 바꿔주세요
        help="Path to the YAML config file"
    )

    parser.add_argument(
        '--override', nargs='*', default=[],
        help="Override config parameters, e.g., train_params.batch_size=4"
    )

    args = parser.parse_args()
    
    config = load_config(args.config, overrides=args.override)
    config = cfg_mgr.resolve_config(config)  # 실제 경로로 치환 완료
    # print("\n🔧 Final Config (with overrides if given):")
    pprint.pprint(config)
