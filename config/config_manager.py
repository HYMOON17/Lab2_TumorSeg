import os
import yaml
import socket
import torch

class EnvConfigManager:
    def __init__(self, config_path: str = None):
        # 1) 로드할 YAML 경로
        if config_path is None:
            # 파일 위치에 따라 적절히 조정
            config_path = os.path.join(
                os.path.dirname(__file__),
                os.pardir, "config", "env_config.yaml"
            )
        with open(config_path, "r") as f:
            self._env = yaml.safe_load(f)

        # 2) 현재 서버 ID 감지
        self.server_id = self._detect_server()
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

    def get(self, key: str, default=None):
        """서버 설정에서 키 조회, fallback 지원"""
        value = self.server_cfg.get(key, default)
        if value is None:
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
        return self.get("cache_dir")
    
    @property
    def weight_path(self):
        return self.get("weight_path")
    
    @property
    def tmp_dir(self):
        return self.get("tmp_dir")

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
