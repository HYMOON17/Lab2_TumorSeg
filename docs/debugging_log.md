# debugging.md의 목적과 역할
단순 로그가 아닌, 반복적으로 발생할 수 있는 에러 상황에 대한 해결 레퍼런스입니다.
즉, 실제 발생한 문제 → 원인 분석 → 해결법 → 재현 조건을 문서화해두는 백과사전 같은 존재입니다.

## 핵심 원칙 3가지:
원칙	설명
① 문제 중심 정리	날짜별 나열이 아니라 유형별 또는 에러 유형 중심으로 정리
② 재현 가능성 명시	"언제든 다시 생길 수 있는 문제인가?" → 체크 필수
③ 빠르게 찾을 수 있어야 함	핵심은 “디버깅 속도 단축” → 검색과 읽기 쉬운 구조로 작성

## 📌 디버깅 작성 포맷 템플릿

```markdown
## [이슈 명칭] - YYYY-MM-DD

### 현상 요약
...

### 원인 분석
...

### 해결 방안
...

### 재현 조건 / 테스트 환경
...

### 관련 커밋 / 태그 
예시
- commit: `abc123_fix_sw_infer`
- tag: `exp_2025_05_14_fix_sw_infer`


---


## 🔧 주요 이슈 카탈로그

### 1. [학습 속도 저하 (SlidingWindow + 병렬 test)] 2025-05-14
- **현상**: GPU는 idle인데 epoch 속도 급격히 느려짐
- **주요 원인**: sliding_window_inference GPU 점유, CPU 후처리 병목, cache_dir 충돌 등
- **해결**: 

### 2. PersistentDataset hang 이슈
- **현상**: test 실행 시 진행이 멈추거나 무한 대기
- **원인**: 여러 process가 동일 `cache_dir` 접근
- **해결**: 병렬 실행 말고 순차 실행

### 3. Zombie 프로세스 누적
- **현상**: GPU는 비어있는데 학습 시작 안 됨
- **원인**: 이전 테스트 종료 시 `defunct` process 잔류
- **해결**: `ps -ef | grep defunct`, `kill -9`

## \[wandb-core 충돌로 인한 학습 중단] - 2025-05-15

### 현상 요약

* 학습이 중단되며 터미널에 `fatal error: concurrent map iteration and map write` 메시지 출력
* wandb 내부에서 `BrokenPipeError`, goroutine 관련 로그가 다량 발생
* GPU 사용률은 낮고, 학습 프로세스는 멈춘 상태에서 종료되지 않음

### 원인 분석

* wandb는 내부적으로 `wandb-core`라는 시스템 모니터링 프로세스를 실행함
* `wandb-core`는 GPU/CPU 사용률을 실시간으로 수집하여 dashboard에 표시함
* 이 모듈은 Go 언어 기반의 멀티스레드 프로그램이며, GPU 상태를 주기적으로 polling함
* 문제는 Go의 map 자료구조에 대해 **동시 read/write가 허용되지 않음**
* 따라서 하나의 쓰레드가 GPU 상태를 읽는 중 다른 쓰레드가 값을 쓰면 **충돌이 발생**
* wandb가 병렬로 실행되거나, 하나의 GPU에 여러 wandb 세션이 동시에 접근할 경우 이 현상이 빈번하게 발생
* 특히 서버 환경에서 다른 사용자가 동일 GPU를 사용 중일 때 취약함

### 해결 방안

1. **wandb 시스템 모니터링 기능 비활성화**
   → GPU 모니터링 자체를 꺼서 wandb-core 충돌 방지

```python
import os
os.environ["WANDB_DISABLE_SYSTEM"] = "true"
```

2. **병렬 wandb 실행 시 주의 사항 적용**

   * `WANDB_RUN_GROUP`, `run.name` 등을 명시적으로 분리해 세션 충돌 회피

3. **wandb 버전 업그레이드**

   * 해당 버그가 wandb 최신 버전에서 해결됐을 수 있으므로 다음 명령으로 업그레이드

```bash
pip install --upgrade wandb
```

4. **로그 기반 추가 분석 가능**

   * `.wandb` 디렉토리 내 `debug.log`, `wandb-service.log` 참고해 wandb-core 충돌 여부 확인

### 재현 조건 / 테스트 환경

* 환경: A100 서버, PyTorch + MONAI 실험, wandb 0.15.x \~ 0.16.x
* test\_main.py 또는 train.py에서 wandb 활성화
* GPU 모니터링을 포함한 시스템 리소스 수집이 켜져 있음
* 동일 GPU에서 복수 wandb 세션 존재 (ex. 병렬 test/train or 다중 사용자)

### 관련 커밋 / 태그

* commit: `hotfix_disable_wandb_monitor`
* tag: `exp_2025_05_15_fix_wandbcore`
