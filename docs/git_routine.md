# AI 석사과정 개인연구용 Git 운영 규칙

## 🔰 운영 목적

* 실험 코드 변경 이력 추적
* GitHub는 공유 및 백업 용도
* 최신 실험 코드는 항상 로컬이 우선

---

## ✅ 1. 작업 전 기본 점검

```bash
git status
```

* 현재 로컬 작업 내용 확인
* 수정된 파일이 많을 경우, 반드시 커밋 단위로 논리적으로 묶을 항목을 정리

---

## ✅ 2. 커밋 규칙 (기본 브랜치: `main`)

```bash
# 전체 또는 일부만 선택적으로 stage
git add file1.py file2.yaml

# 또는 전체 stage
git add .

# 커밋
git commit -m "[모듈명] 한줄 설명 (ex: [train] loss 추가 및 config 수정)"
```

* 커밋 메시지 규칙:

  * `[폴더/기능명] 핵심 변경 요약`
  * 예: `[loader] Liver/Lung 분기 구조 개선`, `[tsne] 분석 함수 디버깅`

---

## ✅ 3. 커밋 후 푸시

```bash
git push origin main
```

* 푸시는 하루 1회 이상 정기적으로 수행
* 실험 로그나 노트 정리와 함께 푸시하는 루틴 권장

---

## ✅ 4. 외부에서 GitHub 수정 시 (예: 웹에서 README 수정)

```bash
git fetch origin
git checkout -b github_version origin/main
git diff main...github_version     # 변경 비교
git checkout main
git checkout github_version -- <원하는 파일>
git branch -D github_version       # 임시 브랜치 제거
```

* 가져온 파일 반영 후 커밋 & 푸시

---

## ✅ 5. 기타 규칙

* 브랜치는 현재 `main`만 사용

  * 향후 대규모 리팩토링이나 협업 전까지는 단일 브랜치 유지
* `.ipynb`, `*.ckpt`, `output/`, `wandb/` 등은 `.gitignore`로 관리
* 중요한 실험은 `docs/progress.md`나 `README.md`에 실험 목적/의도 기록

---

## 📝 커밋 메시지 작성 가이드

### 1. 커밋 메시지 구조

```
<타입>(<스코프>): <간결한 설명>

[변경의 이유와 상세 설명]

[관련 이슈나 참고 링크]
```

* **타입**:

  * `feat`: 새로운 기능 추가
  * `fix`: 버그 수정
  * `docs`: 문서 관련 변경
  * `style`: 코드 포맷팅, 세미콜론 누락 등
  * `refactor`: 코드 리팩토링 (기능 변화 없음)
  * `test`: 테스트 추가 또는 수정
  * `chore`: 빌드 업무, 패키지 매니저 설정 등

* **스코프**: 변경된 파일이나 모듈 명시 (예: `loader`, `train`, `README`)

* **간결한 설명**: 50자 이내로 요약

* **본문**: 72자 이내 줄바꿈 기준으로 상세 변경 이유 작성

* **푸터**: 관련 이슈 번호나 참고 링크

### 2. 작성 예시

```
feat(loader): Add data augmentation for training

- Implemented random horizontal and vertical flips
- Improved model generalization on validation set

Resolves: #42
```

### 3. 작성 원칙

* 명령형 현재 시제 사용: "Add", "Fix", "Update" 등
* 50/72 규칙 준수
* "무엇"보다 "왜" 중심 설명
* 전체 커밋 메시지 스타일 일관성 유지

### 4. AI 도구 활용

* `aicommits`와 같은 AI 기반 도구로 커밋 메시지 초안 작성 가능
* 항상 직접 검토하여 실험 의도와 맥락을 반영할 것

---

## 📌 운영 목표

* 커밋 하나하나가 "실험의 진화 단위"가 되도록 한다
* GitHub에는 항상 의미 있는 상태로 정리된 코드만 남긴다
* 불확실할 땐 작은 단위로 커밋, 정리는 나중에 `rebase`나 `squash`로 정제
