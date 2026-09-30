# 별도 관계 진단 실행

기존 그래프 생성 코드와 원본 결과는 변경하지 않습니다. 인코더 추론, CLIP 다운로드,
학습, 분포 보정은 수행하지 않습니다. Python 3.10 이상을 사용하세요.

## 설치

```powershell
python -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r requirements-diagnostics.txt
```

## 2번: consistency만 제거한 비교

```powershell
.\.venv-analysis\Scripts\python.exe run_relation_diagnostics.py
```

기본 입력은 `outputs/Toys_multimodal_mutual`, 출력은 별도 `outputs/Toys_relation_diagnostics`입니다.
`edge_evidence.csv`의 정규화 행동 점수와 원시 유사도를 사용하며 `run_config.json`의
λ, 변환, 결측 처리 설정을 그대로 적용합니다. 저장된 최종 점수가 재현되는지 먼저 검증합니다.

- Behavior: 저장된 정규화 행동 점수 B
- Mean: `(1-λ) B + λ (t+i)/2`
- Consistency: `(1-λ) B + λ (t+i)/2 * (1-|t-i|)`

동점은 엄격한 순위 역전 집계에서 제외합니다. 표시용 Top-K는 상품 ID 오름차순으로
동점을 해소합니다. 상품별 degree가 K 이하이면 Top-K 집합은 변하지 않으므로,
집합 유지율 요약은 degree > K인 상품만 대상으로 합니다.

## 1번과 2번 함께: 원본 MACRec 임베딩 제공

```powershell
.\.venv-analysis\Scripts\python.exe run_relation_diagnostics.py `
  --text-npy "D:\data\Toys\Toys.emb-llama-td.npy" `
  --image-npy "D:\data\Toys\Toys.emb-ViT-L-14.npy" `
  --random-pairs 100000 --seed 20261001 `
  --output outputs/Toys_relation_diagnostics_full
```

NPY는 행 인덱스가 MACRec numeric item ID이고 상품 수와 행 수가 정확히 같아야 합니다.
경로는 실제 데이터 위치로 바꾸세요. 두 배열의 차원은 달라도 됩니다.

또는 아래 옵션 하나를 사용합니다.

```powershell
.\.venv-analysis\Scripts\python.exe run_relation_diagnostics.py `
  --embedding-npz "D:\data\Toys_embeddings.npz" `
  --output outputs/Toys_relation_diagnostics_full
```

NPZ 필수 키: `text_embeddings`, `image_embeddings`.
선택 키: `item_ids`, `text_valid_mask`, `image_valid_mask`.
`item_ids`가 있으면 numeric ID에 맞춰 재정렬합니다. 보안을 위해 object/pickle 배열은
읽지 않으므로 ID는 정수 또는 NumPy 문자열 배열이어야 합니다.

두 모달리티 모두 유효하고 유한한 비영벡터가 있는 상품들에서 **서로 다른 무방향
상품 쌍을 균등하게, 중복 없이** 추출합니다. 기본 100,000쌍이며 가능한 쌍이 더 적으면
전체를 사용합니다. 행동 간선도 추출 대상에 포함하며 포함 개수를 기록합니다.
무작위 쌍은 비관련 정답/negative label이 아닙니다. 기본 비교는 인기도를 맞춘 비교가
아니므로 행동 선택 편향이 있습니다. 상품 ID 0을 임의로 padding이라 간주하지 않습니다.

입력 임베딩으로 기존 행동 간선의 코사인을 다시 계산해 저장된 값과 비교합니다.
최대 절대 오차가 1e-4를 초과하면 ID 정렬 또는 모델 불일치로 간주하고 중단합니다.
무작위 분포와 비교하는 행동 표본도 양쪽 모달리티가 유효한 동일 조건으로 제한합니다.

## 산출물과 해석

- `REPORT.md`, `diagnostics.json`: 실행 범위, 수치와 검증 정보
- `similarity_distributions.png/.pdf`: 원시 코사인 히스토그램, ECDF
- `consistency_ablation.png/.pdf`: 세 가중치 분포와 추가 penalty
- `ablation_edges.csv`: 간선별 점수 비교
- `ranks_*.csv`: 상품별 순위 변화
- `case_rank_changes.csv`: 기존 사례의 전체 이웃 순위. 원본 사례에 없는 상품 제목은 빈칸
- `largest_consistency_penalties.csv`: 추가 감소량 상위 100개
- `random_pair_scores.csv`: 임베딩 제공 시에만 생성

원본 임베딩이 없으면 2번과 행동 간선 분포만 실행하며 1번은 명시적으로 미실행 처리합니다.
확률형 분포 비교 수치는 `P(행동 점수 > 무작위 점수) + 0.5 P(동점)`이며 추천 AUC가 아닙니다.
동일 상품을 공유하는 쌍이 있어 독립 표본 가정의 p-value를 제시하지 않습니다.
추천 성능과 branch guidance의 우열은 이 분석으로 판단하지 않습니다.
각 실행은 별도 출력 경로를 권장합니다. 같은 출력 경로의 진단 파일은 덮어씁니다.

## 검증

```powershell
.\.venv-analysis\Scripts\python.exe -m unittest discover -s tests -p test_relation_diagnostics.py -v
```

알려진 수치의 penalty, 결측 처리, 동점 제외, 무작위 표본 재현성, ID 재정렬,
0벡터 제외, 잘못된 임베딩 매핑 거부를 확인합니다. 합성 데이터는 테스트용이며
실제 Toys 무작위 비교 결과로 사용하지 않습니다.
