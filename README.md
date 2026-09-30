# I-I Graph Experimentation Framework — Multimodal Behavior Reweighting

현재 버전은 **Behavior I-I graph를 backbone으로 고정한 뒤 Text/Image evidence로 edge weight만 soft reweight**하는 실험을 바로 할 수 있도록 수정한 버전입니다.

핵심 흐름은 다음과 같습니다.

```text
U-I interaction
   ↓
Behavior I-I (co-occurrence → min_cooc → Top-K → union/mutual)
   ↓  edge set E_B freeze
   ├─ Text cosine similarity on E_B
   └─ Image cosine similarity on E_B
                ↓
      Multimodal reliability M_ij
                ↓
W_ij = (1-λ) B_ij + λ M_ij
                ↓
Multimodal-aware Behavior I-I Graph
```

## 이번 수정의 핵심

### 1. Behavior topology를 고정

`multimodal_behavior` 모드에서는 Text/Image가 새로운 edge를 만들거나 기존 edge를 제거하지 않습니다.
Behavior graph가 **어떤 item pair를 relation으로 볼지** 결정하고, multimodal 정보는 **그 relation의 strength만 조정**합니다.

### 2. 전체 Text/Image KNN graph를 만들지 않음

기존 코드처럼 전체 item pair를 대상으로 semantic KNN graph를 만들 필요 없이, 이미 선택된 Behavior edge `E_B`에 대해서만 cosine similarity를 계산합니다.

따라서 Toys처럼 item 수가 큰 경우에도 semantic full-KNN의 O(N²) 계산을 피할 수 있습니다.

### 3. Multimodal reliability

기본 설정은 해석하기 쉬운 실험식입니다.

```text
text_sim_01  = clip(cos_text,  0, 1)
image_sim_01 = clip(cos_image, 0, 1)

semantic_strength = (text_sim_01 + image_sim_01) / 2
consistency       = 1 - |text_sim_01 - image_sim_01|

M_ij = semantic_strength × consistency
```

즉 Text/Image similarity가 둘 다 높고 서로 비슷한 판단을 할수록 reliability가 커집니다.

> 이 수식은 DA-MRS의 cross-modal consistency 아이디어를 참고한 **실험용 구현 후보**이며, DA-MRS 수식을 그대로 재현했다고 주장하지 않습니다.

### 4. Behavior reweighting

기본값은:

```text
B_ij = log1p(behavior_score) / max(log1p(behavior_score))
W_ij = (1 - λ) B_ij + λ M_ij
λ = 0.3
```

`min-max` 대신 `log_max`를 기본으로 둬서 낮은 co-occurrence edge가 무조건 0이 되는 문제와 long-tail scale 문제를 완화했습니다.

### 5. Missing modality 처리

기본값 `behavior_only`에서는 Text/Image 중 하나가 없으면 해당 edge를 제거하거나 0점 처리하지 않고, 원래 Behavior score를 그대로 사용합니다.

---

# 가장 바로 쓰기 좋은 실행 파일

기존 `run_toys_behavior.py`와 같은 `.train.inter`, `.item.json` 구조를 쓰는 경우:

```bash
python run_toys_multimodal.py \
  --train ../data/Toys/Toys.train.inter \
  --items ../data/Toys/Toys.item.json \
  --embedding-npz /path/to/Toys_embeddings.npz \
  --min-cooc 2 \
  --top-k 20 \
  --symmetrize mutual \
  --lambda-mm 0.3 \
  --output outputs/Toys_multimodal_mutual
```

Embedding NPZ는 다음 key를 사용합니다.

```text
text_embeddings      # [N, D_text]
image_embeddings     # [N, D_image]
item_ids              # optional, 없으면 row index == item id로 간주
text_valid_mask       # optional
image_valid_mask      # optional
```

## 주요 산출물

```text
behavior_graph.npz
multimodal_behavior_graph.npz
edge_evidence.csv
graph_analysis.json
case_study.json
top_edges_preview.csv
run_config.json
```

`edge_evidence.csv`에는 edge별로 다음 값이 남습니다.

```text
cooccurrence_count
behavior_score
normalized_behavior_score
text_similarity
image_similarity
text_similarity_01
image_similarity_01
semantic_strength
cross_modal_consistency
multimodal_reliability
final_score
```

---

# 모듈형 pipeline (`main.py`)

`config.py` 기본값도 새 실험에 맞게 설정되어 있습니다.

```python
cfg.fusion.strategy = "multimodal_behavior"
cfg.fusion.lambda_multimodal = 0.3
cfg.fusion.reliability_mode = "strength_x_consistency"
cfg.fusion.cosine_transform = "clip01"
cfg.fusion.behavior_normalization = "log_max"
cfg.fusion.missing_modality = "behavior_only"
```

이 모드에서는:

1. collaborative Behavior graph 생성
2. Text/Image embedding 준비
3. Behavior edge에 대해서만 Text/Image cosine 계산
4. multimodal reliability 계산
5. 동일한 topology에 final weight 부여

순서로 동작합니다.

기존 `separate`, `semantic_backbone`, `behavior_backbone`, `union` 전략도 ablation/비교용으로 남겨두었습니다.

---

# 파일 구조

```text
ii_graph_v2/
├── config.py
├── data_loader.py
├── evidence.py
├── main.py
├── run_toys_behavior.py
├── run_toys_multimodal.py          # NEW
│
├── embeddings/
├── graph/
│   ├── collaborative.py
│   ├── semantic.py
│   ├── denoise.py
│   ├── fusion.py
│   └── multimodal_reweight.py      # NEW
│
├── analysis/
└── hierarchy/
```

## 아직 연구적으로 확정하지 않은 부분

- λ의 최종 값 (`0.1/0.3/0.5` 등 sweep 필요)
- multimodal reliability의 최종 수식
- cosine transform 방식 (`clip01` vs `shift01`)
- behavior normalization (`log_max`, degree-aware 등)
- 최종 graph에서 Global–Local–Fine hierarchy를 추출하는 방식

즉 이번 수정은 **“Behavior relation을 Text/Image consistency로 soft reweight해 multimodal-aware graph를 만든다”**는 단계까지 안정적으로 실험하기 위한 코드입니다.
