# 수정 메모 — Multimodal Behavior Reweighting

이번 수정에서는 기존의 `Text graph + Image graph + Behavior graph` 전체 fusion보다, 현재 연구 흐름에 맞는 **Behavior-backbone multimodal reweighting**을 명시적으로 구현했습니다.

## 추가/수정

- `graph/multimodal_reweight.py` 추가
  - Behavior edge set 고정
  - Text/Image similarity를 [0,1] evidence로 변환
  - semantic strength + cross-modal consistency 계산
  - multimodal reliability `M_ij` 계산
  - `W_ij = (1-λ)B_ij + λM_ij` soft reweight
- `main.py`
  - `multimodal_behavior` 전략에서는 full Text/Image KNN graph 생성을 생략
  - Behavior candidate edge에 대해서만 semantic score 계산
- `config.py`
  - `lambda_multimodal`, `reliability_mode`, `behavior_normalization`, `missing_modality` 등 추가
- `run_toys_multimodal.py` 추가
  - 기존 Toys `.train.inter` 실험 흐름에서 multimodal reweight까지 한 번에 실행 가능
- `requirements.txt`
  - `scipy` 명시적으로 추가

## 기본 reliability

```text
semantic_strength = mean(text, image)
consistency = 1 - abs(text - image)
M = semantic_strength * consistency
```

기본 Behavior normalization은 `log_max`, λ는 `0.3`입니다.
이 값들은 최종 방법이 아니라 실험을 위한 기본값이므로 ablation/sweep 대상으로 두었습니다.
