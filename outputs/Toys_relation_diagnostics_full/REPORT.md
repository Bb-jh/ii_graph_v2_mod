# Toys 관계 진단

분석 간선: 79,332, 상품: 47,520. 기존 파일은 변경하지 않았습니다.

## 1. 행동 간선과 무작위 상품 쌍

공통 유효 상품 47,520개에서 서로 다른 무방향 상품 쌍 100,000개를 균등 추출했습니다.
무작위 쌍은 알려진 비관련/음성 정답이 아닙니다. 결과는 선택된 행동 간선과의 기술적 비교이며 유의성 검정이 아닙니다.

| 모달리티 | 행동 평균 | 무작위 평균 | P(행동 > 무작위) + 0.5 P(동점) |
|---|---:|---:|---:|
| text | 0.8088 | 0.6189 | 0.8339 |
| image | 0.6843 | 0.5027 | 0.8674 |

## 2. Consistency 유무 비교

간선, 정규화 행동 점수 B, 임베딩 점수, λ는 고정했습니다.
- mean: W = (1-λ)B + λS
- consistency: W = (1-λ)B + λS(1-|t-i|)
- 저장 점수 재현 최대 오차: 0
- consistency 추가 감소량: 평균 0.038880, 중앙값 0.036848, 최대 0.136263

| 비교 (기준 → 후보) | 단독 1위 변경률 | 엄격한 이웃 쌍 순위 역전율 |
|---|---:|---:|
| behavior_vs_mean | 14.45% | 7.87% |
| behavior_vs_consistency | 23.66% | 12.27% |
| mean_vs_consistency | 21.00% | 12.21% |

| 가중치 | 평균 |
|---|---:|
| behavior | 0.237142 |
| mean | 0.389963 |
| consistency | 0.351084 |

단독 1위 변경률은 이웃이 2개 이상이고 기준 방식의 1위가 동점이 아닌 상품만 분모로 사용합니다.
Behavior 기준 두 비교는 같은 분모이며, mean 기준 비교는 분모가 다릅니다. 정확한 개수는 diagnostics.json에 있습니다.
현재 consistency 항은 mean보다 각 간선 점수를 낮추거나 유지합니다. 중요한 것은 절대 감소뿐 아니라 이웃 간 차등 감소에 따른 순위 변화입니다.

### 기존 사례의 변화

| 기준 상품 ID | 이웃 | Behavior 순위 | Mean 순위 | Consistency 순위 |
|---|---|---:|---:|---:|
| 1 | Ticket To Ride - Europe. | 6 | 2 | 2 |
| 1 | Pandemic Board Game. | 2 | 3 | 3 |
| 1 | Ticket to Ride: USA 1910 Expansion. | 1 | 1 | 1 |
| 5 | Jenga Classic Game. | 2 | 2 | 1 |
| 5 | Sorry! 2013 Edition Game. | 1 | 1 | 3 |
| 5 | Trouble Game. | 3 | 3 | 2 |
| 83 | Nerf Zombie Strike Hammershot Blaster. | 4 | 4 | 2 |
| 83 | Official Nerf N-Strike Elite Series 30-Dart Refill. | 2 | 2 | 3 |
| 83 | Nerf N-Strike Elite Dart Refill Pack (75 Darts). | 1 | 1 | 1 |
| 83 | Nerf N-Strike Elite Retaliator. | 3 | 3 | 4 |

동점 이웃은 엄격한 역전 계산에서 제외합니다. Top-10 및 사례 표시에는 상품 ID로 동점을 해소하므로 해당 결과는 동점 처리에 영향을 받습니다.
두 방식의 점수 차이는 penalty의 효과를 분리하지만, 추천/quantization 품질의 우열은 평가하지 않습니다.

## 산출물

- similarity_distributions.png / .pdf: 원시 유사도 히스토그램·누적분포
- consistency_ablation.png / .pdf: 가중치 분포와 추가 penalty
- diagnostics.json: 설정·검증·전체 수치
- ablation_edges.csv: 간선별 세 점수
- ranks_*.csv: 상품별 순위 변화
- case_rank_changes.csv: 기존 사례의 전체 이웃 순위
- largest_consistency_penalties.csv: 추가 감소량 상위 100개
