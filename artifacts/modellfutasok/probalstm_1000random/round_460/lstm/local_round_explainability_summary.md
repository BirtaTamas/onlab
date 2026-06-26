# Local Round Explainability

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-3dmax-vs-mibr-bo3-O12tFfVag47APQdKBJkGZl/3dmax-vs-mibr-m2-ancient-p3.csv`
- round_num: `11`

## Largest probability jumps

- tick `107559`, seconds `74.00`, LSTM `0.9074`, delta `+0.2157`
- tick `106407`, seconds `56.00`, LSTM `0.9052`, delta `+0.1519`
- tick `106823`, seconds `62.50`, LSTM `0.8150`, delta `-0.1228`
- tick `107399`, seconds `71.50`, LSTM `0.7414`, delta `-0.1175`
- tick `106311`, seconds `54.50`, LSTM `0.7299`, delta `+0.0870`
- tick `106535`, seconds `58.00`, LSTM `0.9516`, delta `+0.0805`
- tick `106247`, seconds `53.50`, LSTM `0.6234`, delta `+0.0704`
- tick `106375`, seconds `55.50`, LSTM `0.7533`, delta `+0.0618`
- tick `105319`, seconds `39.00`, LSTM `0.6005`, delta `-0.0575`
- tick `105287`, seconds `38.50`, LSTM `0.6580`, delta `+0.0560`

## Top 15 local ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002062`, |coef| `0.002062`
- `lag_00__kill_diff_last_3s`: coefficient `0.002012`, |coef| `0.002012`
- `lag_07__CT_place_TUNNEL`: coefficient `0.001940`, |coef| `0.001940`
- `lag_00__CT_kills_last_3s`: coefficient `0.001901`, |coef| `0.001901`
- `lag_15__CT_place_TSIDELOWER`: coefficient `-0.001886`, |coef| `0.001886`
- `lag_05__CT_place_TUNNEL`: coefficient `-0.001828`, |coef| `0.001828`
- `lag_09__CT_place_TUNNEL`: coefficient `0.001814`, |coef| `0.001814`
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001692`, |coef| `0.001692`
- `lag_07__CT_place_WATER`: coefficient `-0.001623`, |coef| `0.001623`
- `lag_03__CT_place_TSIDEUPPER`: coefficient `0.001510`, |coef| `0.001510`
- `lag_00__CT_place_TSPAWN`: coefficient `0.001428`, |coef| `0.001428`
- `lag_04__CT_place_TUNNEL`: coefficient `-0.001255`, |coef| `0.001255`
- `lag_12__CT_place_WATER`: coefficient `0.001242`, |coef| `0.001242`
- `lag_02__T_place_TSIDELOWER`: coefficient `-0.001227`, |coef| `0.001227`
- `lag_02__T5__duck_amount`: coefficient `0.001203`, |coef| `0.001203`

## Top 10 utility ridge features

- `lag_10__T_B_site_active_infernos`: coefficient `-0.000681` (lowers CT win probability)
- `lag_00__CT3__flash_duration`: coefficient `0.000599` (raises CT win probability)
- `lag_00__T3__smoke`: coefficient `-0.000584` (lowers CT win probability)
- `lag_00__T1__smoke`: coefficient `-0.000580` (lowers CT win probability)
- `lag_12__CT_active_infernos`: coefficient `-0.000572` (lowers CT win probability)
- `lag_10__T_active_infernos`: coefficient `-0.000514` (lowers CT win probability)
- `lag_03__CT3__flash_duration`: coefficient `-0.000508` (lowers CT win probability)
- `lag_00__T1__utility_total`: coefficient `-0.000504` (lowers CT win probability)
- `lag_07__T_B_site_active_infernos`: coefficient `-0.000495` (lowers CT win probability)
- `lag_05__T3__smoke`: coefficient `-0.000490` (lowers CT win probability)

## Top 10 non-utility ridge features

- `lag_00__CT_shots_fired_sum`: coefficient `0.002062` (raises CT win probability)
- `lag_00__kill_diff_last_3s`: coefficient `0.002012` (raises CT win probability)
- `lag_07__CT_place_TUNNEL`: coefficient `0.001940` (raises CT win probability)
- `lag_00__CT_kills_last_3s`: coefficient `0.001901` (raises CT win probability)
- `lag_15__CT_place_TSIDELOWER`: coefficient `-0.001886` (lowers CT win probability)
- `lag_05__CT_place_TUNNEL`: coefficient `-0.001828` (lowers CT win probability)
- `lag_09__CT_place_TUNNEL`: coefficient `0.001814` (raises CT win probability)
- `lag_00__CT_place_TSIDEUPPER`: coefficient `0.001692` (raises CT win probability)
- `lag_07__CT_place_WATER`: coefficient `-0.001623` (lowers CT win probability)
- `lag_03__CT_place_TSIDEUPPER`: coefficient `0.001510` (raises CT win probability)

## Largest Jump Contribution Breakdown


### tick `107559`, seconds `74.00`, LSTM delta `+0.2157`

Top all feature movements:
- `lag_07__CT_place_TUNNEL`: contribution `+0.031159`
- `lag_05__CT_place_TUNNEL`: contribution `+0.029362`
- `lag_09__CT_place_TUNNEL`: contribution `+0.029132`
- `lag_15__CT_place_TSIDELOWER`: contribution `+0.025626`
- `lag_07__CT_place_WATER`: contribution `+0.009863`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `106407`, seconds `56.00`, LSTM delta `+0.1519`

Top all feature movements:
- `lag_03__CT_place_TSIDEUPPER`: contribution `+0.011350`
- `lag_01__CT_shots_fired_sum`: contribution `+0.006111`
- `lag_00__CT_kills_last_3s`: contribution `+0.005489`
- `lag_00__kill_diff_last_3s`: contribution `+0.004843`
- `lag_02__T_place_TSIDELOWER`: contribution `+0.004598`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `106823`, seconds `62.50`, LSTM delta `-0.1228`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `-0.012722`
- `lag_00__CT_shots_fired_sum`: contribution `-0.007163`
- `lag_00__kill_diff_last_3s`: contribution `-0.004843`
- `lag_08__CT5__shots_fired`: contribution `-0.004004`
- `lag_01__CT_shots_fired_sum`: contribution `+0.003820`

Top utility-only movements:
- `lag_00__CT3__flash_duration`: contribution `-0.003142`
- `lag_03__CT3__flash_duration`: contribution `-0.002666`
- `lag_10__T_B_site_active_infernos`: contribution `-0.001924`

### tick `107399`, seconds `71.50`, LSTM delta `-0.1175`

Top all feature movements:
- `lag_04__CT_place_TUNNEL`: contribution `-0.020162`
- `lag_10__CT_place_TSIDELOWER`: contribution `-0.016284`
- `lag_07__CT_place_WATER`: contribution `-0.009863`
- `lag_13__CT_place_TSIDELOWER`: contribution `-0.009831`
- `lag_02__CT_place_TUNNEL`: contribution `-0.009014`

Top utility-only movements:
- No utility movement among the top local contributors.

### tick `106311`, seconds `54.50`, LSTM delta `+0.0870`

Top all feature movements:
- `lag_00__CT_place_TSIDEUPPER`: contribution `+0.012722`
- `lag_00__CT_shots_fired_sum`: contribution `+0.005730`
- `lag_02__T_place_TSIDELOWER`: contribution `+0.004598`
- `lag_14__CT3__duck_amount`: contribution `+0.003285`
- `lag_14__CT2__duck_amount`: contribution `+0.002218`

Top utility-only movements:
- No utility movement among the top local contributors.
